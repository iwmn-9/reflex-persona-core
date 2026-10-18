import hashlib,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from tools.run_owned_value import run


class HorizonStudyTests(unittest.TestCase):
    def test_same_host_pair_matches_independent_worlds_and_keeps_profile_identity(self):
        from tools.run_horizon_pair import run_pair
        jobs=[['gambling',3,52994,'outside',3,1,tail,0] for tail in (1,2)]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'pair'
            mismatch=[list(j) for j in jobs];mismatch[1][2]+=1
            with self.assertRaisesRegex(ValueError,'same world'):run_pair(root,mismatch,'diagnostic')
            self.assertFalse(root.exists())
            proof=run_pair(root,jobs,'diagnostic')
            self.assertTrue(proof['complete']);self.assertTrue(proof['same_source'])
            self.assertEqual(proof['profile'],'ego')
            for job in jobs:
                child=root/f'h{job[6]+1}'
                self.assertEqual(json.loads((child/'pair_execution.json').read_text()),proof)
                actual=json.loads((child/'trajectories.jsonl').read_text())
                expected=run(job,'receding_recovery')
                actual.pop('seconds');expected.pop('seconds')
                self.assertEqual(actual,expected)
            with self.assertRaises(FileExistsError):run_pair(root,jobs,'diagnostic')

    def test_offline_accounts_are_zero_mean_over_exhaustive_public_events(self):
        from itertools import product
        from tools.continuing_games import ContinuingGame,GAMES
        from tools.run_owned_value import law
        from tools.summarize_horizon_study import realized_account
        from reflex.laboratory import PROFILES
        for game in GAMES:
            for variant in (0,1):
                g=ContinuingGame(game,PROFILES[0],3,variant)
                residual=bellman=mass=0.;expected_score=conditional_score=0.
                for symbols in product(('0','1','2'),repeat=3):
                    w=g.start();trace=[];probability=1.
                    for tick,symbol in enumerate(symbols):
                        truth=law(tick,3,'shift-return');probability*=truth[symbol]
                        keys=g.keys(w);key=keys[(tick+int(w['cash']))%len(keys)]
                        after,_=g.transition(w,key,symbol)
                        trace.append(dict(tick=tick,before=w,after=after,action=key,truth=truth))
                        w=after
                    row=dict(game=game,profile='growth',seed=variant,turns=3,regime='shift-return',
                             trace=trace,score=g.goal(w).value)
                    before=json.dumps(row,sort_keys=True);account=realized_account(row)
                    self.assertEqual(json.dumps(row,sort_keys=True),before)
                    mass+=probability;residual+=probability*account['residual']
                    expected_score+=probability*row['score']
                    conditional_score+=probability*(account['initial']+account['conditional_increments'])
                    if game=='auction':bellman+=probability*account['bellman']['residual']
                self.assertAlmostEqual(mass,1.,places=12)
                self.assertAlmostEqual(residual,0.,places=12)
                self.assertAlmostEqual(bellman,0.,places=12)
                self.assertAlmostEqual(expected_score,conditional_score,places=12)

    def test_frozen_resume_keeps_experience_and_rejects_wrong_source(self):
        from unittest.mock import patch
        from tools.freeze_experiment import dispatch
        from tools.resume_horizon_study import resume
        job=['gambling',1,52993,'outside',3,1,2,0]
        expected=run(job,'receding_recovery',tail_recovery_limit=64)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'study'
            with patch('tools.freeze_experiment.subprocess.run'):
                dispatch(root,'run_horizon_study.py',[])
            program='''import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from tools.run_horizon_study import __doc__ as semantics
from tools.run_owned_value import run
root=Path(sys.argv[2]);job=json.loads(sys.argv[3])
registration=dict(jobs=[job],arms=['receding_recovery'],world_seeds=[job[2]],stage='diagnostic',semantics=semantics,configuration={'tail_recovery_limit':64})
(root/'registration.json').write_text(json.dumps(registration))
assert run(job,'receding_recovery',checkpoint=root/'checkpoint.json',max_decisions=1,tail_recovery_limit=64) is None
'''
            subprocess.run([sys.executable,'-c',program,str(root/'_source'),str(root),json.dumps(job)],check=True,capture_output=True)
            manifest=hashlib.sha256((root/'source_snapshot.json').read_bytes()).hexdigest()
            before=(root/'checkpoint.json').read_bytes()
            with self.assertRaisesRegex(AssertionError,'wrong frozen source'):resume(root,'0'*64,job)
            source=root/'_source/reflex/core.py';original=source.read_bytes();source.write_bytes(original+b'\n')
            with self.assertRaisesRegex(AssertionError,'source changed'):resume(root,manifest,job)
            source.write_bytes(original);self.assertEqual((root/'checkpoint.json').read_bytes(),before)
            proof=resume(root,manifest,job)
            self.assertEqual(proof['resume_at'],1);self.assertTrue(proof['original_prefix_unchanged'])
            actual=json.loads((root/'trajectories.jsonl').read_text());actual.pop('seconds');expected.pop('seconds')
            self.assertEqual(actual,expected)
            with self.assertRaisesRegex(AssertionError,'already complete'):resume(root,manifest,job)

    def test_read_only_instrumentation_reproduces_saved_choice(self):
        from tools.diagnose_owner_horizon import diagnose,prefix_choice
        job=['combat',1,47,'outside',3,1,2,0]
        row=run(job,'receding_recovery')
        original=json.dumps(row,sort_keys=True)
        result=diagnose(row,0,2)
        self.assertEqual(result['selected_action'],row['trace'][0]['action'])
        self.assertEqual(json.dumps(row,sort_keys=True),original)
        self.assertEqual(result['capped_paths'],0)
        self.assertEqual(sum(result['controller_counts'].values()),
            sum(len(path) for d in result['distributions'] for path in d['paths'].values()))
        json.dumps(result,allow_nan=False)
        full=prefix_choice(row,0,result,3)
        self.assertEqual(full['action'],result['selected_action'])
        shortened=prefix_choice(row,0,result,2)
        separate=diagnose(row,0,1)
        self.assertEqual(shortened['action'],separate['selected_action'])
        for plan,value in separate['forecast']['purpose_mean_estimator']['roots'].items():
            key=separate['forecast']['plans'][plan]['root']
            self.assertAlmostEqual(shortened['purpose'][key],value['unprojected'],places=12)
        legacy=json.loads(json.dumps(result));legacy.pop('children')
        self.assertEqual(prefix_choice(row,0,legacy,2)['action'],shortened['action'])

    def test_frozen_checkpointed_study_preserves_execution_and_rejects_overwrite(self):
        job=['resources',2,43,'outside',3,1,1,0]
        base=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'study'
            args=['--root',str(root),'--job',json.dumps(job),'--stage','diagnostic','--recovery-limit','64']
            subprocess.run([sys.executable,str(base/'tools/run_horizon_study.py'),*args],check=True,capture_output=True)
            actual=json.loads((root/'trajectories.jsonl').read_text())
            reference=run(job,'receding_recovery',tail_recovery_limit=64)
            actual.pop('seconds');reference.pop('seconds')
            self.assertEqual(actual,reference)
            audit=json.loads((root/'audit.json').read_text())
            self.assertTrue(audit['complete']);self.assertTrue(audit['source_freeze_verified'])
            self.assertEqual(audit['checks']['restored_action_feedback'],3)
            before=(root/'trajectories.jsonl').read_bytes()
            repeated=subprocess.run([sys.executable,str(root/'_source/tools/run_horizon_study.py'),*args,'--frozen'],capture_output=True)
            self.assertNotEqual(repeated.returncode,0)
            self.assertEqual((root/'trajectories.jsonl').read_bytes(),before)


if __name__=='__main__':unittest.main()
