import hashlib,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from tools.run_owned_value import run


class HorizonStudyTests(unittest.TestCase):
    def test_independent_batch_preserves_partial_worlds_and_fresh_execution(self):
        from unittest.mock import patch
        from tools.freeze_experiment import dispatch
        from tools.run_horizon_batch import run_batch
        jobs=[[g,p,52990+p,'outside',3,1,1+p%2,0]
              for g,p in (('gambling',1),('resources',3),('combat',0),('auction',2))]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'batch';specs=[];prefixes={}
            for index,job in enumerate(jobs):
                spec=dict(key=f'case-{index}',job=job);specs.append(spec)
                if index>=2:continue
                child=root/f'study{index}'
                with patch('tools.freeze_experiment.subprocess.run'):dispatch(child,'run_horizon_study.py',[])
                program='''import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from tools.run_horizon_study import __doc__ as semantics
from tools.run_owned_value import run
root=Path(sys.argv[2]);job=json.loads(sys.argv[3])
(root/'registration.json').write_text(json.dumps(dict(jobs=[job],arms=['receding_recovery'],world_seeds=[job[2]],stage='diagnostic',semantics=semantics)))
assert run(job,'receding_recovery',checkpoint=root/'checkpoint.json',max_decisions=int(sys.argv[4])) is None
'''
                subprocess.run([sys.executable,'-c',program,str(child/'_source'),str(child),json.dumps(job),str(index+1)],check=True,capture_output=True)
                spec['resume_manifest']=hashlib.sha256((child/'source_snapshot.json').read_bytes()).hexdigest()
                prefixes[index]=json.loads((child/'checkpoint.json').read_text())['payload']['trace']
            source=Path(__file__).resolve().parents[1]
            result=run_batch(root,specs,source,'diagnostic')
            self.assertTrue(result['complete']);self.assertTrue(result['same_source'])
            for index,job in enumerate(jobs):
                child=root/f'study{index}';actual=json.loads((child/'trajectories.jsonl').read_text())
                expected=run(job,'receding_recovery');actual.pop('seconds');expected.pop('seconds')
                self.assertEqual(actual,expected)
                if index in prefixes:
                    self.assertTrue(json.loads((child/'resume_provenance.json').read_text())['original_prefix_unchanged'])
                    self.assertEqual(len(prefixes[index]),index+1)
            with self.assertRaisesRegex(FileExistsError,'fresh batch'):run_batch(root,specs,source,'diagnostic')
            # Original partial artifacts remain beside audited continuations;
            # collect each registered world once, without dropping the partials.
            import shutil
            from tools.summarize_horizon_study import summarize
            initial=root/'initial';continued=root/'continued'
            (initial/'horizon-case-0').mkdir(parents=True)
            for index,spec in enumerate(specs):shutil.copytree(root/f'study{index}',continued/('horizon-'+spec['key']))
            plan=dict(retained_baselines=[],new_jobs=specs,stage='diagnostic')
            summary=summarize(source,initial,plan,additional_roots=(continued,))
            self.assertTrue(summary['complete']);self.assertEqual(summary['collected_runs'],4)
            with self.assertRaisesRegex(AssertionError,'duplicate completed'):
                summarize(source,initial,plan,additional_roots=(continued,continued))

    def test_combat_reference_matches_exhaustive_rules_with_death_and_reloading(self):
        from tools.combat_reference import reference
        from tools.continuing_games import ContinuingGame
        from tools.run_owned_value import law
        from reflex.laboratory import PROFILES
        for end,start in ((3,0),(10,7)):
            g=ContinuingGame('combat',PROFILES[0],end)
            value,choices=reference(end,lambda t:tuple(law(t,end,'shift-return').values()))
            def brute(w):
                if g.terminal(w):return g.goal(w).value
                return max(sum(p*brute(g.transition(w,k,z)[0]) for z,p in law(w['tick'],end,'shift-return').items()) for k in g.keys(w))
            for hp in (0.,.25,1.,11.75,12.):
                for ammo in (0,1,3):
                    w=g.start();w.update(tick=start,hp=hp,ammo=ammo)
                    self.assertAlmostEqual(value(w),brute(w),places=12)
                    q=choices(w);self.assertEqual(set(q),set(g.keys(w)))
                    for k in q:
                        expected=sum(p*brute(g.transition(w,k,z)[0]) for z,p in law(start,end,'shift-return').items())
                        self.assertAlmostEqual(q[k],expected,places=12)

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
                    if 'bellman' in account:bellman+=probability*account['bellman']['residual']
                self.assertAlmostEqual(mass,1.,places=12)
                self.assertAlmostEqual(residual,0.,places=12)
                self.assertAlmostEqual(bellman,0.,places=12)
                self.assertAlmostEqual(expected_score,conditional_score,places=12)

    def test_frozen_resume_keeps_experience_and_rejects_wrong_source(self):
        from unittest.mock import patch
        from tools.freeze_experiment import dispatch
        from tools.resume_horizon_study import resume
        job=['gambling',1,52993,'outside',3,1,2,0]
        expected=run(job,'receding_recovery',tail_recovery_limit=64,model_seeds=(801,802),sampling_standard_errors=2.)
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
registration=dict(jobs=[job],arms=['receding_recovery'],world_seeds=[job[2]],stage='diagnostic',semantics=semantics,configuration={'tail_recovery_limit':64,'model_seeds':[801,802],'sampling_standard_errors':2.})
(root/'registration.json').write_text(json.dumps(registration))
assert run(job,'receding_recovery',checkpoint=root/'checkpoint.json',max_decisions=1,tail_recovery_limit=64,model_seeds=(801,802),sampling_standard_errors=2.) is None
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
        self.assertEqual(result['seeds'],[701,702,703])
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
        alternate=diagnose(row,0,1,seeds=(911,))
        self.assertEqual(alternate['seeds'],[911])
        self.assertEqual({s for d in alternate['distributions'] for s in d['paths']},{'911'})
        from tools.diagnose_owner_horizon import pool_diagnostics
        other=diagnose(row,0,1,seeds=(912,));together=diagnose(row,0,1,seeds=(911,912))
        pooled=prefix_choice(row,0,pool_diagnostics([alternate,other]),2)
        direct=prefix_choice(row,0,together,2)
        self.assertEqual(pooled,direct)
        with self.assertRaisesRegex(ValueError,'overlapping'):pool_diagnostics([alternate,alternate])
        mismatched=json.loads(json.dumps(other));mismatched['owner_digest']='different owner'
        with self.assertRaisesRegex(ValueError,'identical owner'):pool_diagnostics([alternate,mismatched])
        from tools.audit_continuation_returns import audit_run
        self.assertEqual(audit_run(row),audit_run(row,ticks=(0,)))
        self.assertEqual([x['tick'] for x in audit_run(row,ticks=(1,))['cases']],[1])

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
