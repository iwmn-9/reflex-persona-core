import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from tools.run_owned_value import run


class HorizonStudyTests(unittest.TestCase):
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
