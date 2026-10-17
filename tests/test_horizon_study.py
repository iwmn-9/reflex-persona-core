import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from tools.run_owned_value import run


class HorizonStudyTests(unittest.TestCase):
    def test_frozen_checkpointed_study_preserves_execution_and_rejects_overwrite(self):
        job=['resources',2,43,'outside',3,1,1,0]
        base=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'study'
            args=['--root',str(root),'--job',json.dumps(job),'--stage','diagnostic']
            subprocess.run([sys.executable,str(base/'tools/run_horizon_study.py'),*args],check=True,capture_output=True)
            actual=json.loads((root/'trajectories.jsonl').read_text())
            reference=run(job,'receding_recovery')
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
