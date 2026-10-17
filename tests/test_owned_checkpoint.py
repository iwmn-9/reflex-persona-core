import json,tempfile,unittest
from pathlib import Path
from tools.owned_checkpoint import encoded,sha
from tools.run_owned_value import run


def without_time(row):return {k:v for k,v in row.items() if k!='seconds'}


class OwnedCheckpointTests(unittest.TestCase):
    def test_recovery_budget_survives_resume_and_cannot_change_mid_world(self):
        job=('gambling',1,41907,'outside',3,1,2,0)
        expected=run(job,'receding_recovery',tail_recovery_limit=64)
        with tempfile.TemporaryDirectory() as d:
            checkpoint=Path(d)/'execution.json'
            self.assertIsNone(run(job,'receding_recovery',checkpoint=checkpoint,max_decisions=1,tail_recovery_limit=64))
            original=checkpoint.read_bytes()
            with self.assertRaisesRegex(ValueError,'contract changed'):
                run(job,'receding_recovery',checkpoint=checkpoint,max_decisions=1)
            self.assertEqual(checkpoint.read_bytes(),original)
            actual=run(job,'receding_recovery',checkpoint=checkpoint,tail_recovery_limit=64)
            self.assertEqual(without_time(actual),without_time(expected))

    def test_planning_learning_and_world_rng_survive_multiple_pauses(self):
        job=('gambling',2,41901,'shift-return',4,1,1,0)
        expected=run(job,'receding_recovery')
        with tempfile.TemporaryDirectory() as d:
            checkpoint=Path(d)/'execution.json'
            self.assertIsNone(run(job,'receding_recovery',checkpoint=checkpoint,max_decisions=1))
            self.assertIsNone(run(job,'receding_recovery',checkpoint=checkpoint,max_decisions=2))
            actual=run(job,'receding_recovery',checkpoint=checkpoint,max_decisions=8)
            self.assertEqual(without_time(actual),without_time(expected))
            # Reading an already completed checkpoint must not start a new world.
            restored=run(job,'receding_recovery',checkpoint=checkpoint,max_decisions=1)
            self.assertEqual(without_time(restored),without_time(expected))

    def test_changed_job_runtime_source_and_corrupted_state_are_rejected(self):
        job=('resources',0,41901,'outside',3,1,1,0)
        with tempfile.TemporaryDirectory() as d:
            checkpoint=Path(d)/'execution.json'
            self.assertIsNone(run(job,'reflex_segmented',checkpoint=checkpoint,max_decisions=1))
            original=checkpoint.read_bytes()
            with self.assertRaisesRegex(ValueError,'contract changed'):
                run(job,'reflex',checkpoint=checkpoint,max_decisions=1)
            for changed in ('job','runtime','sources','checksum'):
                with self.subTest(changed=changed):
                    envelope=json.loads(original);payload=envelope['payload']
                    if changed=='job':payload['contract']['job'][2]+=1
                    elif changed=='runtime':payload['contract']['runtime']['numpy']='different'
                    elif changed=='sources':payload['contract']['sources']['reflex/core.py']='different'
                    else:payload['world']['cash']+=1
                    if changed!='checksum':envelope['sha256']=sha(encoded(payload))
                    checkpoint.write_bytes(encoded(envelope))
                    with self.assertRaises(ValueError):
                        run(job,'reflex_segmented',checkpoint=checkpoint,max_decisions=1)
                    checkpoint.write_bytes(original)

    def test_slices_require_a_real_checkpoint_and_positive_budget(self):
        job=('resources',0,41901,'outside',3,1,1,0)
        with self.assertRaisesRegex(ValueError,'persistent checkpoint'):
            run(job,'reflex_segmented',max_decisions=1)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'positive decision slice'):
                run(job,'reflex_segmented',checkpoint=Path(d)/'execution.json',max_decisions=0)


if __name__=='__main__':unittest.main()
