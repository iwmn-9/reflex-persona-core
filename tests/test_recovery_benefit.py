import json
from pathlib import Path
import tempfile
import unittest

from tools.owned_checkpoint import encoded, sha
from tools.recovery_benefit import run_branch, truth_after, load_origin
from tools.run_adverse_recovery import run
from tools.run_recovery_campaign import retain_origin


class RecoveryBenefitTests(unittest.TestCase):
    def test_same_origin_control_restore_and_resume(self):
        spec = dict(id='branch-test',mechanism='world',game='combat',profile=1,
                    seed=601,regime='outside',turns=3)
        registration = dict(plan_sha256='test',source_sha256='test')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIsNone(run(root/'prefix',spec,'expected',max_decisions=1,
                registration=registration,continuation_scope='shared'))
            origin = root/'prefix/checkpoint.json'
            origin_record = json.loads(origin.read_bytes())['payload']
            recovered_origin=root/'recovered.json'
            retain_origin(origin,recovered_origin,spec,1)
            retain_origin(origin,recovered_origin,spec,1)
            self.assertEqual(origin.read_bytes(),recovered_origin.read_bytes())
            rows = {}
            for mode in ('changed','steady'):
                for experience in ('learning','frozen'):
                    rows[mode,experience] = run_branch(root/(mode+experience),spec,origin,
                        mode,experience,registration,boundary=1,window=2)
                    result = rows[mode,experience]
                    self.assertTrue(result['complete'] and result['window_complete'])
                    self.assertEqual(result['actual_steps'],2)
                    self.assertEqual(result['trace'][:1],origin_record['trace'])
                    self.assertEqual(result['origin_rng'],origin_record['rng'])
            for mode in ('changed','steady'):
                learned,frozen = rows[mode,'learning'],rows[mode,'frozen']
                self.assertEqual([t['uniform'] for t in learned['trace'][1:]],
                                 [t['uniform'] for t in frozen['trace'][1:]])
                payload=json.loads((root/(mode+'frozen')/'checkpoint.json').read_bytes())['payload']
                self.assertEqual(payload['owner']['predictor'],origin_record['owner']['predictor'])
                self.assertEqual(payload['owner']['last_tick'],2)
            self.assertIsNone(run_branch(root/'split',spec,origin,'changed','frozen',registration,
                max_decisions=1,boundary=1,window=2))
            resumed=run_branch(root/'split',spec,origin,'changed','frozen',registration,boundary=1,window=2)
            for key in ('trace','final','final_owner','final_rng','checks','metrics'):
                self.assertEqual(resumed[key],rows['changed','frozen'][key],key)
            with self.assertRaises(ValueError):
                run_branch(root/'split',spec,origin,'changed','learning',registration,boundary=1,window=2)
            bad=Path(tmp)/'bad.json'; bad.write_bytes(origin.read_bytes().replace(b'"tick":1',b'"tick":2'))
            with self.assertRaises(AssertionError):load_origin(bad,spec,1)
            death=json.loads(origin.read_bytes()); death['payload']['world']['hp']=0
            death['sha256']=sha(encoded(death['payload'])); bad.write_bytes(encoded(death))
            retain_origin(bad,root/'dead-origin.json',spec,1)
            self.assertFalse((root/'dead-origin.json').exists())

    def test_intervention_is_event_category_based(self):
        spec=dict(turns=128,regime='outside')
        self.assertEqual(truth_after(spec,32,32,'steady'),{'0':.1,'1':.1,'2':.8})
        self.assertEqual(truth_after(spec,32,32,'changed'),{'0':.1,'1':.8,'2':.1})
        spec['regime']='shift-return'
        self.assertEqual(truth_after(spec,32,32,'steady'),{'0':.8,'1':.1,'2':.1})
        self.assertEqual(truth_after(spec,32,32,'changed'),{'0':.1,'1':.1,'2':.8})


if __name__ == '__main__':unittest.main()
