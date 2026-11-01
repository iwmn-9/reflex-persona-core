import copy
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools import review_adverse_predictions as module


class PredictionReviewTests(unittest.TestCase):
    def fixture(self):
        spec=dict(id='example',game='opaque',profile=0,turns=2)
        plan=dict(specs=[spec],arms=['old','new'],reference_arm='old',candidate_arm='new')
        trace=[dict(tick=i,truth={'a':.8,'b':.2},response_prediction={'a':.2,'b':.8},
            response='a',action='x',before={'tick':i},before_owner='same') for i in range(2)]
        rows=[dict(spec=spec,arm=arm,result=dict(complete=True,
            configuration=dict(spec=spec,arm=arm),trace=copy.deepcopy(trace))) for arm in plan['arms']]
        rows[1]['result']['trace'][0]['action']='y'
        return plan,rows

    def test_known_prediction_error_is_not_labeled_sampling_error(self):
        plan,rows=self.fixture();r=module.review(plan,rows);p=r['pairs'][0]
        self.assertAlmostEqual(p['means']['old']['expected_brier_excess'],.72)
        self.assertAlmostEqual(p['means']['old']['expected_log_regret'],.6*__import__('math').log(4))
        self.assertEqual(p['actual_prediction_max_difference'],0)
        self.assertTrue(p['first_action_divergence']['same_owner'])
        self.assertFalse(r['adoption'])
        self.assertIn('Not identifiable',r['sampling_attribution'])
        exact=module.score_prediction({'a':.8,'b':.2},{'a':.8,'b':.2},'b')
        self.assertEqual(exact['expected_brier_excess'],0)
        self.assertEqual(exact['expected_log_regret'],0)

    def test_missing_partial_duplicate_and_unpaired_histories_are_rejected(self):
        for kind in ('missing','partial','duplicate','unpaired'):
            with self.subTest(kind=kind):
                plan,rows=self.fixture()
                if kind=='missing':rows.pop()
                elif kind=='partial':rows[0]['result']['trace'].pop()
                elif kind=='duplicate':rows.append(copy.deepcopy(rows[0]))
                else:rows[1]['result']['trace'][0]['response']='b'
                with self.assertRaises(ValueError):module.review(plan,rows)

    def test_audited_input_tampering_is_rejected(self):
        plan,rows=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            files={'plan.json':json.dumps(plan).encode(),
                'runs.jsonl.gz':gzip.compress(b'\n'.join(json.dumps(r).encode() for r in rows))}
            for name,data in files.items():(root/name).write_bytes(data)
            proof=dict(complete=True,pairs=1,executions=2,actual_steps=4,source_ref='fixture',
                artifacts={name:hashlib.sha256(data).hexdigest() for name,data in files.items()})
            (root/'verification.json').write_text(json.dumps(proof),encoding='utf8')
            self.assertTrue(module.from_archive(root)['complete'])
            (root/'plan.json').write_bytes(files['plan.json']+b' ')
            with self.assertRaisesRegex(ValueError,'checksum'):module.from_archive(root)


if __name__=='__main__':unittest.main()
