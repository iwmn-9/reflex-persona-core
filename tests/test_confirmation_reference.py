"""The reserved three-way comparison must retain the actual adopted H16."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.adverse_segment import audit
from tools.run_adverse_recovery import run
from tools.run_continuation_portfolio import execution


class ConfirmationReferenceTests(unittest.TestCase):
    def test_reference_matches_old_execution_in_each_rule_family(self):
        for game in ('auction', 'combat', 'gambling', 'resources'):
            with self.subTest(game=game), tempfile.TemporaryDirectory() as folder:
                spec = dict(id='reference-'+game, mechanism='world', game=game,
                            profile=1, seed=601, regime='outside', turns=3)
                root = Path(folder)
                expected = execution(root/'old', spec, False)
                actual = run(root/'new', spec, 'original16')
                self.assertEqual(expected['score'], actual['score'])
                for a,b in zip(expected['trace'], actual['trace']):
                    for key in ('tick','before','after','action','owner','before_owner','truth','response','conditional_increment','selected_purpose','predicted_actions'):
                        self.assertEqual(a[key], b[key], key)
                old = json.loads((root/'old/checkpoint.json').read_bytes())['payload']
                new = json.loads((root/'new/checkpoint.json').read_bytes())['payload']
                for key in ('world','owner','rng'):
                    self.assertEqual(old[key], new[key], key)

    def test_original16_resume_and_exact_owner_audit(self):
        spec = dict(id='resume-original16', mechanism='world', game='combat',
                    profile=2, seed=601, regime='outside', turns=3)
        registration = dict(plan_sha256='test', source_sha256='test')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            full = run(root/'full', spec, 'original16', registration=registration)
            self.assertIsNone(run(root/'split', spec, 'original16', max_decisions=1, registration=registration))
            resumed = run(root/'split', spec, 'original16', registration=registration)
            for key in ('trace','final','final_owner','final_rng','checks','configuration'):
                self.assertEqual(full[key], resumed[key], key)
            self.assertTrue(audit(root/'split', spec, 'original16', registration)['complete'])


if __name__ == '__main__':
    unittest.main()
