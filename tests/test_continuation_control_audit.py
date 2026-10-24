import copy
import json
import tempfile
import unittest
from pathlib import Path

from tools.audit_continuation_controls import audit_control
from tools.owned_checkpoint import encoded, sha
from tools.run_continuation_portfolio import execution


class FinishedControlAuditTests(unittest.TestCase):
    def test_factual_control_is_replayed_and_semantic_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = dict(id='control', mechanism='world', game='resources', profile=2,
                turns=2, seed=56101, regime='outside', controller={})
            directory = root / spec['id'] / 'baseline'
            result = execution(directory, spec, False)
            envelope = json.loads((directory / 'checkpoint.json').read_bytes())
            manifest = {'tools/' + k: v for k, v in result['configuration']['sources'].items()}
            proof = audit_control(root, spec, manifest)
            self.assertEqual(proof['actual_decisions'], 2)
            # Keep envelope and result mutually consistent: these are semantic
            # changes, not merely a broken checksum that masks missing replay.
            for mutation in ('owner', 'physics', 'expectation', 'rng', 'incomplete'):
                with self.subTest(mutation=mutation):
                    row, saved = copy.deepcopy(result), copy.deepcopy(envelope['payload'])
                    if mutation == 'owner':
                        row['trace'][-1]['owner'] = 'wrong owner'
                    elif mutation == 'physics':
                        row['trace'][-1]['after']['cash'] += 1
                        row['final']['cash'] += 1
                    elif mutation == 'expectation':
                        row['trace'][-1]['conditional_increment'] += .01
                    elif mutation == 'rng':
                        saved['rng']['state']['state'] += 1
                    else:
                        row['complete'] = False
                    saved['trace'] = copy.deepcopy(row['trace'])
                    saved['world'] = copy.deepcopy(row['final'])
                    changed = dict(envelope, payload=saved, sha256=sha(encoded(saved)))
                    (directory / 'result.json').write_bytes(encoded(row))
                    (directory / 'checkpoint.json').write_bytes(encoded(changed))
                    with self.assertRaises(AssertionError):
                        audit_control(root, spec, manifest)


if __name__ == '__main__':
    unittest.main()
