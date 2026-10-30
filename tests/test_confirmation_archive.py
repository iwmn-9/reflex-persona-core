"""Confirmation packaging cannot hide one control or mislabel a failed gate."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import archive_adverse_confirmation as target
from tools.owned_checkpoint import encoded, sha


class ConfirmationArchiveTests(unittest.TestCase):
    def test_both_control_losses_survive_packaging_and_quality_failure(self):
        plan = dict(specs=target.reserved_specs(), arms=['original16', 'stable', 'candidate'],
                    candidate_arm='candidate', concordant_adverse=[])
        comparison = dict(checks={'original16_complete': True, 'stable_complete': True},
            numeric_gate_passed=False, adoption=False, remaining_review='all losses/persona',
            comparisons={ref: dict(complete=True, pairs_complete=40, loss=loss,
                development_gate_passed=False, gate_checks={}, independent_confirmation='not started')
                for ref, loss in [('original16', -0.2), ('stable', -0.1)]})

        def audit(root, plan_path, source, output):
            self.assertEqual(json.loads(plan_path.read_bytes()), plan)
            self.assertEqual(source, 'frozen-ref')
            output.mkdir()
            (output / 'source.zip').write_bytes(b'unchanged source archive')
            (output / 'checkpoints.zip').write_bytes(b'unchanged all-arm checkpoint archive')
            receipt = dict(complete=True, pairs=40, executions=120, actual_steps=15360,
                workers=[{'id': 'fixture', 'arm': a, 'hashes': {'result.json': a + '-checksum'}}
                         for a in plan['arms']], development_gate_passed=False,
                independent_confirmation='not started', adoption=False, auditor_sha256='original-auditor',
                limitations=['original exact source/history limitation', 'development only'],
                artifacts={p.name: sha(p.read_bytes()) for p in output.iterdir()})
            (output / 'verification.json').write_bytes(encoded(receipt))
            return receipt

        def analyze(root, selected):
            reference = selected['reference_arm']
            return dict(complete=True, development_gate_passed=False, limits=['development only'],
                        source_results={'adverse-fixture-' + a: a + '-checksum' for a in plan['arms']},
                        rows=[dict(id=reference + '-loss', score=-0.2 if reference == 'original16' else -0.1)])

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); path = folder / 'plan.json'; output = folder / 'archive'
            path.write_bytes(encoded(plan))
            with patch.object(target, 'audit_study', side_effect=audit), \
                    patch.object(target, 'review', return_value=comparison), \
                    patch.object(target, 'analyze', side_effect=analyze):
                proof = target.archive_confirmation(folder, path, 'frozen-ref', output)
            self.assertFalse(proof['numeric_gate_passed']); self.assertFalse(proof['adoption'])
            self.assertEqual(proof['independent_confirmation'], 'complete_audited')
            self.assertEqual(proof['source_history_auditor_sha256'], 'original-auditor')
            self.assertEqual([w['arm'] for w in proof['workers']], plan['arms'])
            self.assertTrue(all(sha((output / name).read_bytes()) == h for name, h in proof['artifacts'].items()))
            self.assertEqual((output / 'checkpoints.zip').read_bytes(), b'unchanged all-arm checkpoint archive')
            summary = json.loads((output / 'summary.json').read_bytes())
            outcomes = json.loads((output / 'outcome_review.json').read_bytes())
            self.assertEqual(set(outcomes['comparisons']), {'original16', 'stable'})
            self.assertEqual(outcomes['comparisons']['original16']['rows'][0]['score'], -0.2)
            self.assertEqual(outcomes['comparisons']['stable']['rows'][0]['score'], -0.1)
            self.assertNotIn('development_gate_passed', proof)
            self.assertTrue(all('independent_confirmation' not in c for c in summary['comparisons'].values()))

    def test_partial_second_control_never_creates_complete_archive(self):
        plan = dict(specs=target.reserved_specs(), arms=['original16', 'stable', 'candidate'],
                    candidate_arm='candidate')
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); path = folder / 'plan.json'; output = folder / 'archive'
            path.write_bytes(encoded(plan))
            with patch.object(target, 'audit_study') as audit, patch.object(target, 'review',
                    return_value=dict(checks={'original16_complete': False, 'stable_complete': True})):
                with self.assertRaises(ValueError):
                    target.archive_confirmation(folder, path, 'frozen-ref', output)
                audit.assert_not_called()
            self.assertFalse(output.exists())

    def test_raw_source_history_audit_is_mandatory_even_after_numeric_pass(self):
        comparison = dict(checks={'original16_complete': True, 'stable_complete': True}, numeric_gate_passed=True)
        for change in ({'complete': False}, {'executions': 119}, {'actual_steps': 15359}):
            proof = dict(complete=True, pairs=40, executions=120, actual_steps=15360)
            proof.update(change)
            with self.assertRaises(ValueError):
                target.confirmation_metadata(proof, comparison)


if __name__ == '__main__':
    unittest.main()
