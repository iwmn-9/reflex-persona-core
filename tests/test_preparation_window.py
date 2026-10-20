"""The retained deadline belongs to actual feedback, including after resume."""
import json
import tempfile
import unittest
from pathlib import Path

from tools.run_preparation_window import execution


class PreparationWindowTests(unittest.TestCase):
    def test_resumed_clock_and_future_choices_match_the_uninterrupted_owner(self):
        with tempfile.TemporaryDirectory() as folder:
            whole = Path(folder) / 'whole'
            paused = Path(folder) / 'paused'
            options = dict(delay=1, shock=False, mode='fixed-window', turns=5, horizon=3)
            expected = execution(whole, **options)
            self.assertIsNone(execution(paused, **options, max_decisions=2))
            prefix = json.loads((paused/'checkpoint.json').read_bytes())['payload']['trace']
            result = execution(paused, **options)
            self.assertEqual(result['trace'], expected['trace'])
            self.assertEqual(prefix, result['trace'][:2])
            self.assertEqual([t['horizon'] for t in result['trace']], [3, 2, 1, 3, 2])
            self.assertEqual(result['checks'], dict(fixed_axes=5, owner_parity=5, clock=5))
            for i, t in enumerate(result['trace']):
                actual = [s['action'] for s in result['trace'][i:i+t['horizon']]]
                self.assertEqual(t['modeled_actions'], actual)

    def test_future_public_change_does_not_leak_and_is_observed_afterwards(self):
        with tempfile.TemporaryDirectory() as folder:
            runs = [execution(Path(folder)/str(shock), delay=1, shock=shock,
                mode='fixed-window', turns=4, horizon=3) for shock in (False, True)]
            for x, y in zip(runs[0]['trace'][:2], runs[1]['trace'][:2]):
                for key in ('before', 'action', 'before_owner', 'horizon', 'clock'):
                    self.assertEqual(x[key], y[key])
            self.assertEqual(runs[1]['trace'][1]['after']['payout'], 2.)
            self.assertEqual(runs[1]['trace'][2]['before']['payout'], 2.)
            with self.assertRaisesRegex(ValueError, 'contract changed'):
                execution(Path(folder)/'True', delay=1, shock=True,
                    mode='rolling', turns=4, horizon=3)


if __name__ == '__main__':
    unittest.main()
