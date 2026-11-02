import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import run_bounded_campaign as entry


class BoundedCampaignSchedulingTests(unittest.TestCase):
    def test_interrupted_slice_still_stops_at_actual_intervention_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = root / 'plan.json'
            plan.write_text(json.dumps(dict(boundary=32, slice_decisions=4)))
            calls = []

            def run(folder, spec, arm, **kwargs):
                calls.append((Path(folder).name, spec, arm, kwargs))
                return 'retained-result'

            def campaign(*args):
                for tick in (None, 29, 31, 32, 33):
                    folder = root / str(tick)
                    folder.mkdir()
                    if tick is not None:
                        (folder / 'checkpoint.json').write_text(json.dumps(
                            dict(payload=dict(world=dict(tick=tick)))))
                    result = entry.run_recovery_campaign.run(folder, {'id': 'case'},
                        'expected', max_decisions=4, registration={'id': 'fixed'})
                    self.assertEqual(result, 'retained-result')

            with patch.object(entry.multiprocessing, 'set_start_method'), \
                 patch.object(entry, 'activate', return_value={}), \
                 patch.object(entry.run_recovery_campaign, 'run', run), \
                 patch.object(entry.run_recovery_campaign, 'main', campaign), \
                 patch('sys.argv', ['entry', '--plan', str(plan), '--output', str(root / 'out')]):
                entry.main()

            self.assertEqual([c[3]['max_decisions'] for c in calls], [4, 3, 1, 4, 4])
            for _, spec, arm, kwargs in calls:
                self.assertEqual((spec, arm, kwargs['registration']),
                    ({'id': 'case'}, 'expected', {'id': 'fixed'}))


if __name__ == '__main__':
    unittest.main()
