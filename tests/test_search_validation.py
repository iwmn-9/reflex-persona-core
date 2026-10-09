import copy
import unittest
from reflex.core import Policy, digest
from reflex.examples import context, action, effect
from reflex.continuation_search import search_forecast
from reflex.deliberation import select
from reflex.purpose_plan import Goal


class SearchValidationTests(unittest.TestCase):
    def fixture(self):
        c = context('bank-check', [action(k, effect()) for k in ('a', 'b')],
                    needs={'physiology': 0., 'safety': 0.})
        def advance(s, k, seed):
            return (1, k, seed), effect()
        def assess(s):
            win = (s[1] == 'a') == (s[2] == 0)
            return Goal('finish', 'success' if win else 'failure', 1. if win else -1.).record()
        return c, dict(observe=lambda s, m: copy.deepcopy(c), advance=advance,
                       terminal=lambda s: s[0] == 1, assess=assess,
                       horizon=1, depth=0, seeds=(0,), target='finish',
                       policy=Policy(principle_priority='finite'))

    def test_frozen_candidates_are_rechecked_before_selection(self):
        c, kw = self.fixture(); saved = digest(c)
        old = search_forecast(c, (0, None, None), **kw)
        new = search_forecast(c, (0, None, None), validation_seeds=(1,), **kw)
        self.assertEqual(select([c], old, policy=kw['policy'])[0][0]['action_id'], 'a')
        self.assertEqual(select([c], new, policy=kw['policy'])[0][0]['action_id'], 'b')
        a = old.audit['continuation_search']; b = new.audit['continuation_search']
        self.assertEqual(a['evaluated'], b['evaluated'])
        self.assertEqual(b['validation']['evaluated'], len(a['proposals']))
        self.assertEqual(b['validation']['discovery'], {k: n['branches'] for k, n in a['proposals'].items()})
        self.assertEqual({n['root'] for n in b['proposals'].values()}, {'a', 'b'})
        self.assertTrue(all(row['seed'] == 1 for n in b['proposals'].values() for row in n['branches']))
        self.assertEqual(digest(c), saved)

    def test_disjoint_bank_preserves_schedule_generation(self):
        from test_schedule_sampling import ScheduleSamplingTests
        c, kw = ScheduleSamplingTests().fixture()
        old = search_forecast(c, (0, 0), samples=16, **kw)
        new = search_forecast(c, (0, 0), samples=16, validation_seeds=(1, 2), **kw)
        a = old.audit['continuation_search']; b = new.audit['continuation_search']
        self.assertEqual({k: (n['root'], n['schedule']) for k, n in a['proposals'].items()},
                         {k: (n['root'], n['schedule']) for k, n in b['proposals'].items()})
        self.assertEqual(a['layers'], b['layers'])
        self.assertEqual(a['sampling'], b['sampling'])
        self.assertEqual(b['validation']['discovery'], {k: n['branches'] for k, n in a['proposals'].items()})
        self.assertEqual(digest(new.audit), digest(search_forecast(c, (0, 0), samples=16, validation_seeds=(1, 2), **kw).audit))

    def test_disabled_path_is_identical(self):
        c, kw = self.fixture()
        a = search_forecast(c, (0, None, None), **kw)
        b = search_forecast(c, (0, None, None), validation_seeds=None, **kw)
        self.assertEqual(digest(a.audit), digest(b.audit))
        self.assertNotIn('validation', a.audit['continuation_search'])

    def test_invalid_or_overlapping_banks_are_rejected(self):
        c, kw = self.fixture()
        for seeds in ((), (0,), (1, 1), (True,), (1.5,), '1', tuple(range(1, 10))):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                search_forecast(c, (0, None, None), validation_seeds=seeds, **kw)


if __name__ == '__main__':
    unittest.main()
