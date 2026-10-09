import copy
import unittest
from reflex.core import Policy, digest
from reflex.examples import context, action, effect
from reflex.intertemporal import Branch
from reflex.purpose_plan import Goal, goal_forecast
from reflex.continuation_search import search_forecast
from reflex.deliberation import select


class SettlementWeightTests(unittest.TestCase):
    def fixture(self, status='success', value=1.):
        c = context('terminal-weight', [action('a', effect())],
                    needs={'physiology': 0., 'safety': 0.})
        flows = tuple(effect(needs={'physiology': .2}, values={'power': .3}, cost=.1) for _ in range(3))
        paths = {'a': (Branch(1., flows, (1.,) * 3),)}
        audit = {'a': [dict(terminal=status != 'running', actions=['a'] * 3,
                            assessment=Goal('finish', status, value).record())]}
        kw = dict(horizon=3, unit='turns', target='finish', max_regret=2.,
                  policy=Policy(principle_priority='finite'))
        return c, paths, audit, kw

    def test_terminal_goal_once_while_preference_flows_keep_time_weight(self):
        for status, value in (('success', 1.), ('failure', -1.), ('draw', 0.), ('scored', .4)):
            with self.subTest(status=status):
                c, paths, audit, kw = self.fixture(status, value); saved = digest(c)
                old = goal_forecast(c, paths, audit, **kw)
                new = goal_forecast(c, paths, audit, settlement_weight='absolute', **kw)
                a = old.contexts[0]['actions'][0]['outcomes'][0]
                b = new.contexts[0]['actions'][0]['outcomes'][0]
                self.assertAlmostEqual(a['objective'], value * old.audit['discount'] ** 2)
                self.assertEqual(b['objective'], value)
                self.assertEqual({k: v for k, v in a.items() if k != 'objective'},
                                 {k: v for k, v in b.items() if k != 'objective'})
                self.assertEqual(old.purpose, new.purpose)
                self.assertEqual(digest(c), saved)
        c, paths, audit, kw = self.fixture('running', .6)
        old = goal_forecast(c, paths, audit, **kw)
        new = goal_forecast(c, paths, audit, settlement_weight='absolute', **kw)
        self.assertEqual(old.contexts, new.contexts)

    def test_confidence_and_large_principle_sacrifice_are_preserved(self):
        c, paths, audit, kw = self.fixture()
        paths['a'] = (Branch(1., paths['a'][0].effects, (.5,) * 3),)
        f = goal_forecast(c, paths, audit, settlement_weight='absolute', **kw)
        self.assertEqual(f.contexts[0]['actions'][0]['outcomes'][0]['objective'], .5)
        # Three separate acts of aid, not one reward paid three times. A strong
        # principle can still beat a certain personal victory when no purpose
        # corridor forbids that sacrifice.
        c = context('principle-sacrifice', [action(k, effect()) for k in ('aid', 'take')],
                    needs={'physiology': 0., 'safety': 0.}, values={'benevolence': 1.}, mode='principle')
        saved = digest(c)
        paths = {k: (Branch(1., tuple(effect(values={'benevolence': sign}) for _ in range(3)), (1.,) * 3),)
                 for k, sign in (('aid', 1.), ('take', -1.))}
        audit = {k: [dict(terminal=True, actions=[k] * 3,
                          assessment=Goal('finish', 'failure' if k == 'aid' else 'success',
                                          -1. if k == 'aid' else 1.).record())] for k in paths}
        f = goal_forecast(c, paths, audit, settlement_weight='absolute', **kw)
        decisions, info = select([c], f, policy=kw['policy'])
        self.assertEqual(decisions[0]['action_id'], 'aid')
        self.assertEqual(info['selected_purpose'], -1.)
        self.assertEqual(info['best_purpose'], 1.)
        self.assertEqual(decisions[0]['next_state']['intent_action'], 'aid')
        self.assertEqual(digest(c), saved)

    def test_search_ranking_uses_the_same_settlement_weight(self):
        c = context('short-patience', [action('start', effect())],
                    needs={'physiology': 1., 'safety': 1.})
        c['personality'].update(conscientiousness=0., neuroticism=1.)
        def observe(s, m):
            cc = copy.deepcopy(c); cc['tick'] = s[0]; cc['state'] = m
            cc['actions'] = [action(k, effect()) for k in ('cash', 'prepare')]
            return cc
        def advance(s, k, seed):
            return (s[0] + 1, s[1] + int(k == 'prepare'), s[2] or k == 'cash'), effect()
        def assess(s):
            status, value = ('scored', .3) if s[2] else ('success', 1.) if s[1] == 2 else ('draw', 0.)
            return Goal('finish', status, value).record()
        kw = dict(observe=observe, advance=advance, terminal=lambda s: s[0] == 3 or s[2],
                  assess=assess, horizon=3, target='finish', samples=32, depth=0, width=1,
                  max_regret=2., policy=Policy(principle_priority='finite'))
        old = search_forecast(c, (0, 0, False), **kw)
        new = search_forecast(c, (0, 0, False), settlement_weight='absolute', **kw)
        self.assertEqual(max(old.purpose.values()), .3)
        self.assertEqual(max(new.purpose.values()), 1.)
        self.assertIn('subjective_settlement_semantics', new.audit)
        _, info = select([c], new, policy=kw['policy'])
        proposal = new.audit['plans'][info['selected_plan']]['proposal']
        self.assertEqual(new.audit['continuation_search']['proposals'][proposal]['schedule'], ['prepare', 'prepare'])

    def test_default_compatibility_and_invalid_modes(self):
        c, paths, audit, kw = self.fixture()
        a = goal_forecast(c, paths, audit, **kw)
        b = goal_forecast(c, paths, audit, settlement_weight='discounted', **kw)
        self.assertEqual(digest(a.audit), digest(b.audit))
        self.assertEqual(a.contexts, b.contexts)
        for mode in (None, True, '', 'unknown'):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                goal_forecast(c, paths, audit, settlement_weight=mode, **kw)


if __name__ == '__main__':
    unittest.main()
