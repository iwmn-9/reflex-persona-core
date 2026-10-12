import unittest
from tools.run_foundation_audit import graph_values, play, prepare
from tools.run_procedural_stability import Trial
from reflex.laboratory import profiles


class FoundationAuditTests(unittest.TestCase):
    def test_exact_oracle_matches_exhaustive_feedback_policies(self):
        t=Trial(23100,profiles()[0],3);v,q=graph_values(t)
        def exhaustive(state):
            if t.terminal(state):return float(state[1] in t.winning)
            return max(sum(exhaustive((state[0]+1,n)) for n in t.edges[*state,a])/2
                       for a in ('option-u','option-v'))
        for state,value in v.items():self.assertEqual(value,exhaustive(state))
        self.assertTrue(all(0<=x<=1 for x in q.values()))

    def test_oracle_uses_rules_not_actual_chance_and_replay_is_complete(self):
        case=dict(game='public_graph',graph=23101,length=3)
        for seed in (33100,33101):
            r=play((case,profiles()[0],seed,'oracle'))
            self.assertTrue(r['replay']);self.assertEqual(sum(t['oracle_regret'] for t in r['trace']),0)
            self.assertEqual(len(r['trace']),3)

    def test_same_opening_for_personalities_and_arms(self):
        case=dict(game='auction_contest',opening_steps=2,opening_seed=123)
        positions=[prepare(case,p,33100)[0:3] for p in profiles()]
        self.assertTrue(all(b.record(w)==positions[0][0].record(positions[0][1]) for b,w,_ in positions))
        self.assertTrue(all(prefix==positions[0][2] for _,_,prefix in positions))

    def test_small_full_episodes_all_comparison_arms(self):
        case=dict(game='public_graph',graph=23102,length=3)
        for arm in ('reflex','persona4','purpose4','persona8'):
            r=play((case,profiles()[2],33100,arm));self.assertTrue(r['replay'])
            for t in r['trace']:
                f=t['forecast']
                if arm=='purpose4':self.assertAlmostEqual(f['planned_purpose_cost'],0)
                if f:self.assertEqual(set(f['predicted']),{'option-u','option-v'})

    def test_persona_arm_matches_existing_forecast_controller(self):
        from reflex.reliability_trials import play as existing
        from reflex.forecast_control import ForecastBudget
        case=dict(game='public_graph',graph=23102,length=3);p=profiles()[1]
        a=play((case,p,33100,'persona4'))
        b=existing(case,p,33100,'same',budgets={'same':ForecastBudget(horizon=4,direct_continuation=True)})
        self.assertEqual(a['actions'],b['actions']);self.assertEqual(a['goal'],b['goal'])
        self.assertEqual([t['decision'] for t in a['trace']],[t['decision'] for t in b['trace']])

    def test_exact_policy_value_matches_every_actual_chance_sequence(self):
        from unittest.mock import patch
        from tools.measure_foundation_policy import measure
        from reflex.contingent_games import make_bridge
        case=dict(game='public_graph',graph=23108,length=3);profile=profiles()[2]
        for arm in ('reflex','persona4','purpose4'):
            wins=0
            for bits in range(8):
                def factory(p,c):
                    b=make_bridge(p,c)
                    def actual(w,key,seed):
                        wanted=(bits>>w.turns)&1
                        synthetic=next(s for s in range(100) if b.rules.outcome(w.position,key,s)==wanted)
                        return b.advance(w,key,synthetic)
                    b.actual=actual;return b
                with patch('tools.run_foundation_audit.make_bridge',factory):
                    r=play((case,profile,33100,arm));wins+=r['goal']['status']=='success'
            exact=measure((23108,3,profile,arm))
            self.assertEqual(exact['success_probability'],wins/8)
            self.assertAlmostEqual(exact['optimal_probability']-wins/8,exact['expected_opportunity_loss'])


if __name__=='__main__':unittest.main()
