"""Check the mechanism diagnostic, not a production strength claim."""
import unittest
from tools.probe_observation_plans import (Model,State,observation_solution,
    script_solution,oracle_value,independent_check,run_case,profiles)


class ObservationCapacityProbeTests(unittest.TestCase):
    def support(self,m):
        return [(.5,m.step(State(world=w),m.names['a1'])) for w in range(2)]

    def test_visible_information_allows_recourse_without_world_key(self):
        m=Model(5,3,True,.75); support=self.support(m)
        value,policy=observation_solution(m,support)
        self.assertEqual(value,1.)
        self.assertEqual(independent_check(m,support),(1.,9))
        self.assertEqual(script_solution(m,support)[0],.5)
        for node in policy:
            self.assertTrue(all(item[0] in ('observe','act') for item in node))

    def test_hidden_states_must_share_decision_despite_oracle_advantage(self):
        m=Model(5,3,False,.75); support=self.support(m)
        self.assertEqual(observation_solution(m,support)[0],.5)
        self.assertEqual(independent_check(m,support),(.5,3))
        self.assertEqual(sum(p*oracle_value(m,s) for p,s in support),1.)

    def test_information_cannot_change_actions_before_reveal(self):
        m=Model(8,6,True,.75)
        left,right=State(1,m.names['a1'],0),State(1,m.names['a1'],1)
        while left.tick < m.reveal:
            self.assertEqual(m.public(left),m.public(right))
            left,right=m.step(left,m.names['n0']),m.step(right,m.names['n0'])
        self.assertNotEqual(m.public(left),m.public(right))

    def test_existing_flow_and_persona_selector_choose_conditional_investment(self):
        row=run_case(Model(3,1,True,.75),profiles()[0])
        self.assertEqual((row['fixed_root'],row['conditional_root']),('a0','a1'))
        self.assertEqual(row['actual_complete_worlds'],[1.,1.])

    def test_existing_selector_does_not_profit_from_hidden_oracle(self):
        row=run_case(Model(5,3,False,.75),profiles()[1])
        self.assertEqual((row['fixed_goal'],row['conditional_goal']),(.75,.75))
        self.assertEqual((row['fixed_root'],row['conditional_root']),('a0','a0'))

    def test_opaque_names_do_not_supply_action_heuristic(self):
        a=run_case(Model(5,3,True,.75),profiles()[2])
        b=run_case(Model(5,3,True,.75,True),profiles()[2])
        self.assertEqual((a['fixed_goal'],a['conditional_goal']),
                         (b['fixed_goal'],b['conditional_goal']))
        self.assertNotEqual(a['conditional_root'],b['conditional_root'])


if __name__=='__main__': unittest.main()
