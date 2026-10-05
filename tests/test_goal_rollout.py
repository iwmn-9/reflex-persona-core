"""Default-off sampled opponent models: bounded, public, root-independent."""
import copy
from dataclasses import replace, FrozenInstanceError
from itertools import product
import unittest
from unittest.mock import patch
from reflex.combat import Battle, alive, legal, make_context, resolve
from reflex.combat_planning import TacticalControl, forecast, opponent_intents
from reflex.congestion_experiment import authored_world, enemy_hypotheses
from reflex.core import Policy
from reflex.goal_rollout import GoalRollout, _sample
from reflex.laboratory import profiles


class GoalRolloutTests(unittest.TestCase):
    def test_exact_legacy_helper_across_public_conditions(self):
        for scene, goal, team in product(('friendly_convergence','enemy_convergence','mixed_junction'), ('secure','eliminate','both','either'), (0,1)):
            w=authored_world(scene,goal,team)
            self.assertEqual([opponent_intents(w,team,a) for a in (False,True)], enemy_hypotheses(w,team))

    def test_default_and_invalid_configuration(self):
        self.assertEqual(TacticalControl().opponent_model,'fixed')
        for value in ('adaptive','oracle',True,1,None):
            with self.assertRaises(ValueError): TacticalControl(opponent_model=value)
        for value in ('uniform-coverage','goal-uniform'):
            c=TacticalControl(opponent_model=value)
            self.assertEqual((c.horizon,c.samples,c.max_plans,c.recovery_plans),(6,4,24,48))
            self.assertFalse(c.root_completion)

    def test_each_actor_has_both_goals_in_four_bounded_samples(self):
        for scene,team in product(('friendly_convergence','enemy_convergence','mixed_junction'),(0,1)):
            w=authored_world(scene,'both',team)
            samples=[GoalRollout(w,team,s,'goal-uniform') for s in range(4)]
            for a in alive(w,1-team):
                goals=[dict(s.goals)[a] for s in samples]
                self.assertEqual(goals.count('secure'),2)
                self.assertEqual(goals.count('eliminate'),2)
            self.assertIn(sum(s.broad for s in samples),(0,1))

    def test_stratified_one_sample_marginals_match_declared_prior(self):
        w=authored_world('enemy_convergence','both',0)
        for sample in range(4):
            broad=secure=total=0
            for coverage_phase,goal_phase in product(tuple((i+.5)/100 for i in range(100)),(.125,.375,.625,.875)):
                def unit(key):
                    return coverage_phase if key[2]=='coverage' else goal_phase
                with patch('reflex.goal_rollout._unit',side_effect=unit):
                    m=GoalRollout(w,0,sample,'goal-uniform')
                broad+=m.broad;secure+=dict(m.goals)[3]=='secure';total+=1
            self.assertAlmostEqual(broad/total,.2)
            self.assertAlmostEqual(secure/total,.5)

    def test_methods_share_coverage_and_goals_and_broad_actions(self):
        w=authored_world('mixed_junction','both',0)
        for s in range(4):
            a,b=(GoalRollout(w,0,s,m) for m in ('uniform-coverage','goal-uniform'))
            self.assertEqual(a.broad,b.broad);self.assertEqual(a.goals,b.goals)
            if a.broad:self.assertEqual(a.choose(w,0),b.choose(w,0))
            else:self.assertEqual(a.choose(w,0),opponent_intents(w,0,s%2==1))

    def test_repeated_sampling_is_immutable_deterministic_legal(self):
        w=authored_world('enemy_convergence','both',0);before=copy.deepcopy(w)
        for method in ('uniform-coverage','goal-uniform'):
            for sample in range(4):
                m=GoalRollout(w,0,sample,method);saved=copy.deepcopy(m)
                for _ in range(4):
                    choices=m.choose(w,0)
                    self.assertEqual(choices,GoalRollout(w,0,sample,method).choose(w,0))
                    self.assertTrue(all(k in legal(w,a) for a,k in choices.items()))
                self.assertEqual(m,saved)
                with self.assertRaises(FrozenInstanceError):m.broad=False
        self.assertEqual(w,before)

    def test_no_controller_rng_or_online_learning_path(self):
        w=authored_world('mixed_junction','both',0)
        with patch('reflex.combat.opponent',side_effect=AssertionError('controller')), patch('reflex.combat.reference',side_effect=AssertionError('controller')), patch('reflex.combat.resolve',side_effect=AssertionError('actual transitions')), patch('reflex.combat_planning.resolve',side_effect=AssertionError('alias transitions')), patch('reflex.goal_opponent.GoalOpponentMemory',side_effect=AssertionError('online memory')), patch('reflex.goal_beliefs.GoalBeliefs',side_effect=AssertionError('online memory')):
            for method in ('uniform-coverage','goal-uniform'):
                for sample in range(4):GoalRollout(w,0,sample,method).choose(w,0)
        with self.assertRaises(TypeError):GoalRollout(w,0,0,'goal-uniform',seed=123)

    def test_public_known_single_goal_is_only_latent_goal(self):
        for goal in ('secure','eliminate'):
            w=Battle.start('open',goal)
            for s in range(4):self.assertEqual(set(dict(GoalRollout(w,0,s,'goal-uniform').goals).values()),{goal})
        w=replace(Battle.start('open','both'),secured=(False,True))
        self.assertEqual(set(dict(GoalRollout(w,0,0,'goal-uniform').goals).values()),{'eliminate'})

    def test_future_responses_depend_on_modeled_public_state_only(self):
        w=authored_world('enemy_convergence','both',0);m=GoalRollout(w,0,0,'goal-uniform')
        enemy=m.choose(w,0)
        x,_=resolve(w,{0:'guard',1:'guard',2:'guard',**enemy},123)
        self.assertEqual(m.choose(x,1),copy.deepcopy(m).choose(x,1))
        self.assertTrue(all(k in legal(x,a) for a,k in m.choose(x,1).items()))
        self.assertEqual(m.choose(w,0),enemy)

    def test_wrong_depth_root_or_unregistered_opponent_rejected(self):
        w=Battle.start('open','both');m=GoalRollout(w,0,0,'goal-uniform')
        for depth in (-1,1,16,True):
            with self.assertRaises(ValueError):m.choose(w,depth)
        with self.assertRaises(ValueError):m.choose(replace(w,hold=(1,0)),0)
        dead=replace(w,units=w.units[:5]+(replace(w.units[5],hp=0),))
        m=GoalRollout(dead,0,0,'goal-uniform')
        with self.assertRaises(ValueError):m.choose(replace(w,tick=1),1)

    def test_bounded_constructor(self):
        w=Battle.start('open','both')
        for args in ((w,True,0,'goal-uniform'),(w,0,True,'goal-uniform'),(w,0,4,'goal-uniform'),(w,0,0,'unknown'),(w,0,0,'goal-uniform',0),(w,0,0,'goal-uniform',9)):
            with self.assertRaises(ValueError):GoalRollout(*args)

    def test_cdf_sampling_includes_all_positive_actions(self):
        p={'a':.2,'b':.3,'c':.5}
        self.assertEqual([_sample(p,u) for u in (.0,.199999,.2,.499999,.5,.999999)],['a','a','b','b','c','c'])

    def test_saved_root_sample_same_for_every_current_own_plan(self):
        w=authored_world('enemy_convergence','both',0);actors=alive(w,0)
        cs=[make_context(w,i,profiles()[0],44,'secure',survival_security=True) for i in actors]
        for method in ('fixed','uniform-coverage','goal-uniform'):
            control=TacticalControl(horizon=2,samples=4,max_plans=4,trace_execution=True,opponent_model=method)
            f=forecast(w,actors,cs,[Policy().choose(c) for c in cs],control)
            expected=[opponent_intents(w,0,s%2==1) if method=='fixed' else GoalRollout(w,0,s,method).choose(w,0) for s in range(4)]
            for plan in f.audit['plans'].values():
                for s,path in enumerate(plan['execution_samples']):
                    self.assertEqual(path[0]['opponent'],{str(a):k for a,k in expected[s].items()})
            self.assertLessEqual(f.audit['nodes'],4*4*2)
            for root in f.roots.values():self.assertTrue(all(k in legal(w,a) for a,k in zip(actors,root)))
            if method=='fixed':self.assertNotIn('opponent_samples',f.audit)
            else:self.assertEqual(len(f.audit['opponent_samples']),4)


if __name__=='__main__':unittest.main()
