import copy
from dataclasses import replace
import unittest
from reflex.core import NEEDS
from reflex.examples import context,action,effect
from reflex.deliberation import JointForecast,select
from reflex.goal_ties import GoalTieRefinement,dominates,reflex_goal_planner


def fixture():
    c=context('alias',[action(k,effect(1.,values={'achievement':1.})) for k in ('a','b','c')],
        needs={k:0. for k in NEEDS},values={'achievement':1.},mode='principle')
    lotteries={'a':((.9,.2),(.1,.5)), 'b':((.1,.2),(.9,.5)), 'c':((.5,.2),(.5,.5))}
    f=JointForecast((copy.deepcopy(c),),{k:(k,) for k in lotteries},{k:.3 for k in lotteries},1,
        max_regret=2.,target='score',goal_ties=GoalTieRefinement('score',lotteries))
    return c,f


class GoalTieTests(unittest.TestCase):
    def test_dominance_refines_alias_without_mutation_or_weight_changes(self):
        c,f=fixture();before=copy.deepcopy((c,f))
        self.assertEqual(select([c],replace(f,goal_ties=None))[0][0]['action_id'],'a')
        result,audit=select([c],f)
        self.assertEqual(result[0]['action_id'],'b')
        self.assertEqual(audit['goal_ties']['original'],'a')
        self.assertEqual((c,f),before)

    def test_crossing_risk_is_not_resolved_by_mean(self):
        c,f=fixture();lotteries={'a':((1.,.2),),'b':((.5,-.2),(.5,.9)),'c':((1.,.1),)}
        f=replace(f,goal_ties=GoalTieRefinement('score',lotteries))
        self.assertEqual(select([c],f)[0][0]['action_id'],'a')
        self.assertFalse(dominates(lotteries['b'],lotteries['a']))

    def test_progress_mask_remains_authoritative(self):
        c,f=fixture()
        self.assertEqual(select([c],f,root_allowed=[{'a'}])[0][0]['action_id'],'a')
        self.assertEqual(select([c],f,root_allowed=[{'a','c'}])[0][0]['action_id'],'c')

    def test_different_intrinsic_or_action_contract_is_not_aliased(self):
        for field in ('values','confidence','familiarity','target','switch_cost'):
            c,f=fixture()
            for a in c['actions'][1:]:
                if field=='values':a['outcomes'][0]['values']['benevolence']=-.2
                elif field=='target':a[field]='someone'
                else:a[field]=.4
            f=replace(f,contexts=(copy.deepcopy(c),))
            ordinary=select([c],replace(f,goal_ties=None))[0][0]['action_id']
            self.assertEqual(select([c],f)[0][0]['action_id'],ordinary)

    def test_different_conditional_effects_remain_unmodified(self):
        c,f=fixture()
        for a in c['actions'][1:]:
            a['outcomes']=[dict(effect(.9,values={'achievement':1.}),p=.5),dict(effect(1.,values={'achievement':1.}),p=.5)]
        f=replace(f,contexts=(copy.deepcopy(c),))
        self.assertEqual(select([c],f)[0][0]['action_id'],select([c],replace(f,goal_ties=None))[0][0]['action_id'])

    def test_invalid_or_social_contract_declines(self):
        c,f=fixture()
        with self.assertRaises(ValueError):select([c],f,appraisal=[{}])
        with self.assertRaises(ValueError):select([c],replace(f,horizon=2))
        with self.assertRaises(ValueError):select([c],replace(f,target='different'))
        with self.assertRaises(ValueError):select([c],replace(f,goal_ties=GoalTieRefinement('score',{'a':((1.,float('nan')),)})))

    def test_real_owner_refinement_is_detached_and_separate_from_recovery_budget(self):
        from reflex.laboratory import PROFILES
        from reflex.closed_loop import owned_rollout
        from tools.continuing_games import ContinuingGame
        g=ContinuingGame('gambling',PROFILES[0],16);w=g.start();w.update(tick=5,cash=20,investment=1)
        c=g.context(w);owner=g.actor(c,learning='segmented');before=copy.deepcopy(owner.record())
        args=dict(observe=g.context,events=g.events,terminal=g.terminal,assess=lambda w:g.goal(w).record(),
            record=lambda w:w,request_for=g.request,feedback=g.feedback,model_id=g.model_id,target=g.target,
            tail_steps=2,seeds=(701,),tail_recovery=2,tail_recovery_limit=1)
        old=owned_rollout(owner,**args);new=owned_rollout(owner,**args,tail_goal_refinement=True)
        self.assertNotEqual(old.frontier.identity,new.frontier.identity)
        new.frontier.distribution(w,owner)
        self.assertEqual(new.frontier.future_calls,0)
        self.assertEqual(new.frontier.refinement_calls,1)
        self.assertEqual(owner.record(),before)

    def test_refinement_uses_ordinary_feedback_in_all_rule_families(self):
        from tools.run_owned_value import run
        for game in ('auction','resources','combat','gambling'):
            job=[game,0,55990,'outside',4,1,2,0]
            row=run(job,'receding_recovery',cumulative_achievement=True,tail_goal_refinement=True)
            self.assertEqual(row['checks']['owner_parity'],4)
            self.assertEqual(row['checks']['fixed_axes'],4)
            self.assertTrue(row['tail_goal_refinement'])

    def test_registered_mechanisms_cannot_silently_change_or_disappear(self):
        from tools.value_contract_config import configurations,arguments
        self.assertEqual(configurations({}),{'baseline':{},'candidate':{'cumulative_achievement':True}})
        plan={'configurations':{'baseline':{'cumulative_achievement':True},'candidate':{'cumulative_achievement':True,'tail_goal_refinement':True}}}
        self.assertEqual(arguments(configurations(plan)['candidate']),['--cumulative-achievement','--tail-goal-refinement'])
        for bad in ({'baseline':{},'candidate':{}},{'baseline':{},'candidate':{'hidden_cheat':True}},
                    {'baseline':{},'candidate':{'tail_goal_refinement':False}}):
            with self.assertRaises(ValueError):configurations({'configurations':bad})

    def test_unrelated_aliases_do_not_override_the_selected_persona_action(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        c,f=fixture();c['actions'][0]['outcomes'][0]['values']['benevolence']=.6
        request=SimpleNamespace(context=c,reader=None,value_proposal=None)
        person=SimpleNamespace(social=None,predictor=Mock());events=Mock()
        planner=reflex_goal_planner(person,request,{},events=events,assess=Mock(),target='score')
        self.assertIsNotNone(planner)
        self.assertIsNone(planner([c],[{'action_id':'a'}]))
        events.assert_not_called();person.predictor.predict.assert_not_called()


if __name__=='__main__':unittest.main()
