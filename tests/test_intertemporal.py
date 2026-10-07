import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.examples import context,action,effect
from reflex.core import Policy
from reflex.intertemporal import Branch,forecast,patience
from reflex.deliberation import select
from reflex.intertemporal_experiment import auction_paths,resource_paths,combat_paths,make_context,replay,planning_paths
from reflex.decision_loop import DecisionLoop,Request
from reflex.judgment import Binding
from reflex.planning import vector


def fixture():
    paths={'take':(Branch(1.,(effect(.4),effect()),(1.,1.)),),
           'invest':(Branch(1.,(effect(-.1,cost=.03),effect(.8)),(1.,1.)),)}
    c=context('tradeoff',[action(k,b[0].effects[0]) for k,b in paths.items()],needs={k:0 for k in ('physiology','safety','belonging','esteem','growth')})
    return c,paths


def chosen(c,paths,discount=None):
    f=forecast(c,paths,horizon=len(next(iter(paths.values()))[0].effects),unit='turns',target='incremental-goal',discount=discount,max_regret=2.)
    decisions,audit=select([c],f)
    return decisions[0]['action_id'],f,audit


class IntertemporalTests(unittest.TestCase):
    def test_persona_time_can_accept_present_loss_for_later_gain(self):
        c,paths=fixture();self.assertEqual(Policy().choose(c,False)['action_id'],'take')
        self.assertEqual(chosen(c,paths,.95)[0],'invest')
        self.assertEqual(chosen(c,paths,.4)[0],'take')

    def test_urgency_shortens_time_weight_without_editing_personality(self):
        c,paths=fixture();c['personality'].update(conscientiousness=.95,neuroticism=.1)
        before=copy.deepcopy(c)
        self.assertEqual(chosen(c,paths)[0],'invest')
        c['needs']['physiology']['deficit']=.95
        self.assertLess(patience(c),patience(before))
        self.assertEqual(chosen(c,paths)[0],'take')
        self.assertEqual(c['personality'],before['personality'])

    def test_unknown_credit_cannot_erase_loss_or_root_cost(self):
        c,paths=fixture();weak=planning_paths(paths,0.)
        root,f,_=chosen(c,weak,.95);self.assertEqual(root,'take')
        plan=next(a for a in f.contexts[0]['actions'] if f.roots[a['id']]==('invest',))
        r=plan['outcomes'][0]
        self.assertAlmostEqual(r['objective'],-.1/1.95)
        self.assertAlmostEqual(r['cost'],.03/1.95)

    def test_future_losses_survive_zero_confidence(self):
        c,paths=fixture();paths['take']=(Branch(1.,(effect(.4),effect(-.9)),(1.,0.)),)
        paths['invest']=(Branch(1.,(effect(),effect()),(1.,0.)),)
        self.assertEqual(chosen(c,paths,.95)[0],'invest')

    def test_complete_equal_horizons_and_root_coverage_required(self):
        c,paths=fixture()
        for bad in ({'take':paths['take']},{**paths,'phantom':paths['take']},
                    {**paths,'invest':(Branch(1.,(effect(),),(1.,)),)},
                    {**paths,'invest':(Branch(.5,(effect(),effect()),(1.,1.)),)},
                    {**paths,'invest':(Branch(1.,(effect(p=.5),effect()),(1.,1.)),)}):
            with self.assertRaises(ValueError):forecast(c,bad,horizon=2,unit='turns',target='goal')

    def test_illegal_or_known_failed_root_cannot_enter(self):
        c,paths=fixture();c['actions'][0]['legal']=False
        with self.assertRaises(ValueError):chosen(c,paths)

    def test_branch_distribution_retains_downside_instead_of_averaging_it_away(self):
        c,paths=fixture();paths['invest']=(Branch(.5,(effect(-.1),effect(.8)),(1.,1.)),Branch(.5,(effect(-.1),effect(-.8)),(1.,1.)))
        root,f,_=chosen(c,paths,.95);self.assertEqual(root,'take')
        a=next(a for a in f.contexts[0]['actions'] if f.roots[a['id']]==('invest',))
        self.assertEqual(len(a['outcomes']),2);self.assertTrue(any(r['objective']<0 for r in a['outcomes']))

    def test_original_actor_contract_and_inputs_are_immutable(self):
        c,paths=fixture();before=copy.deepcopy((c,paths));_,f,_=chosen(c,paths,.95)
        self.assertEqual((c,paths),before)
        for k in c:
            if k not in ('facts','actions'):self.assertEqual(c[k],f.contexts[0][k])

    def test_public_auction_reservation_changes_with_patience(self):
        paths,traces=auction_paths(2);p=dict(traits=(.5,.5,.5,.5,.5),values={})
        c=make_context('auction-probe',paths,p,0.)
        self.assertEqual(chosen(c,paths,.95)[0],'bid:0')
        self.assertEqual(chosen(c,paths,.5)[0],'bid:3')
        self.assertGreater(replay('auction',traces),0)

    def test_deadline_and_weak_future_can_reverse_reservation(self):
        paths,_=auction_paths(2);p=dict(traits=(.5,.5,.5,.5,.5),values={});c=make_context('auction-probe',paths,p,0.)
        self.assertEqual(chosen(c,planning_paths(paths,.25),.95)[0],'bid:3')
        short,_=auction_paths(1);sc=make_context('auction-short',short,p,0.)
        self.assertEqual(chosen(sc,short,.95)[0],'bid:3')
        # The larger window cannot repeatedly collect the final prize.
        long,_=auction_paths(8)
        for root in paths:
            self.assertAlmostEqual(sum(r['objective'] for r in paths[root][0].effects),sum(r['objective'] for r in long[root][0].effects))

    def test_actual_infrastructure_cost_is_repaid_only_by_rule_production(self):
        paths,traces=resource_paths(8)
        self.assertLess(paths['build:market'][0].effects[0]['objective'],0)
        self.assertGreater(sum(r['objective'] for r in paths['build:market'][0].effects),sum(r['objective'] for r in paths['gather:gold'][0].effects))
        self.assertGreater(replay('resources',traces),0)

    def test_storage_saturation_can_cancel_investment_payback(self):
        paths,_=resource_paths(8,gold=4)
        self.assertAlmostEqual(sum(r['objective'] for r in paths['build:market'][0].effects),sum(r['objective'] for r in paths['gather:gold'][0].effects))

    def test_expired_combat_has_zero_tail_not_a_repeated_terminal_prize(self):
        paths,traces=combat_paths(8,hp=3,limit=1,seeds=(800,))
        for bs in paths.values():
            for b in bs:self.assertTrue(all(r==effect() for r in b.effects[1:]))
        self.assertGreater(replay('combat',traces),0)

    def test_actual_feedback_keeps_the_immediate_target_and_roundtrips(self):
        c,paths=fixture();loop=DecisionLoop(c)
        request=Request(c,{k:Binding(k,'public',('objective',)) for k in paths},{k:() for k in paths})
        def planner(cs,ds):return forecast(cs[0],paths,horizon=2,unit='turns',target='future-goal',discount=.95,max_regret=2.)
        r=DecisionLoop.decide_batch([(loop,request)],False,planner)[0]
        self.assertEqual(r['decision']['action_id'],'invest')
        self.assertEqual(loop.pending['prior'][0]['objective'],-.1)
        loop.observe(r['ticket'],vector(effect(-.12,cost=.03)))
        self.assertEqual(next(iter(loop.memory.entries.values()))['samples'][0][0],-.12)
        self.assertEqual(DecisionLoop.from_record(c,loop.record()).record(),loop.record())

    def test_one_step_preserves_intent_switch_cost_and_root_certainty(self):
        c=context('intent',[action('keep',effect(.4),switch_cost=.1,confidence=.8,familiarity=.9),
                            action('switch',effect(.5),switch_cost=.1,confidence=.8,familiarity=.9)])
        c['state']['intent_action']='keep'
        paths={a['id']:(Branch(1.,(a['outcomes'][0],),(1.,)),) for a in c['actions']}
        root,f,_=chosen(c,paths)
        self.assertEqual(root,Policy().choose(c,False)['action_id'])
        self.assertEqual(root,'keep')

    def test_zero_tail_cannot_invent_extra_switch_inertia(self):
        c=context('switch-fee',[action('keep',effect(.1)),action('switch',effect(.2),switch_cost=.03)])
        c['state']['intent_action']='keep'
        self.assertEqual(Policy().choose(c,False)['action_id'],'switch')
        for h in (1,8):
            paths={a['id']:(Branch(1.,(a['outcomes'][0],)+(effect(),)*(h-1),(1.,)*h),) for a in c['actions']}
            self.assertEqual(chosen(c,paths,1.)[0],'switch')


if __name__=='__main__':unittest.main()
