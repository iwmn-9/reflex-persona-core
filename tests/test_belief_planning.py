import copy
import json
import unittest
from dataclasses import replace
from reflex.belief_planning import BeliefKernel, HiddenOutcome, UnsupportedObservation
from reflex.closed_loop import ClosedLoop, TreeBudget
from reflex.core import digest
from reflex.decision_loop import DecisionLoop
from reflex.examples import context, action, effect
from reflex.integrated_trials import request_for
from reflex.model_observer import ModelFeedback


def sensor_model(h,w,key):
    z=int(h)
    return [HiddenOutcome(.8 if signal==z else .2,dict(signal=signal),effect(),h) for signal in (0,1)]


def kernel(model=sensor_model,**kwargs):
    return BeliefKernel(names=('0','1'),model=model,public_record=lambda s:s,model_id='sensor-v1',owner='observer',**kwargs)


class BeliefPlanningTests(unittest.TestCase):
    def test_observable_events_merge_before_deciding_and_condition_only_on_reveal(self):
        k=kernel();s=k.start({},(.25,.75));saved=digest(k.record(s));rows=k(s,'inspect')
        self.assertEqual(len(rows),2)
        a=next(t for t in rows if t.state.public['signal']==0)
        self.assertAlmostEqual(a.probability,.35)
        self.assertAlmostEqual(a.state.weights[0],4/7)
        self.assertEqual(k.reveal(s,'inspect',{'signal':0},effect()),a.state)
        self.assertEqual(digest(k.record(s)),saved)
        with self.assertRaises(UnsupportedObservation):k.reveal(s,'inspect',{'signal':7},effect())
        self.assertEqual(digest(k.record(s)),saved)

    def test_uninformative_observation_does_not_reveal_sampled_hypothesis(self):
        k=kernel(lambda h,w,a:[HiddenOutcome(1.,{'same':True},effect(),h)])
        s=k.start({},(.3,.7));rows=k(s,'act')
        self.assertEqual(len(rows),1);self.assertEqual(rows[0].state.weights,s.weights)
        self.assertEqual(set(k.record(rows[0].state)),{'format','identity','tick','weights','public'})

    def test_hidden_dynamics_revive_possible_regimes_without_forged_observations(self):
        k=kernel(lambda h,w,a:[HiddenOutcome(.8,{'signal':h},effect(),h),HiddenOutcome(.2,{'signal':h},effect(),str(1-int(h)))])
        s=k.start({},(1.,0.));n=k.reveal(s,'act',{'signal':'0'},effect())
        self.assertEqual(n.weights,(.8,.2))
        self.assertGreater(k.reveal(n,'act',{'signal':'1'},effect()).weights[1],.5)

    def test_forecast_ablation_propagates_drift_but_does_not_condition(self):
        model=lambda h,w,a:[HiddenOutcome(1.,{'signal':h},effect(),str(1-int(h)))]
        k=kernel(model,update=False);s=k.start({},(.25,.75));rows=k(s,'act')
        self.assertTrue(all(t.state.weights==(.75,.25) for t in rows))
        self.assertEqual(sorted(t.probability for t in rows),[.25,.75])

    def test_checkpoint_identity_and_pure_detached_outputs(self):
        k=kernel();s=k.start({'history':[]},(.5,.5));r=json.loads(json.dumps(k.record(s)))
        self.assertEqual(k.restore(r,lambda p:p),s)
        r['public']['history'].append('outside');self.assertEqual(s.public['history'],[])
        other=BeliefKernel(names=k.names,model=sensor_model,public_record=lambda s:s,model_id='sensor-v2',owner='observer')
        with self.assertRaises(ValueError):other.restore(k.record(s),lambda p:p)
        def mutate(h,w,a):
            w['history'].append(h)
            return [HiddenOutcome(1.,w,effect(),h)]
        rows=kernel(mutate)(s,'act');self.assertEqual(s.public['history'],[])
        rows[0].state.public['history'].append('mutated')
        self.assertEqual(len(rows[1].state.public['history']),1)

    def test_invalid_model_declines_complete_distribution(self):
        for p in (-1.,1.1,float('nan'),True):
            k=kernel(lambda h,w,a:[HiddenOutcome(p,{},effect(),h)])
            with self.assertRaises(ValueError):k(k.start({},(.5,.5)),'act')
        k=kernel(branches=1)
        with self.assertRaises(ValueError):k(k.start({},(.5,.5)),'act')

    def test_float_roundoff_does_not_emit_probability_above_one(self):
        k=kernel(lambda h,w,a:[HiddenOutcome(.7,{},effect(),h),HiddenOutcome(.3000000000000001,{},effect(),h)])
        row=k(k.start({},(.65,.3500000000000001)),'act')[0]
        self.assertEqual(row.probability,1.)
        self.assertAlmostEqual(sum(row.state.weights),1.)

    def test_bounded_certain_goal_survives_multistep_float_branching(self):
        # Found by a full episode, not a synthetic selector-only input.
        from tools.run_learning_value import evaluate,cases
        from reflex.laboratory import profiles
        row=evaluate((cases('continuation')[2],profiles()[2],'frozen',[62000]))
        self.assertTrue(-1. <= row['value'] <= 1.)
        self.assertTrue(row['executions'])

    def test_information_cost_has_same_resource_meaning_as_other_losses(self):
        from tools.learning_value_games import LearningGame
        from reflex.laboratory import profiles
        for game,key in (('resources','survey'),('gambling','probe')):
            g=LearningGame(dict(game=game,limit=4),profiles()[0]);s=g.start()
            rows=g.model(g.names[0],s.public,key)
            self.assertTrue(all(r.flow==g.money_flow(-1) for r in rows))
            self.assertTrue(all(r.flow['style']=={} and r.flow['needs']['growth']<0 for r in rows))
        k=kernel(lambda h,w,a:[HiddenOutcome(1.,{},effect(),'alien')])
        with self.assertRaises(ValueError):k(k.start({},(.5,.5)),'act')

    def test_information_is_worth_buying_only_if_later_choice_benefits(self):
        # Analytical independent result: informed payoff .8 - fee versus .4.
        # Entropy reduction is identical for every fee/reward configuration.
        for fee,benefit,expected in ((.1,.8,'inspect'),(.3,.6,'safe'),(.1,.3,'safe')):
            def model(h,w,key):
                if w['tick']==0:
                    if key=='safe':return [HiddenOutcome(1.,dict(tick=2,value=.4),effect(),h)]
                    return [HiddenOutcome(1.,dict(tick=1,signal=h,value=-fee),effect(),h)]
                return [HiddenOutcome(1.,dict(tick=2,value=w['value']+(benefit if key==h else -benefit)),effect(),h)]
            k=kernel(model);s=k.start({'tick':0,'value':0.},(.5,.5))
            def observe(s,state=None):
                c=context('info',[action(a,effect()) for a in (('inspect','safe') if s.public['tick']==0 else ('0','1'))],mode=None)
                c['tick']=s.tick
                if state is not None:c['state']=copy.deepcopy(state)
                return c
            def build(update):
                return ClosedLoop(observe=observe,transitions=kernel(model,update=update),terminal=lambda s:s.public['tick']==2,
                    assess=lambda s:dict(target='goal',status='scored' if s.public['tick']==2 else 'running',value=s.public['value']),record=k.record,
                    request_for=lambda c:request_for(c,'x'),feedback=lambda *args:ModelFeedback(None),unit='turn',target='goal',model_id='test-'+str(update),
                    budget=TreeBudget(horizon=2),max_regret=0.)
            actor=DecisionLoop(observe(s));req=request_for(observe(s),'x');saved=digest(actor.record());cap={}
            ctrl=build(True);result=DecisionLoop.decide_batch([(actor,req)],False,ctrl.planner(actor,req,s,captured=cap))[0]
            self.assertEqual(result['decision']['action_id'],expected)
            self.assertAlmostEqual(cap['node']['purpose'], max(.4,benefit-fee))
            actor.abandon(result['ticket'])
            fresh=DecisionLoop(observe(s));frozen=build(False)
            d=DecisionLoop.decide_batch([(fresh,req)],False,frozen.planner(fresh,req,s))[0]
            self.assertEqual(d['decision']['action_id'],'safe')
            self.assertNotEqual(saved,digest(actor.record()))


if __name__=='__main__':unittest.main()
