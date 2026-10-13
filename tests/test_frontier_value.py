import copy
import unittest
import numpy as np
from reflex.core import digest,Policy,character_key
from reflex.decision_loop import DecisionLoop
from reflex.frontier_value import EndpointEstimate,TerminalMemory,LearnedFrontier,PolicyTail
from reflex.contingent_games import make_bridge
from reflex.integrated_trials import request_for
from reflex.laboratory import profiles
from tools.closed_loop_models import controller


class FrontierValueTests(unittest.TestCase):
    def test_terminal_memory_distinct_episodes_restore_and_support(self):
        m=TerminalMemory([[0.]]*50+[[0.],[0.]],[1.]*50+[-1.,0.],['long']*50+['short','draw'],reference='reflex-v1',neighbors=3)
        e=m.predict([0.]);self.assertEqual(e.outcomes,((1/3,-1.),(1/3,0.),(1/3,1.)))
        self.assertEqual(TerminalMemory.from_record(m.record()).predict([0.]),e)
        self.assertIsNone(m.predict([99.]))
        saved=m.identity;record=m.record();record['values'][0]=0.;self.assertEqual(m.identity,saved)
        with self.assertRaises(ValueError):m.predict([float('nan')])
        with self.assertRaises(ValueError):TerminalMemory([[0.]],[-1.],['x'],reference='p',neighbors=2)
        with self.assertRaises(ValueError):TerminalMemory([[0.],[1.]],[-1.,1.],['x','x'],reference='p',neighbors=1)

    def test_distribution_validation_does_not_confuse_spread_with_confidence(self):
        e=EndpointEstimate(((.5,-1.),(.5,1.)),'uncertain-outcome',1.)
        self.assertEqual(e.settled_fraction,1.)
        for rows in ((),((.2,0.),),((1.,2.),),((True,0.),),((.5,float('nan')),(.5,0.))):
            with self.assertRaises(ValueError):EndpointEstimate(rows,'x',1.)

    def test_identical_numbers_do_not_authorize_another_rules_or_policy_contract(self):
        m=TerminalMemory([[0.],[0.]],[1.,-1.],['a','b'],reference='rules-a/goal-a/features-v1/reflex-v1',neighbors=2)
        def forbidden(*_):raise AssertionError('different contract must decline before feature extraction')
        incompatible=LearnedFrontier(m,forbidden,schema_id='v1',reference='rules-b/goal-a/features-v1/reflex-v1')
        self.assertIsNone(incompatible.evaluate(None,None))
        supported=LearnedFrontier(m,lambda *_:[0.],schema_id='v1',reference=m.record()['reference'])
        self.assertIsNotNone(supported.evaluate(None,None));self.assertNotEqual(supported.identity,incompatible.identity)

    def test_bootstrap_retains_running_status_distribution_and_owner(self):
        b=make_bridge(profiles()[0],dict(game='public_graph',graph=33301,length=4));w=b.start();c=b.observe(w)
        actor=DecisionLoop(c);req=request_for(c,'public_graph');saved=digest(actor.record())
        class Frontier:
            identity='fixture-tail'
            def evaluate(self,world,own):
                own.state['age']=999
                return EndpointEstimate(((.25,-1.),(.75,1.)),'resolved-training-labels',1.)
        ctrl=controller(b,horizon=1,exact=True,frontier=Frontier());cap={}
        ctrl.planner(actor,req,w,captured=cap)([c],[])
        self.assertEqual(digest(actor.record()),saved)
        rows=cap['node']['returns'];self.assertTrue(all(not r[4] and r[3]['status']=='running' and len(r[2])==1 for r in rows))
        self.assertAlmostEqual(sum(r[0]*r[3]['value'] for r in rows),.5)
        self.assertEqual({r[3]['value'] for r in rows},{-1.,1.})
        self.assertIn('frontier',cap['node']['forecast'].audit)
        self.assertTrue(all(x['supported'] for x in cap['node']['frontiers']))
        ctrl.frontier.identity='retrained-fixture';ctrl.planner(actor,req,w,captured=cap)([c],[])
        self.assertNotEqual(cap['node']['stamp'],next(iter(ctrl.cache)))

    def test_unsupported_falls_back_and_terminal_does_not_invoke_predictor(self):
        b=make_bridge(profiles()[0],dict(game='public_graph',graph=33301,length=2));w=b.start();c=b.observe(w)
        actor=DecisionLoop(c);req=request_for(c,'public_graph')
        class Decline:
            identity='unsupported'
            def evaluate(self,*_):return None
        a=controller(b,horizon=1,exact=True);d=controller(b,horizon=1,exact=True,frontier=Decline());ca={};cd={}
        a.planner(actor,req,w,captured=ca)([c],[]);d.planner(actor,req,w,captured=cd)([c],[])
        self.assertEqual(ca['node']['decision'],cd['node']['decision']);self.assertEqual(ca['node']['returns'],cd['node']['returns'])
        class NoCall:
            identity='not-at-terminal'
            def evaluate(self,*_):raise AssertionError('terminal is already known')
        ctrl=controller(b,horizon=2,exact=True,frontier=NoCall());ctrl.planner(actor,req,w)([c],[])

    def test_tail_matches_independent_owned_reflex_replay_and_is_pure(self):
        from reflex.closed_loop import reveal
        b=make_bridge(profiles()[2],dict(game='public_graph',graph=33303,length=4));w=b.start();c=b.observe(w)
        p=Policy(principle_priority='personal',commitments={character_key(c):.5});actor=DecisionLoop(c,policy=p)
        ctrl=controller(b,horizon=1,exact=True);seeds=(701,702,703,704)
        tail=PolicyTail(observe=b.observe,advance=b.modeled_advance,terminal=b.terminal,assess=lambda w:b.goal(w).record(),request_for=ctrl.request_for,feedback=ctrl.feedback,model_id=ctrl.model_id,target=b.target,seeds=seeds,steps=4)
        saved=digest(actor.record());estimate=tail.evaluate(w,actor);values=[]
        for seed in seeds:
            own=copy.deepcopy(actor);world=copy.deepcopy(w)
            while not b.terminal(world):
                cc=b.observe(world,own.state);rq=request_for(cc,'public_graph');effective=own._prepare(rq)['effective']
                d=own.decide(rq,False);key=d['decision']['action_id'];after,flow=b.modeled_advance(world,key,seed)
                reveal(own,d['ticket'],ctrl.feedback(world,after,key,effective,flow));world=after
            values.append(b.goal(world).value)
        self.assertAlmostEqual(sum(p*v for p,v in estimate.outcomes),np.mean(values));self.assertEqual(estimate.settled_fraction,1.)
        self.assertEqual(digest(actor.record()),saved)
        short=PolicyTail(observe=b.observe,advance=b.modeled_advance,terminal=b.terminal,assess=lambda w:b.goal(w).record(),request_for=ctrl.request_for,feedback=ctrl.feedback,model_id=ctrl.model_id,target=b.target,seeds=seeds,steps=1)
        self.assertEqual(short.evaluate(w,actor).settled_fraction,0.)

    def test_tail_cache_keeps_owner_seed_and_remaining_deadline(self):
        b=make_bridge(profiles()[2],dict(game='public_graph',graph=33303,length=5));w=b.start();actor=DecisionLoop(b.observe(w));ctrl=controller(b,horizon=1,exact=True)
        args=dict(observe=b.observe,advance=b.modeled_advance,terminal=b.terminal,assess=lambda w:b.goal(w).record(),request_for=ctrl.request_for,feedback=ctrl.feedback,model_id=ctrl.model_id,target=b.target,seeds=(701,702),steps=4)
        cold=PolicyTail(**args);cached=PolicyTail(**args,record=b.record,cache_entries=16)
        for _ in range(2):self.assertEqual(cold.evaluate(w,actor),cached.evaluate(w,actor))
        self.assertGreater(cached.cache_hits,0)
        own=copy.deepcopy(actor);c=b.observe(w,own.state);d=own.decide(request_for(c,'public_graph'),False);own.abandon(d['ticket'])
        later,_=b.modeled_advance(w,d['decision']['action_id'],701)
        self.assertEqual(cold.evaluate(later,own),cached.evaluate(later,own));self.assertLessEqual(len(cached.cache),16)
        cached.cache.clear();self.assertEqual(cold.evaluate(later,own),cached.evaluate(later,own))


if __name__=='__main__':unittest.main()
