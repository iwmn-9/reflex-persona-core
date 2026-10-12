import copy
import unittest
import numpy as np
from reflex.core import Policy,compile_batch,character_key,digest
from reflex.examples import context,action,effect
from reflex.decision_loop import DecisionLoop
from reflex.closed_loop import DecisionWindow,TreeBudget,Transition
from reflex.contingent_games import make_bridge
from reflex.integrated_trials import request_for
from reflex.laboratory import profiles
from tools.closed_loop_models import controller


class ClosedLoopTests(unittest.TestCase):
    def test_wide_distributions_preserve_loss_and_default_capacity(self):
        a=action('gamble',effect(.8,p=.5),effect(-.6,p=.5));c=context('wide',[a,action('safe',effect(.05))])
        p=Policy();baseline=p.decide(compile_batch([c]),False)
        wide=copy.deepcopy(c)
        wide['actions'][0]['outcomes']=[dict(copy.deepcopy(row),p=row['p']/8) for row in a['outcomes'] for _ in range(8)]
        with self.assertRaises(ValueError):compile_batch([wide])
        d=p.decide(compile_batch([wide],outcome_limit=16),False)
        np.testing.assert_allclose(d.scores,baseline.scores,atol=1e-14)
        np.testing.assert_allclose(d.features,baseline.features,atol=1e-14)
        for bad in (True,0,4097):
            with self.assertRaises(ValueError):compile_batch([wide],outcome_limit=bad)

    def test_exact_closed_loop_matches_all_real_branches_and_restore(self):
        b=make_bridge(profiles()[0],dict(game='public_graph',graph=23100,length=4));w=b.start();c=b.observe(w)
        p=Policy(principle_priority='personal',commitments={character_key(c):.5});owner=DecisionLoop(c,policy=p)
        ctrl=controller(b,horizon=4,exact=True,max_regret=0);saved=digest(owner.record());capture={}
        req=request_for(c,'public_graph');result=DecisionLoop.decide_batch([(owner,req)],False,ctrl.planner(owner,req,w,captured=capture))[0]
        owner.abandon(result['ticket']);node=capture['node'];self.assertTrue(capture['audit']['complete'])
        self.assertNotEqual(digest(owner.record()),saved)
        def actual(world,actor,remaining,node):
            if b.terminal(world):return b.goal(world).value
            cc=b.observe(world,actor.state);rq=request_for(cc,'public_graph');got={}
            result=DecisionLoop.decide_batch([(actor,rq)],False,ctrl.planner(actor,rq,world,remaining=remaining,captured=got))[0]
            self.assertEqual(result['decision']['action_id'],node['action']);actor.abandon(result['ticket'])
            expected=0.
            for t in ctrl.transitions(world,node['action']):
                child=next(x for x in node['children'][node['action']] if x['public']==b.record(t.state))
                restored=DecisionLoop.from_record(cc,actor.record(),policy=p)
                self.assertEqual(digest(restored.record()),child['owner'])
                expected+=t.probability*(b.goal(t.state).value if b.terminal(t.state) else actual(t.state,restored,remaining-1,ctrl.cache[child['child']]))
            return expected
        initial=DecisionLoop(c,policy=p);value=actual(w,initial,4,node)
        predicted=sum(p*g['value'] for p,flows,actions,g,ended in node['returns'])
        self.assertAlmostEqual(value,predicted)
        from tools.run_foundation_audit import graph_values
        self.assertAlmostEqual((value+1)/2,graph_values(b.rules)[0][0,0])

    def test_budget_declines_whole_proposal_without_owner_mutation(self):
        b=make_bridge(profiles()[0],dict(game='public_graph',graph=23100,length=5));w=b.start();c=b.observe(w)
        for options in (dict(nodes=1),dict(outcomes=1)):
            owner=DecisionLoop(c);req=request_for(c,'public_graph');saved=digest(owner.record());ctrl=controller(b,horizon=4,exact=True,**options)
            capture={'node':{'action':'stale-root'}}
            self.assertIsNone(ctrl.planner(owner,req,w,captured=capture)([c],[]));self.assertFalse(ctrl.last['complete']);self.assertEqual(digest(owner.record()),saved)
            self.assertNotIn('node',capture)
            self.assertIsNone(ctrl.planner(owner,req,w)([c],[]));self.assertFalse(ctrl.last['complete'])

    def test_warm_and_cold_tree_and_changed_request_contract(self):
        from dataclasses import replace
        b=make_bridge(profiles()[2],dict(game='public_graph',graph=25101,length=4));w=b.start();c=b.observe(w)
        actor=DecisionLoop(c);req=request_for(c,'public_graph');ctrl=controller(b,horizon=4,exact=True)
        cap={};ctrl.planner(actor,req,w,captured=cap)([c],[]);first=cap['node'];work=ctrl.last['nodes']
        ctrl.planner(actor,req,w,captured=cap)([c],[])
        self.assertEqual(ctrl.last['nodes'],work);self.assertEqual(cap['node']['decision'],first['decision'])
        ctrl.clear();ctrl.planner(actor,req,w,captured=cap)([c],[])
        self.assertEqual(cap['node']['decision'],first['decision']);self.assertEqual(cap['node']['returns'],first['returns'])
        changed=replace(req,threatened=True);ctrl.planner(actor,changed,w,captured=cap)([actor._prepare(changed)['effective']],[])
        self.assertNotEqual(cap['node']['stamp'],first['stamp'])

    def test_window_preserves_deadline_and_goal_after_checkpoint(self):
        c=context('window',[action('act',effect(.2))]);window=DecisionWindow(c,3)
        for i,expected in enumerate((3,2,1,3,2)):
            c['tick']=i*7;self.assertEqual(window.remaining(c),expected);self.assertEqual(window.remaining(c),expected)
            window.advance(c);window=DecisionWindow.from_record(c,window.record())
            with self.assertRaises(ValueError):window.advance(c)
        changed=copy.deepcopy(c);changed['tick']+=1;changed['objective']='other goal'
        with self.assertRaises(ValueError):window.remaining(changed)

    def test_normalized_kernel_required_and_real_owner_preserved(self):
        b=make_bridge(profiles()[0],dict(game='public_graph',graph=23100,length=2));w=b.start();c=b.observe(w)
        owner=DecisionLoop(c);req=request_for(c,'public_graph');ctrl=controller(b,horizon=2,exact=True);saved=digest(owner.record())
        original=ctrl.transitions
        ctrl.transitions=lambda s,a:[Transition(.9,t.state,t.flow) for t in original(s,a)]
        with self.assertRaises(ValueError):ctrl.planner(owner,req,w)([c],[])
        self.assertEqual(digest(owner.record()),saved)

    def test_gambling_public_kernel_matches_real_transition_and_flow(self):
        from tools.closed_loop_models import gambling_kernel
        from reflex.gambling import Table,offers
        from reflex.contingent_games import Frame
        for regime in ('opportunity','unfavorable'):
            b=make_bridge(profiles()[0],dict(game='gambling',scenario=regime));kernel=gambling_kernel(b)
            for cash in (2,5,12,17):
                w=Frame((Table(cash=cash,tick=2),regime),2)
                for a in b.observe(w)['actions']:
                    rows=kernel(w,a['id']);self.assertAlmostEqual(sum(t.probability for t in rows),1.)
                    if a['id']!='pass':self.assertEqual(rows[0].probability,offers(w.position[0],regime)[a['id']][2])
                    for seed in range(32):
                        after,flow=b.modeled_advance(w,a['id'],seed)
                        self.assertTrue(any(t.state==after and t.flow==flow for t in rows))

    def test_owned_learning_social_and_progress_survive_modeled_and_real_steps(self):
        from tools.closed_loop_models import controller
        from reflex.planning import vector
        from reflex.closed_loop import ClosedLoop,reveal
        from reflex.model_observer import ModelFeedback
        from reflex.progress import PurposeFeedback
        from reflex.social import SocialEvent
        from dataclasses import replace
        from test_progress_social_composition import fixture
        # Real empirical updates in a separate existing rule family.
        b=make_bridge(profiles()[1],dict(game='cyclic_arena'));w=b.start();c=b.observe(b.project(w));owner=DecisionLoop(c)
        opening=copy.deepcopy(c);opening['actions']=[a for a in c['actions'] if a['id']=='STRIKE']
        opening_req=request_for(opening,'cyclic_arena',learn=True)
        initial=owner.decide(replace(opening_req,exact={'STRIKE':opening_req.exact['STRIKE']}),False)
        w,row=b.actual(w,'STRIKE',51100);owner.observe(initial['ticket'],vector(row))
        self.assertTrue(owner.memory.entries)
        ctrl=controller(b,horizon=2,learn=True)
        for _ in range(3):
            c=b.observe(b.project(w),owner.state);req=request_for(c,'cyclic_arena',learn=True);cap={}
            before=digest(owner.record());callback=ctrl.planner(owner,req,b.project(w),captured=cap)
            callback([owner._prepare(req)['effective']],[]);self.assertEqual(digest(owner.record()),before)
            result=DecisionLoop.decide_batch([(owner,req)],False,callback)[0];key=result['decision']['action_id']
            w,row=b.actual(w,key,51100)
            if key!='CHECK':owner.observe(result['ticket'],vector(row))
            else:owner.abandon(result['ticket'])
        self.assertTrue(owner.memory.entries)
        # Existing proof-recovery + social regression fixture, now in a tree.
        c,owner,req=fixture('tree-owned');start=copy.deepcopy(c)
        def observe(t,memory):
            cc=copy.deepcopy(start);cc['tick']=8+t;cc['state']=copy.deepcopy(memory);return cc
        def request(cc):return replace(req,context=cc)
        def transitions(t,key):return [Transition(1.,t+1,effect(.1))]
        def feedback(before,after,key,cc,row):
            return ModelFeedback(PurposeFeedback(0.,0.),social_events=(SocialEvent('friend','tree-'+str(before),cc['tick'],.1),))
        ctrl=ClosedLoop(observe=observe,transitions=transitions,terminal=lambda t:t>=2,
            assess=lambda t:dict(target='goal',status='scored' if t>=2 else 'running',value=0.),record=lambda t:t,
            request_for=request,feedback=feedback,unit='turn',target='goal',model_id='fixture-v1',budget=TreeBudget(horizon=2))
        c1=observe(0,owner.state);rq=request(c1);cap={};saved=digest(owner.record())
        result=DecisionLoop.decide_batch([(owner,rq)],False,ctrl.planner(owner,rq,0,captured=cap))[0]
        key=result['decision']['action_id'];reveal(owner,result['ticket'],feedback(0,1,key,c1,effect(.1)))
        self.assertNotEqual(saved,digest(owner.record()))
        child=cap['node']['children'][key][0];self.assertEqual(child['owner'],digest(owner.record()))
        c2=observe(1,owner.state);rq2=request(c2)
        result2=DecisionLoop.decide_batch([(owner,rq2)],False,ctrl.planner(owner,rq2,1,remaining=1))[0]
        self.assertEqual(result2['decision']['action_id'],ctrl.cache[child['child']]['action'])


if __name__=='__main__':unittest.main()
