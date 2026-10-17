import copy
import unittest
from unittest.mock import patch
from reflex.core import digest
from reflex.closed_loop import TreeBudget
from reflex.decision_loop import DecisionLoop
from reflex.laboratory import PROFILES
from reflex.model_observer import resolve_feedback
from tools.continuing_games import ContinuingGame
from tools.run_owned_value import sample
from reflex.closed_loop import ClosedLoop,Transition
from reflex.frontier_value import PolicyTail
import numpy as np
from reflex.closed_loop import owned_rollout


def controller(g,*,steps,tail_seeds=(701,702,703),future_choices='planned',tail_preferences=True,world_law=None):
    # A model of the world is fixed for this forecast. The imagined learner
    # changes its beliefs, not the world's generating law. At the next REAL
    # decision a new root law is made from real evidence.
    world_law=copy.deepcopy(world_law)
    model_id=digest([g.model_id,world_law])
    def probabilities(own,c):return own.predictor.predict(c) if world_law is None else world_law
    def transitions(w,k,own,c):
        return [Transition(p,*g.transition(w,k,z)) for z,p in probabilities(own,c).items() if p>0]
    def advance(w,k,seed,own,c):
        u=np.random.default_rng(np.random.SeedSequence([seed,w['tick']])).random()
        return g.transition(w,k,sample(probabilities(own,c),u))
    tail=PolicyTail(observe=g.context,advance=lambda *args:None,owner_advance=advance,
        terminal=g.terminal,assess=lambda w:g.goal(w).record(),request_for=g.request,
        feedback=g.feedback,model_id=model_id,target=g.target,seeds=tail_seeds,steps=steps)
    return ClosedLoop(observe=g.context,transitions=lambda *args:None,owner_transitions=transitions,
        terminal=g.terminal,assess=lambda w:g.goal(w).record(),record=lambda w:w,
        request_for=g.request,feedback=g.feedback,unit='owner-decisions',target=g.target,
        model_id=model_id,budget=TreeBudget(horizon=1,nodes=1024,outcomes=128),frontier=tail,
        future_choices=future_choices,opportunity_regret=True,relative_objective=True,tail_preferences=tail_preferences)


class OwnedValueLoopTests(unittest.TestCase):
    def test_exact_reflex_continuation_pruning_preserves_full_tree_value(self):
        for game in ('auction','combat'):
            g=ContinuingGame(game,PROFILES[1],turns=3);w=g.start();c=g.context(w);own=g.actor(c,learning='segmented')
            args=dict(observe=g.context,events=g.events,terminal=g.terminal,assess=lambda w:g.goal(w).record(),
                record=lambda w:w,request_for=g.request,feedback=g.feedback,model_id=g.model_id,target=g.target)
            ctrl=owned_rollout(own,**args,tail_steps=0,tree_steps=2,future_choices='reflex');rq=g.request(c);cap={}
            effective=own.preview(rq)[0];ctrl.planner(own,rq,w,captured=cap)([effective],[])
            law=own.predictor.frozen_forecaster();f=cap['node']['forecast']
            for plan,(key,) in f.roots.items():
                expected=0.
                for z,p in law(effective).items():
                    child=copy.deepcopy(own);first=DecisionLoop.decide_batch([(child,rq)],False,_counterfactual_roots=(key,))[0]
                    state,flow=g.transition(w,key,z);resolve_feedback(child,first['ticket'],g.feedback(w,state,key,effective,flow))
                    cc=g.context(state,child.state);last=DecisionLoop.decide_batch([(child,g.request(cc))],False)[0]
                    expected+=p*sum(q*g.goal(g.transition(state,last['decision']['action_id'],symbol)[0]).value
                        for symbol,q in law(last['context']).items())
                self.assertAlmostEqual(f.purpose[plan],expected,places=13)

    def test_imagined_owner_can_recover_without_a_positive_immediate_action(self):
        g=ContinuingGame('combat',PROFILES[1],turns=16);w=g.start();w['ammo']=0;own=g.actor(g.context(w),learning='segmented')
        args=dict(observe=g.context,events=g.events,terminal=g.terminal,assess=lambda w:g.goal(w).record(),
            record=lambda w:w,request_for=g.request,feedback=g.feedback,model_id=g.model_id,target=g.target,
            tail_steps=8,seeds=(701,),world_model='posterior')
        saved=own.record();plain=owned_rollout(own,**args);recovered=owned_rollout(own,**args,tail_recovery=2)
        a=plain.frontier.trajectories(w,own);b=recovered.frontier.trajectories(w,own)
        self.assertGreater(recovered.frontier.future_calls,0)
        self.assertGreater(sum(p*goal['value'] for p,_,_,goal,_ in b),sum(p*goal['value'] for p,_,_,goal,_ in a))
        self.assertEqual(own.record(),saved)

    def test_conditional_mean_matches_exact_tree_without_erasing_sampled_risk(self):
        g=ContinuingGame('gambling',PROFILES[1],turns=2);w=g.start();c=g.context(w);actor=g.actor(c)
        args=dict(observe=g.context,events=g.events,terminal=g.terminal,
            assess=lambda w:g.goal(w).record(),record=lambda w:w,request_for=g.request,
            feedback=g.feedback,model_id=g.model_id,target=g.target,tail_steps=1,seeds=(817,))
        rq=g.request(c);effective=actor.preview(rq)[0];captures=[]
        for estimator in ('sampled','conditional'):
            ctrl=owned_rollout(actor,**args,purpose_estimator=estimator);cap={}
            ctrl.planner(actor,rq,w,captured=cap)([effective],[]);captures.append(cap)
        raw,corrected=[x['node']['forecast'] for x in captures]
        # Only the mean used by the competence gate changes. Adverse branches
        # remain visible to the personality's risk evaluation.
        self.assertEqual(raw.contexts,corrected.contexts)
        self.assertEqual(raw.audit['endpoint'],corrected.audit['endpoint'])
        law=actor.predictor.frozen_forecaster();p=law(c)
        for plan,root in corrected.roots.items():
            expected=0.
            for z,weight in p.items():
                person=copy.deepcopy(actor)
                result=DecisionLoop.decide_batch([(person,rq)],False,_counterfactual_roots=(root[0],))[0]
                after,flow=g.transition(w,root[0],z)
                resolve_feedback(person,result['ticket'],g.feedback(w,after,root[0],c,flow))
                cc=g.context(after,person.state);future=person.preview(g.request(cc))[1]['action_id']
                expected+=weight*sum(prob*g.goal(g.transition(after,future,symbol)[0]).value for symbol,prob in law(cc).items())
            self.assertAlmostEqual(corrected.purpose[plan],expected,places=13)
        self.assertTrue(any(abs(raw.purpose[k]-corrected.purpose[k])>1e-8 for k in corrected.purpose))

    def test_new_prediction_and_sampling_use_same_real_owner_lifecycle(self):
        g=ContinuingGame('auction',PROFILES[2],turns=5);w=g.start();c=g.context(w)
        own=g.actor(c,renewal=True);saved=own.record();rq=g.request(c)
        args=dict(observe=g.context,events=g.events,terminal=g.terminal,
            assess=lambda w:g.goal(w).record(),record=lambda w:w,request_for=g.request,
            feedback=g.feedback,model_id=g.model_id,target=g.target,tail_steps=4,sampling='stratified',purpose_estimator='conditional')
        ctrl=owned_rollout(own,**args);cap={};effective=own.preview(rq)[0]
        ctrl.planner(own,rq,w,captured=cap)([effective],[])
        self.assertEqual(own.record(),saved)
        clone=DecisionLoop.from_record(c,saved,predictor=g.actor(c,renewal=True).predictor)
        other=owned_rollout(clone,**args);cap2={}
        other.planner(clone,rq,w,captured=cap2)([effective],[])
        self.assertEqual(cap['node']['forecast'].purpose,cap2['node']['forecast'].purpose)
        result=DecisionLoop.decide_batch([(own,rq)],False,ctrl.planner(own,rq,w))[0]
        key=result['decision']['action_id'];end,flow=g.transition(w,key,'2')
        resolve_feedback(own,result['ticket'],g.feedback(w,end,key,c,flow))
        expected=next(x['owner'] for x in cap['node']['children'][key] if x['public']==end)
        self.assertEqual(digest(own.record()),expected)

    def test_exact_augmented_tree_replays_learning_owner_on_every_branch(self):
        g=ContinuingGame('gambling',PROFILES[0],turns=2)
        w=g.start();c=g.context(w);actor=g.actor(c)
        ctrl=controller(g,steps=1,tail_preferences=False,world_law=actor.predictor.predict(c))
        ctrl.frontier=None;ctrl.budget=TreeBudget(horizon=2,outcomes=128)
        saved=digest(actor.record());cap={};rq=g.request(c)
        effective,preview=actor.preview(rq)
        ctrl.planner(actor,rq,w,captured=cap)([effective],[])
        self.assertEqual(digest(actor.record()),saved)

        def replay(w,own,node,left):
            c=g.context(w,own.state);rq=g.request(c);captured={}
            result=DecisionLoop.decide_batch([(own,rq)],False,
                ctrl.planner(own,rq,w,remaining=left,captured=captured))[0]
            self.assertEqual(result['decision']['action_id'],node['action'])
            value=0.
            for child in node['children'][node['action']]:
                after=child['public'];clone=copy.deepcopy(own)
                flow=g.transition(w,node['action'],after['signal'])[1]
                resolve_feedback(clone,result['ticket'],g.feedback(w,after,node['action'],c,flow))
                self.assertEqual(digest(clone.record()),child['owner'])
                self.assertEqual(clone.predictor.observations,after['tick'])
                record=clone.record();restored=DecisionLoop.from_record(c,record,predictor=clone.predictor)
                self.assertEqual(record,restored.record())
                future=g.goal(after).value if g.terminal(after) else replay(after,restored,ctrl.cache[child['child']],left-1)
                value+=child['probability']*future
            return value
        node=cap['node'];actual=replay(w,copy.deepcopy(actor),node,2)
        expected=sum(row[0]*row[3]['value'] for row in node['returns'])
        self.assertAlmostEqual(actual,expected)

    def test_tail_uses_real_lifecycle_and_cannot_train_real_owner(self):
        g=ContinuingGame('resources',PROFILES[1],turns=8)
        w=g.start();actor=g.actor(g.context(w));saved=digest(actor.record());ctrl=controller(g,steps=8,tail_seeds=(701,))
        seen=[];original=ctrl.frontier.owner_advance
        def advance(w,k,seed,own,c):
            self.assertIn('need_pressure',c['facts'])
            self.assertIsNone(own.pending)
            seen.append((w['tick'],own.predictor.observations,len(own.memory.entries)))
            return original(w,k,seed,own,c)
        ctrl.frontier.owner_advance=advance
        estimate=ctrl.frontier.evaluate(w,actor)
        self.assertEqual(estimate.settled_fraction,1.)
        self.assertEqual([x[1] for x in seen],list(range(8)))
        self.assertGreater(seen[-1][2],0)
        self.assertEqual(digest(actor.record()),saved)

    def test_preview_contains_reading_and_is_pure(self):
        g=ContinuingGame('auction',PROFILES[2],turns=4)
        w=g.start();c=g.context(w);own=g.actor(c);saved=digest(own.record())
        effective,d=own.preview(g.request(c))
        clone=copy.deepcopy(own);result=DecisionLoop.decide_batch([(clone,g.request(c))],False)[0]
        self.assertEqual(effective,result['context'])
        self.assertEqual(digest(own.record()),saved)
        self.assertIn(d['action_id'],g.keys(w))

    def test_future_root_changes_have_real_persistent_cost(self):
        for game,key in [('resources','invest'),('gambling','bond'),('auction','bid:4')]:
            g=ContinuingGame(game,PROFILES[0],turns=128);w=g.start()
            after,_=g.transition(w,key,'0')
            self.assertLess(after['cash'],w['cash'])
            self.assertFalse(g.terminal(after))
            self.assertEqual(after['tick'],1)

    def test_public_factory_is_parity_checked_and_stale_models_rejected(self):
        g=ContinuingGame('auction',PROFILES[1],turns=3);w=g.start();c=g.context(w);own=g.actor(c);rq=g.request(c)
        ctrl=owned_rollout(own,observe=g.context,events=g.events,terminal=g.terminal,
            assess=lambda w:g.goal(w).record(),record=lambda w:w,request_for=g.request,
            feedback=g.feedback,model_id=g.model_id,target=g.target,tail_steps=3)
        old=controller(g,steps=3,world_law=own.predictor.predict(c));effective=own.preview(rq)[0]
        left={};right={};ctrl.planner(own,rq,w,captured=left)([effective],[])
        old.planner(own,rq,w,captured=right)([effective],[])
        self.assertEqual(left['node']['action'],right['node']['action'])
        self.assertEqual(left['node']['forecast'].purpose,right['node']['forecast'].purpose)
        call=ctrl.planner(own,rq,w)
        result=DecisionLoop.decide_batch([(own,rq)],False,call)[0]
        n,flow=g.transition(w,result['decision']['action_id'],'2')
        resolve_feedback(own,result['ticket'],g.feedback(w,n,result['decision']['action_id'],c,flow))
        with self.assertRaisesRegex(ValueError,'stale'):call([effective],[])

    def test_long_transport_preserves_cumulative_preference_units(self):
        from reflex.examples import context,action,effect
        from reflex.intertemporal import Branch,forecast
        from reflex.core import compile_batch,Policy
        c=context('long',[action('a',effect()),action('b',effect())])
        paths={k:[Branch(1.,tuple(effect(.01 if k=='a' else 0.) for _ in range(64)),(1.,)*64)] for k in ('a','b')}
        f=forecast(c,paths,horizon=64,unit='decisions',target='test',discount=1.)
        d=Policy().decide(compile_batch(f.contexts),False,return_frame=f.return_frame)
        self.assertEqual(f.return_frame.effects[0][0],64.)
        self.assertGreater(d.scores[0,0],d.scores[0,1])

    def test_frozen_world_law_stays_fixed_while_real_person_learns(self):
        g=ContinuingGame('gambling',PROFILES[0],turns=24);w=g.start();c=g.context(w);own=g.actor(c)
        frozen=own.predictor.frozen_forecaster();original=frozen(c)
        for _ in range(8):
            c=g.context(w,own.state);result=own.decide(g.request(c),False);key=result['decision']['action_id']
            n,flow=g.transition(w,key,'2');resolve_feedback(own,result['ticket'],g.feedback(w,n,key,c,flow));w=n
        c=g.context(w,own.state)
        self.assertEqual(frozen(c),original)
        self.assertGreater(own.predictor.predict(c)['2'],original['2'])
        new=own.predictor.frozen_forecaster()
        self.assertEqual(new(c),own.predictor.predict(c))
        tampered=new(c);tampered['2']=0
        self.assertEqual(new(c),own.predictor.predict(c))

    def test_tail_replanning_replays_the_actual_learning_controller(self):
        g=ContinuingGame('resources',PROFILES[1],turns=4);w=g.start();own=g.actor(g.context(w))
        args=dict(observe=g.context,events=g.events,terminal=g.terminal,
            assess=lambda w:g.goal(w).record(),record=lambda w:w,request_for=g.request,
            feedback=g.feedback,model_id=g.model_id,target=g.target)
        seen=[]
        def future(actor,req,state):
            if state['tick']!=2:return None
            seen.append(actor.predictor.observations)
            return owned_rollout(actor,**args,tail_steps=1,seeds=(1701,)).planner(actor,req,state)
        ctrl=owned_rollout(own,**args,tail_steps=4,seeds=(701,),
            future_planner=future,future_planner_id='one-public-replan',future_planner_limit=1)
        saved=digest(own.record());resolved=[]
        def audited_resolve(actor,ticket,feedback):
            result=resolve_feedback(actor,ticket,feedback)
            resolved.append(digest(actor.record()))
            return result
        with patch('reflex.model_observer.resolve_feedback',audited_resolve):
            rows=ctrl.frontier.trajectories(w,own)
        self.assertEqual(seen,[2]);self.assertEqual(digest(own.record()),saved)
        replay=copy.deepcopy(own);law=own.predictor.frozen_forecaster()
        for expected in rows[0][2]:
            c=g.context(w,replay.state);req=g.request(c);plan=future(copy.deepcopy(replay),req,w)
            result=DecisionLoop.decide_batch([(replay,req)],False,plan)[0]
            self.assertEqual(result['decision']['action_id'],expected)
            u=np.random.default_rng(np.random.SeedSequence([701,w['tick']])).random()
            n,flow=g.transition(w,expected,sample(law(c),u))
            resolve_feedback(replay,result['ticket'],g.feedback(w,n,expected,c,flow));w=n
        self.assertEqual(g.goal(w).record(),rows[0][3])
        self.assertEqual(digest(replay.record()),resolved[-1])
        self.assertEqual(ctrl.frontier.future_calls,1)


if __name__=='__main__':unittest.main()
