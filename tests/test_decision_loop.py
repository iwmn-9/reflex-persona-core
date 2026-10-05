import copy
import json
import unittest
import numpy as np
from reflex.core import FEATURES,Policy,compile_batch,digest
from reflex.examples import context,action,effect
from reflex.judgment import Binding,ReadControl
from reflex.planning import vector
from reflex.decision_loop import DecisionLoop,Request,IntentRequest,Reading,EvidenceGate,cdf_score
from reflex.loop_predictor import CategoricalReader


def fixture(npc='actor',negative=False):
    c=context('loop',[action('A',effect(-.6 if negative else .8),confidence=.7),
                      action('B',effect(-.5 if negative else .3),confidence=.7)])
    c['scope']['npc']=npc
    for n in c['needs']:c['needs'][n]=dict(supported=False,enabled=False,deficit=None)
    return c


def request(c,**kwargs):
    return Request(c,{a['id']:Binding(a['id'],'condition',('objective',)) for a in c['actions']},
                   {a['id']:('cost',) for a in c['actions']},**kwargs)


class LoopTests(unittest.TestCase):
    def test_bounded_experience_preserves_existing_adaptation_without_certification_delay(self):
        c=fixture();loop=DecisionLoop(c);states=[];actions=[]
        for tick in range(8):
            c['tick']=tick;r=loop.decide(request(c),False)
            states.append(r['learning']['A']['gate']['state'])
            actions.append(r['decision']['action_id'])
            observed=effect(-.8 if r['decision']['action_id']=='A' else .3)
            loop.observe(r['ticket'],vector(observed))
        self.assertEqual(states[:3],['unobserved']*3)
        self.assertEqual(actions[:3],['A','A','B'])
        self.assertEqual(loop.memory.entries[('A','condition')]['count'],2)
        self.assertEqual(r['decision']['action_id'],'B')
        self.assertEqual(c['personality'],loop.personality);self.assertEqual(c['values'],loop.values)

    def test_new_bad_forecasts_revoke_old_evidence_and_good_ones_recover(self):
        g=EvidenceGate();prior=[effect(0)];good=[effect(.8)]
        for _ in range(4):g.compare('source',prior,good,vector(effect(.8)),('objective',))
        self.assertTrue(g.accepted('source'))
        for _ in range(4):r=g.compare('source',prior,good,vector(effect(0)),('objective',))
        self.assertEqual(g.status('source')['state'],'revoked')
        for _ in range(12):g.compare('source',prior,good,vector(effect(.8)),('objective',))
        self.assertTrue(g.accepted('source'))

    def test_lottery_scores_probabilities_and_one_loss_is_not_proof(self):
        prior=[effect(.8,p=.9),effect(-.8,p=.1)]
        optimistic=[effect(.8)]
        self.assertLess(cdf_score(prior,vector(effect(-.8)),('objective',)),cdf_score(optimistic,vector(effect(-.8)),('objective',)))
        g=EvidenceGate()
        for v in (.8,.8,-.8):g.compare('gamble',prior,optimistic,vector(effect(v)),('objective',))
        self.assertFalse(g.accepted('gamble'));self.assertNotEqual(g.status('gamble')['state'],'revoked')
        before=g.status('gamble');g.compare('gamble',prior,prior,vector(effect(.8)),('objective',))
        self.assertEqual(before,g.status('gamble'))

    def test_batch_matches_individual_reordered_and_intent_is_internal(self):
        cs=[fixture(str(i)) for i in range(7)]
        batches=[DecisionLoop(c) for c in cs];single=[DecisionLoop(c) for c in cs]
        for tick in range(4):
            for c in cs:c['tick']=tick
            rr=DecisionLoop.decide_batch([(batches[i],request(cs[i])) for i in reversed(range(7))])
            for i,r in zip(reversed(range(7)),rr):
                other=single[i].decide(request(cs[i]));self.assertEqual(r,other)
                outcome=vector(effect(.2));batches[i].observe(r['ticket'],outcome);single[i].observe(other['ticket'],outcome)
                self.assertEqual(batches[i].record(),single[i].record())

    def test_invalid_last_actor_or_reader_cannot_partially_commit_batch(self):
        a,b=fixture('a',True),fixture('b',True);la,lb=DecisionLoop(a),DecisionLoop(b)
        prior=(la.record(),lb.record());bad=copy.deepcopy(b);bad['personality']['openness']=.9
        with self.assertRaises(ValueError):DecisionLoop.decide_batch([(la,request(a)),(lb,request(bad))])
        self.assertEqual(prior,(la.record(),lb.record()))
        def invalid(c,cap):
            c['actions'][0]['legal']=False;return Reading(c,1,'invalid')
        with self.assertRaisesRegex(ValueError,'legality'):DecisionLoop.decide_batch([(la,request(a)),(lb,request(b,reader=invalid,threatened=True))])
        self.assertEqual(prior,(la.record(),lb.record()))

    def test_reader_cannot_change_exact_cost_target_or_budget(self):
        for invalid in ('cost','target','budget'):
            c=fixture(negative=True);loop=DecisionLoop(c)
            def reader(c,cap):
                if invalid=='cost':c['actions'][0]['outcomes'][0]['cost']=.1
                return Reading(c,cap+1 if invalid=='budget' else 1,'test','terminal' if invalid=='target' else 'immediate')
            with self.assertRaises(ValueError):loop.decide(request(c,reader=reader,threatened=True))
            self.assertEqual(loop.last_tick,-1)

    def test_reader_good_for_one_method_cannot_authorize_untried_bad_method(self):
        c=fixture();c['actions'][0]['outcomes']=[effect(.3)];c['actions'][1]['outcomes']=[effect(-.6)]
        loop=DecisionLoop(c)
        def reader(c,cap):
            for a in c['actions']:a['outcomes']=[effect(.8 if a['id']=='A' else 1)]
            return Reading(c,2,'partly-wrong')
        blocked=False
        for tick in range(12):
            c['tick']=tick;r=loop.decide(request(c,reader=reader,threatened=True),False)
            self.assertEqual(r['decision']['action_id'],'A')
            blocked|='B' in r['reading'].get('unvalidated_roots',[])
            loop.observe(r['ticket'],vector(effect(.8)))
        self.assertTrue(blocked)

    def test_stochastic_exact_rules_survive_experience_and_reader_validation(self):
        c=fixture();prior=[effect(.8,cost=0,p=.5),effect(-.8,cost=1,p=.5)]
        c['actions']=[action('A',*prior)];loop=DecisionLoop(c)
        for tick in range(4):
            c['tick']=tick;r=loop.decide(request(c),False)
            loop.observe(r['ticket'],vector(effect(.1*(tick+1),cost=tick%2)))
        def reader(observed,cap):
            # A reader that restores the original exact lottery must remain
            # admissible after estimated effects have acquired experience.
            observed['actions'][0]['outcomes']=copy.deepcopy(prior)
            return Reading(observed,2,'same-exact-rule')
        c['tick']=4
        r=loop.decide(request(c,reader=reader,threatened=True),False)
        self.assertEqual(r['reading']['nodes'],2)
        self.assertTrue(all(row['cost'] in (0,1) for row in r['context']['actions'][0]['outcomes']))
        loop.observe(r['ticket'],vector(effect(.5,cost=1)))
        saved=json.loads(json.dumps(loop.record()))
        restored=DecisionLoop.from_record(c,saved);c['tick']=5
        self.assertEqual(loop.decide(request(c),False),restored.decide(request(c),False))

    def test_feedback_wrong_ticket_duplicate_nan_exact_mismatch_are_atomic(self):
        c=fixture();loop=DecisionLoop(c);r=loop.decide(request(c))
        memory=copy.deepcopy(loop.memory.entries)
        for ticket,observed in (('foreign',vector(effect(.1))),(r['ticket'],np.full(len(FEATURES),np.nan)),(r['ticket'],vector(effect(.1,cost=.5)))):
            with self.assertRaises(ValueError):loop.observe(ticket,observed)
            self.assertEqual(loop.memory.entries,memory);self.assertIsNotNone(loop.pending)
        loop.observe(r['ticket'],vector(effect(.1)))
        with self.assertRaises(ValueError):loop.observe(r['ticket'],vector(effect(.1)))

    def test_missing_feedback_must_be_abandoned_without_fabricated_learning(self):
        c=fixture();loop=DecisionLoop(c);r=loop.decide(request(c));c['tick']=1
        with self.assertRaises(ValueError):loop.decide(request(c))
        loop.abandon(r['ticket']);self.assertEqual(loop.memory.entries,{})
        self.assertEqual(loop.evidence.entries,{})
        self.assertEqual(loop.record()['last_tick'],0)
        loop.decide(request(c))

    def test_owned_hypotheses_learn_only_after_public_event(self):
        c=fixture(negative=True)
        models=lambda c:{'x-model':{'x':.95,'y':.05},'y-model':{'x':.05,'y':.95}}
        outcomes=lambda c,a,r:effect(.8 if r=='x' else -.8)
        model=CategoricalReader(c['scope'],('x-model','y-model'),models,outcomes)
        loop=DecisionLoop(c,predictor=model)
        for tick in range(5):
            c['tick']=tick;r=loop.decide(request(c,threatened=True))
            self.assertEqual(loop.predictor.tracker.observations,tick)
            with self.assertRaises(ValueError):loop.observe(r['ticket'],vector(effect(.8)),{'revealed_action':'unknown'})
            self.assertEqual(loop.predictor.tracker.observations,tick)
            loop.observe(r['ticket'],vector(effect(.8)),{'revealed_action':'x'})
        self.assertEqual(model.tracker.observations,0)
        self.assertEqual(loop.predictor.tracker.observations,5)
        self.assertTrue(all(r['reading']['nodes']<=16 for _ in (0,)))
        saved=json.loads(json.dumps(loop.record()));restored=DecisionLoop.from_record(c,saved,predictor=model)
        c['tick']=5
        self.assertEqual(loop.decide(request(c,threatened=True)),restored.decide(request(c,threatened=True)))

    def test_categorical_response_validation_uses_reveal_not_unchosen_rewards(self):
        gate=EvidenceGate();uniform={'x':.5,'y':.5};forecast={'x':.9,'y':.1}
        for _ in range(4):gate.categorical('public-response',uniform,forecast,'x')
        self.assertTrue(gate.accepted('public-response'))
        for _ in range(4):gate.categorical('public-response',uniform,forecast,'y')
        self.assertFalse(gate.accepted('public-response'))

    def test_conditional_rule_reader_counts_all_root_response_evaluations(self):
        c=fixture(negative=True);calls=[]
        models=lambda c:{'one':{'x':.5,'y':.5},'two':{'x':.5,'y':.5}}
        def outcome(c,a,r):calls.append((a,r));return effect(.1)
        model=CategoricalReader(c['scope'],('one','two'),models,outcome,known_conditionals=True)
        from collections import deque
        model.recent=deque(['x']*3,maxlen=4)
        self.assertIsNone(model(c,3));self.assertEqual(calls,[])
        prediction=model(c,4);self.assertEqual(prediction.nodes,4);self.assertEqual(len(calls),4)

    def test_response_forecast_monitored_even_when_advantage_skips_root_reading(self):
        c=fixture(negative=True)
        models=lambda c:{'x-model':{'x':.95,'y':.05},'y-model':{'x':.05,'y':.95}}
        reader=CategoricalReader(c['scope'],('x-model','y-model'),models,lambda c,a,r:effect(.8),known_conditionals=True)
        loop=DecisionLoop(c,predictor=reader)
        for tick in range(10):
            c['tick']=tick;r=loop.decide(request(c,maintains_advantage=True))
            self.assertFalse(r['reading']['requested']);self.assertEqual(r['reading']['nodes'],0)
            update=loop.observe(r['ticket'],vector(effect(.8)),{'revealed_action':'x'})
        self.assertEqual(loop.predictor.tracker.observations,10)
        self.assertTrue(update['reading']['scored'])

    def test_prediction_acceptance_alone_cannot_replace_adaptive_reading(self):
        from collections import deque
        c=fixture(negative=True)
        models=lambda c:{'one':{'x':.9,'y':.1},'two':{'x':.1,'y':.9}}
        outcome=lambda c,a,r:effect(.8 if (a=='A')==(r=='x') else -.8)
        reader=CategoricalReader(c['scope'],('one','two'),models,outcome,known_conditionals=True)
        reader.recent=deque(['x']*3,maxlen=4);loop=DecisionLoop(c,predictor=reader)
        key='response:'+digest(reader.key)
        for _ in range(4):loop.evidence.categorical(key,{'x':.5,'y':.5},{'x':.9,'y':.1},'x')
        self.assertTrue(loop.evidence.accepted(key))
        r=loop.decide(request(c,threatened=True),False)
        self.assertEqual(r['reading']['response_model'],'bounded_recent_response')
        self.assertEqual(r['reading']['decision_gate']['state'],'unobserved')
        self.assertEqual(r['decision']['action_id'],'A')
        result=loop.observe(r['ticket'],vector(effect(.8)),{'revealed_action':'x'})
        self.assertEqual(set(loop.memory.entries),{('A','condition')})
        self.assertIn('decision_comparison',result)

    def test_qualified_additional_model_is_used_without_learning_unchosen_rule_trials(self):
        from collections import deque
        c=fixture(negative=True)
        models=lambda c:{'one':{'x':.95,'y':.05},'two':{'x':.05,'y':.95}}
        outcome=lambda c,a,r:effect(.8 if (a=='A')==(r=='x') else -.8)
        reader=CategoricalReader(c['scope'],('one','two'),models,outcome,known_conditionals=True)
        reader.recent=deque(['x']*3,maxlen=4);reader.tracker.logs=[-8.,0.];reader.tracker.observations=20
        loop=DecisionLoop(c,predictor=reader);key='response:'+digest(reader.key)
        for _ in range(4):
            loop.evidence.categorical(key,{'x':.9,'y':.1},{'x':.1,'y':.9},'y')
            loop.evidence._update(key+':decision',.9,.1)
        r=loop.decide(request(c,threatened=True),False)
        self.assertEqual(r['reading']['response_model'],'validated_hypotheses')
        self.assertEqual(r['decision']['action_id'],'B')
        update=loop.observe(r['ticket'],vector(effect(.8)),{'revealed_action':'y'})
        self.assertGreater(update['decision_comparison']['gain'],0)
        self.assertEqual(set(loop.memory.entries),{('B','condition')})
        self.assertEqual(loop.memory.entries[('B','condition')]['count'],1)

    def test_known_conditional_feedback_cannot_teach_contradictory_rule_effects(self):
        c=fixture(negative=True)
        models=lambda c:{'one':{'x':.5,'y':.5},'two':{'x':.5,'y':.5}}
        reader=CategoricalReader(c['scope'],('one','two'),models,lambda c,a,r:effect(.8 if r=='x' else -.8),known_conditionals=True)
        loop=DecisionLoop(c,predictor=reader)
        for tick in range(3):
            c['tick']=tick;r=loop.decide(request(c,threatened=True))
            loop.observe(r['ticket'],vector(effect(.8)),{'revealed_action':'x'})
        c['tick']=3;r=loop.decide(request(c,threatened=True));before=copy.deepcopy(loop.predictor.record())
        with self.assertRaisesRegex(ValueError,'contradicts'):loop.observe(r['ticket'],vector(effect(-.8)),{'revealed_action':'x'})
        self.assertEqual(before,loop.predictor.record());self.assertIsNotNone(loop.pending)
        loop.observe(r['ticket'],vector(effect(.8)),{'revealed_action':'x'})

    def test_empirical_forecast_compared_before_update_and_revoked_when_repeatedly_worse(self):
        c=fixture();c['actions']=c['actions'][:1];loop=DecisionLoop(c);updates=[]
        for tick in range(10):
            c['tick']=tick;r=loop.decide(request(c),False)
            updates.append(loop.observe(r['ticket'],vector(effect(-.8 if tick<4 else .8))))
        self.assertFalse(updates[0]['learning']['scored'])
        self.assertTrue(updates[4]['learning']['scored'])
        self.assertGreater(updates[4]['learning']['candidate_loss'],updates[4]['learning']['prior_loss'])
        self.assertTrue(any(u['learning'].get('after')=='revoked' for u in updates))

    def test_game_conditional_table_matches_all_arena_rule_pairs(self):
        from reflex.loop_experiment import arena_outcome,arena_models
        from reflex.judgment_experiment import Arena,snapshot
        from reflex.laboratory import profiles
        w=Arena();c=snapshot(w,profiles()[0],0,0)
        self.assertTrue(all(abs(sum(row.values())-1)<1e-9 for row in arena_models(c).values()))
        for own in w.names:
            for response in w.moves:self.assertEqual(arena_outcome(c,own,response),w.result(own,response))

    def test_complete_mixed_rule_lifecycle_replays(self):
        from reflex.loop_experiment import group,replay
        for game in ('cyclic_arena','four_player_commons','branching_delivery'):
            rows=group(game,(2,), 'integrated_read',turns=10)
            self.assertEqual(sum(replay(r) for r in rows),40)

    def test_advantage_preservation_avoids_reader_and_budget_exhaustion_keeps_reflex(self):
        c=fixture(negative=True);called=[]
        def reader(c,cap):called.append(cap);return None
        loop=DecisionLoop(c);r=loop.decide(request(c,reader=reader,maintains_advantage=True))
        self.assertEqual(called,[]);loop.abandon(r['ticket']);c['tick']=1
        r=loop.decide(request(c,reader=reader,threatened=True))
        self.assertEqual(called,[16]);self.assertFalse(r['reading']['adopted'])

    def test_experience_exact_overlap_and_false_temporal_training_rejected(self):
        c=fixture();loop=DecisionLoop(c)
        bad=Request(c,request(c).bindings,{a['id']:FEATURES for a in c['actions']})
        with self.assertRaisesRegex(ValueError,'exact'):loop.decide(bad)
        with self.assertRaisesRegex(ValueError,'delayed'):loop.decide(request(c,outcome_target='terminal'))
        future=Request(c,{a['id']:Binding(a['id'],'future',()) for a in c['actions']},request(c).exact)
        r=loop.decide(future);loop.observe(r['ticket'],vector(effect(-.03)))
        self.assertEqual(loop.memory.entries,{})

    def test_public_invalidation_discards_old_method_without_changing_axes(self):
        c=fixture();loop=DecisionLoop(c)
        for tick in range(7):
            c['tick']=tick;r=loop.decide(request(c));loop.observe(r['ticket'],vector(effect(-.8)))
        c['tick']=7;r=loop.decide(request(c,invalidate=(('A','condition'),('B','condition'))))
        self.assertEqual(r['learning']['A']['observations'],0)
        self.assertEqual(r['learning']['A']['gate']['state'],'unobserved')

    def test_checkpoint_replay_and_foreign_corrupt_records_rejected(self):
        c=fixture();loop=DecisionLoop(c)
        for tick in range(7):
            c['tick']=tick;r=loop.decide(request(c));loop.observe(r['ticket'],vector(effect(-.8)))
        record=json.loads(json.dumps(loop.record()));restored=DecisionLoop.from_record(c,record)
        c['tick']=7;self.assertEqual(loop.decide(request(c)),restored.decide(request(c)))
        bad=copy.deepcopy(record);bad['scope']['npc']='other'
        with self.assertRaises(ValueError):DecisionLoop.from_record(c,bad)
        bad=copy.deepcopy(record);bad['evidence']['entries'][0]['gains'][0]=float('nan')
        with self.assertRaises(ValueError):DecisionLoop.from_record(c,bad)

    def test_checkpoint_keeps_policy_and_individual_reading_budget(self):
        c=fixture();loop=DecisionLoop(c,policy=Policy(np.ones(8)*.04),read_control=ReadControl(max_nodes=0))
        r=loop.decide(request(c));loop.observe(r['ticket'],vector(effect(.1)))
        saved=json.loads(json.dumps(loop.record()));restored=DecisionLoop.from_record(c,saved)
        self.assertEqual(restored.read_control.max_nodes,0)
        self.assertTrue(np.array_equal(restored.policy.residual,loop.policy.residual))
        with self.assertRaisesRegex(ValueError,'budget'):DecisionLoop.from_record(c,saved,read_control=ReadControl(max_nodes=16))
        with self.assertRaisesRegex(ValueError,'policy'):DecisionLoop.from_record(c,saved,policy=Policy())

    def test_strategy_and_actions_share_transaction_and_separate_intent(self):
        c=fixture();loop=DecisionLoop(c)
        rc=copy.deepcopy(c);rc['actions']=[action('progress',effect(.8)),action('protect',effect(.2))]
        r=loop.decide(IntentRequest(rc,lambda route:request(copy.deepcopy(c))))
        self.assertEqual(loop.strategy.chosen,'progress');self.assertIsNotNone(r['strategy'])
        self.assertEqual(loop.state['intent_action'],r['decision']['action_id'])
        loop.observe(r['ticket'],vector(effect(.1)));c['tick']=1;rc['tick']=1
        before=loop.record()
        with self.assertRaises(ValueError):loop.decide(IntentRequest(rc,lambda route:request(fixture('wrong'))))
        self.assertEqual(before,loop.record())

    def test_proven_waste_removed_without_erasing_personality_conflicts(self):
        from reflex.judgment_experiment import Commons,snapshot
        from reflex.laboratory import profiles
        rows=[]
        for p in profiles():
            w=Commons();c=snapshot(w,p,0,0)
            for a in c['actions']:a['outcomes']=[w.result(a['id'],('HELP','SECURE','HELP'))]
            c['state']['mode']='principle'
            req=Request(c,{a['id']:Binding(a['id'],'exact',()) for a in c['actions']},w.exact()|{'CLAIM':FEATURES})
            r=DecisionLoop(c).decide(req,False);rows.append((p['id'],r['decision']['action_id']))
            self.assertIn('WASTEFUL_SECURE',r['waste_removed'])
        self.assertIn(('care','HELP'),rows);self.assertIn(('steady','SECURE'),rows);self.assertIn(('ego','CLAIM'),rows)


if __name__=='__main__':unittest.main()
