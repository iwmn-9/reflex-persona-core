import copy
import json
import unittest
from dataclasses import replace

from reflex.adaptive_dynamics import EventLearner, AdaptiveKernel, PublicEvent
from reflex.belief_planning import UnsupportedObservation
from reflex.core import digest
from reflex.examples import effect


LEGAL = ('a', 'b', 'c')
LIBRARY = {'left': dict(a=1., b=0., c=0.), 'right': dict(a=0., b=1., c=0.)}


def learner(**kw):
    return EventLearner(owner='npc', model_id='public-rules', names=tuple(LIBRARY), **kw)


class AdaptiveDynamicsTests(unittest.TestCase):
    def test_learns_behavior_absent_from_all_named_hypotheses(self):
        m = learner(); s = m.start(); original = s
        for _ in range(40): s = m.observe(s, (), LEGAL, LIBRARY, 'c')
        self.assertGreater(m.predict(s, (), LEGAL, LIBRARY)['c'], .85)
        self.assertGreater(m.diagnostics(s)['learned_weight'], .95)
        self.assertEqual(original, m.start())
        fixed = learner(learned=False); fs = fixed.start()
        for _ in range(40): fs = fixed.observe(fs, (), LEGAL, LIBRARY, 'c')
        self.assertLess(fixed.predict(fs, (), LEGAL, LIBRARY)['c'], .01)

    def test_discovers_conditional_pattern_without_type_label(self):
        m = learner(factors=('visible-signal', 'irrelevant'))
        s = m.start()
        for i in range(128):
            context = (str(i%2), str((i//2)%2))
            s = m.observe(s, context, LEGAL, LIBRARY, 'c' if i%2 else 'a')
        self.assertGreater(m.predict(s, ('1','0'), LEGAL, LIBRARY)['c'], .85)
        self.assertGreater(m.predict(s, ('0','1'), LEGAL, LIBRARY)['a'], .85)
        # Same base rate cannot explain this: one half of the events are a/c.
        self.assertTrue(any(x['factors'] == ['visible-signal'] and x['weight'] > .1 for x in m.diagnostics(s)['experts']))

    def test_recovers_after_long_success_but_one_outlier_does_not_erase_it(self):
        m = learner(); s = m.start()
        for _ in range(128): s = m.observe(s, (), LEGAL, LIBRARY, 'a')
        outlier = m.observe(s, (), LEGAL, LIBRARY, 'c')
        self.assertGreater(m.predict(outlier, (), LEGAL, LIBRARY)['a'], .6)
        for _ in range(20): s = m.observe(s, (), LEGAL, LIBRARY, 'c')
        self.assertGreater(m.predict(s, (), LEGAL, LIBRARY)['c'], .7)
        self.assertEqual(len(s.history), 128)
        self.assertEqual(s.observations, 148)

    def test_noise_does_not_acquire_confident_spurious_rule(self):
        import random
        m = learner(factors=('coin',)); s = m.start(); r = random.Random(8912)
        for _ in range(256):
            s = m.observe(s, (str(r.randrange(2)),), LEGAL, LIBRARY, r.choice(LEGAL))
        self.assertLess(max(m.predict(s, ('0',), LEGAL, LIBRARY).values()), .65)
        self.assertLess(max(m.predict(s, ('1',), LEGAL, LIBRARY).values()), .65)

    def test_legal_censoring_is_not_evidence_of_preference(self):
        m = EventLearner(owner='npc', model_id='rules'); s = m.start()
        for _ in range(100): s = m.observe(s, (), ('a',), {}, 'a')
        self.assertEqual(m.predict(s, (), LEGAL, {}), dict.fromkeys(LEGAL, 1/3))

    def test_score_is_prequential_not_fitted_to_reveal(self):
        m = learner(); s = m.start()
        p = m.predict(s, (), LEGAL, LIBRARY)
        after = m.observe(s, (), LEGAL, LIBRARY, 'c')
        # Library predicted c with smoothing/3; learned experts with 1/3.
        expected = (1-m.share)*(.5*m.smoothing/3)/p['c'] + m.share*.5
        self.assertAlmostEqual(m.diagnostics(after)['library_weight'], expected)

    def test_checkpoint_ownership_and_atomic_invalid_reveal(self):
        m = learner(factors=('action',)); s = m.start()
        s = m.observe(s, ('inspect',), LEGAL, LIBRARY, 'c')
        saved = json.loads(json.dumps(m.record(s)))
        self.assertEqual(m.restore(saved), s)
        saved['history'][0]['context'][0] = 'different'
        self.assertEqual(s.history[0].context, ('inspect',))
        with self.assertRaises(ValueError): learner(factors=('different',)).restore(m.record(s))
        before = m.record(s)
        with self.assertRaises(UnsupportedObservation): m.observe(s, ('inspect',), LEGAL, LIBRARY, 'd')
        self.assertEqual(m.record(s), before)
        for bad in (replace(s, weights=(float('nan'),)*len(s.weights)), replace(s, observations=0)):
            with self.assertRaises(ValueError): m.record(bad)

    def test_invalid_adapter_inputs_are_rejected(self):
        m = learner(); s = m.start()
        for legal, models in [(('a','a'), LIBRARY), (LEGAL, {}), (LEGAL, {**LIBRARY, 'left':dict(a=True,b=0.,c=0.)})]:
            with self.assertRaises(ValueError): m.predict(s, (), legal, models)

    def test_joint_factors_can_learn_an_interaction_with_no_marginal_cue(self):
        m = learner(factors=('x','y')); s = m.start()
        for i in range(256):
            x,y=i%2,(i//2)%2
            s=m.observe(s,(str(x),str(y)),LEGAL,LIBRARY,'c' if x!=y else 'a')
        self.assertGreater(m.predict(s,('0','1'),LEGAL,LIBRARY)['c'],.8)
        self.assertGreater(m.predict(s,('1','1'),LEGAL,LIBRARY)['a'],.8)

    def test_symbol_renaming_and_factor_order_do_not_add_game_specific_preferences(self):
        m=learner(factors=('x','y')); n=learner(factors=('renamed-y','renamed-x'))
        s=m.start(); t=n.start(); mapping=dict(a='z',b='q',c='u'); legal=tuple(sorted(mapping.values()))
        models={name:{mapping[k]:v for k,v in row.items()} for name,row in LIBRARY.items()}
        for i in range(80):
            c=(str(i%2),str((i//2)%3)); outcome='c' if c[0]=='1' else 'b'
            s=m.observe(s,c,LEGAL,LIBRARY,outcome)
            t=n.observe(t,tuple(reversed(c)),legal,models,mapping[outcome])
        p=m.predict(s,('1','2'),LEGAL,LIBRARY); q=n.predict(t,('2','1'),legal,models)
        for k,v in p.items():self.assertAlmostEqual(v,q[mapping[k]])


def kernel(update=True, events=None):
    m = learner(factors=('action',))
    return AdaptiveKernel(learner=m, model_id='events', update=update,
        events=events or (lambda w,a:[PublicEvent(k, {'last':k,'tick':w['tick']+1}, effect()) for k in LEGAL]),
        features=lambda w,a:(a,), models=lambda w,a:copy.deepcopy(LIBRARY), public_record=lambda w:w)


class AdaptiveKernelTests(unittest.TestCase):
    def test_full_horizon_actual_replanning_matches_the_same_learned_law(self):
        from tools.open_model_games import OpenGame
        from reflex.laboratory import profiles
        from reflex.decision_loop import DecisionLoop
        from reflex.core import Policy, character_key
        from reflex.integrated_trials import request_for
        from reflex.closed_loop import reveal
        from reflex.model_observer import ModelFeedback
        g=OpenGame('resources',profiles()[3],length=2); s=g.start(); c=g.observe(s)
        owner=DecisionLoop(c,policy=Policy(principle_priority='personal',commitments={character_key(c):.5}))
        ctrl=g.controller(2); before=(owner.record(),g.kernel.record(s))
        def evaluate(world,actor,remaining):
            if g.terminal(world):return g.goal(world).value
            own=copy.deepcopy(actor); req=request_for(g.observe(world,own.state),'open-model'); cap={}
            d=DecisionLoop.decide_batch([(own,req)],False,ctrl.planner(own,req,world,remaining=remaining,captured=cap))[0]
            self.assertTrue(cap['audit']['complete'])
            key=d['decision']['action_id']; reveal(own,d['ticket'],ModelFeedback(None))
            expected=sum(t.probability*evaluate(t.state,own,remaining-1) for t in g.kernel(world,key))
            predicted=sum(p*goal['value'] for p,_,_,goal,_ in cap['node']['returns'])
            self.assertAlmostEqual(expected,predicted)
            return expected
        evaluate(s,owner,2)
        self.assertEqual(before,(owner.record(),g.kernel.record(s)))

    def test_novel_legal_events_are_learned_and_forecasts_do_not_mutate_actual_memory(self):
        k = kernel(); s = k.start({'tick':0}); before = digest(k.record(s))
        rows = k(s, 'watch')
        self.assertEqual(digest(k.record(s)), before)
        c = next(t for t in rows if t.state.public['last']=='c')
        self.assertEqual(k.reveal(s, 'watch', c.state.public, c.flow), c.state)
        self.assertEqual(c.state.memory.history[-1].outcome, 'c')
        self.assertEqual(c.state.memory.observations, 1)
        self.assertEqual(s.memory.observations, 0)
        self.assertAlmostEqual(sum(t.probability for t in rows), 1.)

    def test_undistinguishable_hidden_symbols_cannot_leak_through_learning(self):
        k = kernel(events=lambda w,a:[PublicEvent(k, {'tick':1}, effect()) for k in LEGAL])
        with self.assertRaises(ValueError): k(k.start({'tick':0}), 'watch')

    def test_rule_violation_does_not_invent_a_new_world(self):
        k = kernel(); s = k.start({'tick':0}); before = k.record(s)
        with self.assertRaises(UnsupportedObservation): k.reveal(s, 'watch', {'last':'d','tick':1}, effect())
        self.assertEqual(k.record(s), before)

    def test_future_learning_ablation_shares_fact_identity_and_checkpoint(self):
        k = kernel(); s = k.start({'tick':0}); f = kernel(False)
        r = f(s, 'watch')[0]
        self.assertEqual(r.state.memory, s.memory)
        self.assertEqual(k.restore(json.loads(json.dumps(k.record(s))), lambda w:w), s)
        other = AdaptiveKernel(learner=k.learner, events=k.events, features=k.features, models=k.models,
                               public_record=k.public_record, model_id='different')
        with self.assertRaises(ValueError): other.restore(k.record(s), lambda w:w)


if __name__ == '__main__': unittest.main()
