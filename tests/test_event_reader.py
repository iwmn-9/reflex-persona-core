import copy
import json
import unittest

from reflex.core import FEATURES
from reflex.event_reader import EventReader
from reflex.decision_loop import DecisionLoop, Request
from reflex.judgment import Binding, ReadControl
from reflex.examples import context, action, effect
from reflex.model_observer import ModelFeedback, resolve_feedback
from reflex.actor_model import ActorModel
from reflex.planning import vector


def fixture():
    legal = ('a','b','c')
    def outcome(c,k,z): return effect(.7 if k == z else -.3)
    c = context('event-recovery', [action(k,*[dict(outcome(None,k,z),p=1/3) for z in legal]) for k in legal])
    c['facts']['cue']='x'
    reader = EventReader(c['scope'], ('left','right'),
        lambda cc: {'left':dict(a=.9,b=.1,c=0.),'right':dict(a=.1,b=.9,c=0.)},
        outcome, legal_for=lambda cc:legal, model_id='test-rules-v1', known_conditionals=True,
        factors=('cue',), features_for=lambda cc:(cc['facts']['cue'],))
    actor = DecisionLoop(c,predictor=reader,read_control=ReadControl(max_nodes=64))
    def request(cc): return Request(cc,{k:Binding(k,'same',FEATURES) for k in legal},
                                    {k:() for k in legal},threatened=True)
    return c,actor,request,outcome


class EventReaderTests(unittest.TestCase):
    def test_unrepresented_behavior_reaches_actual_choice_without_repricing_personality(self):
        c,a,req,_=fixture(); identity=(copy.deepcopy(a.personality),copy.deepcopy(a.values))
        for tick in range(48):
            c['tick']=tick;r=a.decide(req(c),False)
            resolve_feedback(a,r['ticket'],ModelFeedback(None,event={'revealed_action':'c'}))
        c['tick']=48
        self.assertGreater(a.predictor.predict(c)['c'],.85)
        r=a.decide(req(c),False)
        self.assertEqual(r['decision']['action_id'],'c')
        self.assertEqual(r['reading']['response_model'],'single_response_forecast')
        self.assertEqual((a.personality,a.values),identity)
        self.assertFalse(a.memory.entries,'inferred effects must not become factual samples')

    def test_mixed_channels_and_branch_updates_preserve_the_same_owner(self):
        c,a,req,outcome=fixture()
        def feedback(before,after,k,cc):
            z='a' if before<16 else 'c'
            return ModelFeedback(None,event={'revealed_action':z} if before%4 in (0,2) else None,
                observed=vector(outcome(cc,k,z)) if before%4 in (0,1) else None)
        model=ActorModel(a,root_request=req(c),request_for=req,feedback=feedback)
        for tick in range(40):
            c['tick']=tick;before=copy.deepcopy(a.predictor.record())
            prepared,request,_,_=model.prepare(tick,c,prepared=tick==0)
            preview=model.preview();r=a.decide(req(c),False)
            self.assertEqual(preview,r['decision'])
            model.commit_choice(preview['action_id']);model.advance(tick,tick+1,preview['action_id'],prepared,request)
            resolve_feedback(a,r['ticket'],feedback(tick,tick+1,preview['action_id'],c))
            self.assertEqual(model.actor.record(),a.record())
            if tick%4 in (1,3):self.assertEqual(before,a.predictor.record())
            saved=json.loads(json.dumps(a.record()))
            restored=DecisionLoop.from_record(c,saved,predictor=a.predictor)
            self.assertEqual(saved,restored.record())
        self.assertEqual(a.predictor.observations,20)

    def test_invalid_event_does_not_partially_commit_owner(self):
        c,a,req,_=fixture();r=a.decide(req(c),False)
        before=copy.deepcopy(a.predictor.record())
        with self.assertRaises(ValueError):
            resolve_feedback(a,r['ticket'],ModelFeedback(None,event={'revealed_action':'unknown'}))
        self.assertEqual(before,a.predictor.record());self.assertIsNotNone(a.pending)
        resolve_feedback(a,r['ticket'],ModelFeedback(None,event={'revealed_action':'c'}))
        self.assertEqual(a.predictor.observations,1)

    def test_checkpoint_rejects_new_semantics_owner_or_clock(self):
        c,a,_,_=fixture();r=a.predictor
        r,_=r.updated_event(c,{'revealed_action':'c'},'first')
        saved=json.loads(json.dumps(r.record()));self.assertEqual(r.restored(saved).record(),saved)
        for field,value in (('owner','other'),('key','other'),('last_tick',-1)):
            bad=copy.deepcopy(saved);bad[field]=value
            with self.assertRaises(ValueError):r.restored(bad)
        bad=copy.deepcopy(saved);bad['memory']['identity']='different-rules'
        with self.assertRaises(ValueError):r.restored(bad)
        with self.assertRaises(ValueError):r.updated_event(c,{'revealed_action':'c'},'duplicate')

    def test_cap_and_forecast_queries_never_change_memory(self):
        c,a,_,_=fixture();r=a.predictor;before=r.record()
        self.assertIsNone(r(c,8));self.assertEqual(r(c,9).nodes,9)
        self.assertEqual(r.predict(c),r(c,9).responses);self.assertEqual(r.record(),before)
        alien=copy.deepcopy(c);alien['scope']['npc']='different'
        with self.assertRaises(ValueError):r.predict(alien)

    def test_new_legal_set_does_not_infer_preference_from_forced_events(self):
        c,a,_,_=fixture();r=a.predictor
        r.legal_for=lambda cc:tuple(cc['facts']['legal'].split(','))
        r.models_for=lambda cc:{name:dict.fromkeys(r.legal_for(cc),1/len(r.legal_for(cc))) for name in ('left','right')}
        c['facts']['legal']='a'
        for tick in range(64):
            c['tick']=tick;r,_=r.updated_event(c,{'revealed_action':'a'},str(tick))
        c['facts']['legal']='a,b,c'
        for p in r.predict(c).values():self.assertAlmostEqual(p,1/3)

    def test_wide_event_alphabet_can_learn_but_declines_an_incomplete_policy_read(self):
        c,a,_,_=fixture();r=a.predictor
        legal=tuple(str(i) for i in range(9))
        r.legal_for=lambda cc:legal
        r.models_for=lambda cc:{name:dict.fromkeys(legal,1/9) for name in ('left','right')}
        self.assertEqual(len(r.predict(c)),9)
        self.assertIsNone(r(c,1000))
        changed,_=r.updated_event(c,{'revealed_action':'8'},'observed')
        self.assertEqual(changed.observations,1)


if __name__=='__main__':unittest.main()
