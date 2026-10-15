import copy
import types
from dataclasses import replace
import json
import unittest

from reflex.core import Policy,character_key,digest,FEATURES
from reflex.examples import context,action,effect
from reflex.judgment import Binding,ReadControl
from reflex.decision_loop import DecisionLoop,Request
from reflex.loop_predictor import CategoricalReader,ValidatedSupportedCategoricalReader
from reflex.model_observer import ModelFeedback,resolve_feedback
from reflex.actor_model import ActorModel
from reflex.pressure import NeedPressure
from reflex.progress import Activity,PurposeRequest,PurposeFeedback,ProgressWatch
from reflex.social import SocialEvent,SocialBinding
from reflex.social_withdrawal import WithdrawalSocialMemory
from reflex.planning import vector


def fixture(supported=False,estimated=FEATURES):
    def outcome(c,key,response):
        return effect(.3 if key==response else -.2,values={'benevolence':.1 if key=='A' else 0.})
    c=context('channels',[dict(action(k,*[dict(outcome(None,k,r),p=.5) for r in ('A','B')]),target='peer') for k in ('A','B')])
    models=lambda cc:{'a':{'A':.9,'B':.1},'b':{'A':.1,'B':.9}}
    args=(c['scope'],('a','b'),models,outcome)
    if supported:
        reader=ValidatedSupportedCategoricalReader(*args,classes=('one','two'),opportunity_for=lambda cc:cc['facts']['support'],support_key='test-v1',known_conditionals=True)
    else:reader=CategoricalReader(*args,known_conditionals=True)
    policy=Policy(principle_priority='personal',commitments={character_key(c):.5})
    actor=DecisionLoop(c,policy=policy,predictor=reader,read_control=ReadControl(max_nodes=64),
        pressure=NeedPressure(c['scope']),progress=ProgressWatch(c['scope']),social=WithdrawalSocialMemory(c))
    def request(cc):
        return Request(cc,{k:Binding(k,'same',estimated) for k in ('A','B')},{k:() for k in ('A','B')},threatened=True,
            purpose=PurposeRequest(0.,0.,{k:Activity('uncertain','possible-progress') for k in ('A','B')}),
            social_bindings={k:SocialBinding('peer',1. if k=='A' else -1.) for k in ('A','B')})
    return c,actor,request,outcome


class FeedbackChannelTests(unittest.TestCase):
    def test_event_only_matches_full_channel_when_effects_are_not_learned(self):
        for supported in (False,True):
            c,actor,request,outcome=fixture(supported,estimated=());actor.read_control=ReadControl(max_nodes=0);full=copy.deepcopy(actor)
            for tick in range(96):
                cc=copy.deepcopy(c);cc['tick']=tick;cc['facts']['support']='one' if tick%3 else 'two'
                r=actor.decide(request(cc),False);f=full.decide(request(cc),False)
                self.assertEqual(r,f)
                event={'revealed_action':'A' if tick<24 or tick>=64 else 'B'}
                kwargs=dict(event=event,needs={'growth':.02 if tick%4 else 0.},
                    purpose=PurposeFeedback(0.,0.),social_events=(SocialEvent('peer',str(tick),tick,.1 if tick%2 else -.1),))
                resolve_feedback(actor,r['ticket'],ModelFeedback(observed=None,**kwargs))
                resolve_feedback(full,f['ticket'],ModelFeedback(observed=vector(outcome(cc,f['decision']['action_id'],event['revealed_action'])),**kwargs))
                self.assertEqual(actor.record(),full.record())
                if tick%17==0:
                    saved=json.loads(json.dumps(actor.record()))
                    actor=DecisionLoop.from_record(cc,saved,predictor=actor.predictor)
                    self.assertEqual(digest(saved),digest(actor.record()))
            self.assertFalse(actor.memory.entries)

    def test_censored_effects_do_not_become_zero_or_rule_inferred_experience(self):
        c,a,request,outcome=fixture(estimated=('objective',));r=a.decide(request(c),False)
        fb=ModelFeedback(PurposeFeedback(0.,0.),event={'revealed_action':'B'})
        resolve_feedback(a,r['ticket'],fb)
        self.assertFalse(a.memory.entries)
        self.assertEqual(a.predictor.tracker.observations,1)
        self.assertFalse(any(k.startswith('experience:') for k in a.evidence.entries))
        self.assertEqual(a.memory.last_tick,0)

    def test_missing_everything_does_not_update_predictor(self):
        c,a,request,_=fixture();before=a.predictor.record();r=a.decide(request(c),False)
        resolve_feedback(a,r['ticket'],ModelFeedback(None))
        self.assertEqual(before,a.predictor.record())
        self.assertFalse(a.memory.entries)

    def test_effect_only_learns_effect_without_inventing_response(self):
        for supported in (False,True):
            c,a,request,outcome=fixture(supported);c['facts']['support']='one'
            before=a.predictor.record();r=a.decide(request(c),False)
            update=resolve_feedback(a,r['ticket'],ModelFeedback(None,observed=vector(outcome(c,r['decision']['action_id'],'A'))))
            self.assertEqual(a.predictor.record(),before)
            self.assertEqual(sum(e['count'] for e in a.memory.entries.values()),1)
            self.assertEqual(update['memory']['action'],r['decision']['action_id'])
            self.assertEqual(update['memory']['observations'],1)
            json.dumps(update)

    def test_effect_only_keeps_existing_custom_predictor_protocol(self):
        c,a,request,outcome=fixture();calls=[]
        class LegacyPredictor:
            owner=a.owner
            def __call__(self,cc,cap):return None
            def updated(self,cc,observed,event,ticket):
                calls.append((event,float(observed[0])))
                return self,dict(scored=True)
        a.predictor=LegacyPredictor();r=a.decide(request(c),False)
        resolve_feedback(a,r['ticket'],ModelFeedback(None,observed=vector(effect(.2))))
        self.assertEqual(calls,[(None,.2)])

    def test_effect_only_with_active_reader_does_not_score_a_guessed_response(self):
        c,a,request,outcome=fixture()
        for tick in range(5):
            c['tick']=tick;r=a.decide(request(c),False)
            resolve_feedback(a,r['ticket'],ModelFeedback(None,event={'revealed_action':'A'}))
        c['tick']=5;r=a.decide(request(c),False)
        self.assertIsNotNone(a.pending['read'])
        before=a.predictor.record();evidence=copy.deepcopy(a.evidence.entries)
        resolve_feedback(a,r['ticket'],ModelFeedback(None,observed=vector(outcome(c,r['decision']['action_id'],'B'))))
        self.assertEqual(before,a.predictor.record())
        for k,v in evidence.items():self.assertEqual(v,a.evidence.entries[k])

    def test_rejected_feedback_cannot_mutate_a_custom_predictor_in_place(self):
        c,a,request,_=fixture()
        def mutating(this,cc,observed,event,ticket):
            this.recent.append('B')
            return this,{}
        a.predictor.updated=types.MethodType(mutating,a.predictor)
        r=a.decide(request(c),False);before=a.predictor.record()
        with self.assertRaises(ValueError):
            resolve_feedback(a,r['ticket'],ModelFeedback(None,observed=vector(effect()),event={'revealed_action':'A'},
                social_events=(SocialEvent('peer','future',1,.2),)))
        self.assertEqual(before,a.predictor.record());self.assertIsNotNone(a.pending)

    def test_long_mixed_stream_preserves_relationships_and_independent_evidence(self):
        c,actual,request,outcome=fixture();observed_events=0;seen_social=0
        initial=(copy.deepcopy(actual.personality),copy.deepcopy(actual.values))
        def feedback(before,after,key,cc):
            response='A' if before<64 or before>=160 else 'B'
            return ModelFeedback(PurposeFeedback(0.,0.),
                observed=vector(outcome(cc,key,response)) if before%4 in (0,1) else None,
                event={'revealed_action':response} if before%4 in (0,2) else None,
                needs={'growth':.1 if before%5 else 0.} if before%7 else None,
                social_events=(SocialEvent('peer',str(before),before,.2 if before<100 else -.2),) if before%3==0 else ())
        model=ActorModel(actual,root_request=request(c),request_for=request,feedback=feedback)
        for tick in range(256):
            cc=copy.deepcopy(c);cc['tick']=tick
            old_memory=actual.memory.record()['entries']
            prepared,req,_,_=model.prepare(tick,cc,prepared=tick==0)
            preview=model.preview();r=actual.decide(request(cc),False)
            self.assertEqual(preview,r['decision'])
            fb=feedback(tick,tick+1,preview['action_id'],cc)
            model.commit_choice(preview['action_id']);model.advance(tick,tick+1,preview['action_id'],prepared,req)
            resolve_feedback(actual,r['ticket'],fb)
            observed_events+=int(fb.event is not None);seen_social+=len(fb.social_events)
            self.assertEqual(actual.predictor.tracker.observations,observed_events)
            if fb.observed is None:self.assertEqual(old_memory,actual.memory.record()['entries'])
            self.assertEqual(digest(model.actor.record()),digest(actual.record()))
            self.assertEqual((actual.personality,actual.values),initial)
            self.assertLessEqual(len(actual.evidence.entries),actual.evidence.capacity)
            self.assertLessEqual(len(actual.memory.entries),actual.memory.capacity)
            if tick%31==0:
                saved=json.loads(json.dumps(actual.record()))
                actual=DecisionLoop.from_record(cc,saved,predictor=actual.predictor)
        self.assertEqual(observed_events,128);self.assertEqual(seen_social,86)

    def test_event_and_other_channels_are_transactional_and_cannot_repeat(self):
        c,a,request,_=fixture();r=a.decide(request(c),False)
        pending=copy.deepcopy(a.pending);predictor=a.predictor.record();evidence=copy.deepcopy(a.evidence.entries)
        for fb in (ModelFeedback(None,event={'revealed_action':'unknown'}),
                   ModelFeedback(None,event={'revealed_action':'B'},social_events=(SocialEvent('peer','future',1,.1),))):
            with self.assertRaises(ValueError):resolve_feedback(a,r['ticket'],fb)
            self.assertEqual(a.pending,pending);self.assertEqual(a.predictor.record(),predictor);self.assertEqual(a.evidence.entries,evidence)
        resolve_feedback(a,r['ticket'],ModelFeedback(None,event={'revealed_action':'B'}))
        with self.assertRaises(ValueError):resolve_feedback(a,r['ticket'],ModelFeedback(None,event={'revealed_action':'B'}))

    def test_detached_model_retains_all_channels_and_leaves_owner_untouched(self):
        c,a,request,outcome=fixture(estimated=('objective',));actual=copy.deepcopy(a);saved=a.record()
        def feedback(before,after,key,cc):
            response='A' if before<4 else 'B'
            return ModelFeedback(PurposeFeedback(0.,0.),observed=vector(outcome(cc,key,response)) if before%3==0 else None,
                event={'revealed_action':response},needs={'growth':.1 if before%2 else 0.},
                social_events=(SocialEvent('peer',str(before),before,.1),))
        model=ActorModel(a,root_request=request(c),request_for=request,feedback=feedback)
        for tick in range(16):
            cc=copy.deepcopy(c);cc['tick']=tick
            prepared,req,_,_=model.prepare(tick,cc,prepared=tick==0)
            d=model.preview();r=actual.decide(request(cc),False);self.assertEqual(d,r['decision'])
            model.commit_choice(d['action_id']);model.advance(tick,tick+1,d['action_id'],prepared,req)
            resolve_feedback(actual,r['ticket'],feedback(tick,tick+1,d['action_id'],cc))
            self.assertEqual(model.actor.record(),actual.record())
        self.assertEqual(a.record(),saved)
        self.assertEqual(actual.predictor.tracker.observations,16)
        self.assertEqual(sum(e['count'] for e in actual.memory.entries.values()),6)

    def test_event_only_requires_explicit_consumer(self):
        c,a,request,_=fixture();a.predictor=None
        r=a.decide(request(c),False)
        with self.assertRaises(ValueError):a.abandon(r['ticket'],event={'revealed_action':'A'})
        self.assertIsNotNone(a.pending)


if __name__=='__main__':unittest.main()
