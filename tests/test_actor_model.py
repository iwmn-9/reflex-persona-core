import copy
from dataclasses import replace
import unittest
import numpy as np
from reflex.core import Policy,compile_batch,character_key,digest,NEEDS
from reflex.examples import context,action,effect
from reflex.decision_loop import DecisionLoop,Request
from reflex.judgment import Binding
from reflex.social import SocialMemory,SocialBinding,SocialEvent
from reflex.social_withdrawal import WithdrawalSocialMemory
from reflex.loop_predictor import CategoricalReader,SupportedCategoricalReader,ValidatedSupportedCategoricalReader,GuardedSupportedCategoricalReader
from reflex.valuation import ReturnFrame
from reflex.deliberation import JointForecast,select
from reflex.actor_model import ActorModel
from reflex.flow_rollout import rollout
from reflex.model_observer import ModelFeedback
from reflex.planning import vector
from reflex.owner_continuation import OwnerBranch
from reflex.pressure import NeedPressure
from reflex.progress import ProgressWatch,ProgressConfig,PurposeRequest,PurposeFeedback,Activity


def scene(name='actor'):
    help_=action('help',effect(.15,values={'benevolence':.2}));help_['target']='friend'
    return context(name,[help_,action('work',effect(.20))],values={'benevolence':1.},needs={k:0. for k in NEEDS},mode='principle')

def request(c):
    return Request(c,{a['id']:Binding(a['id'],'same',()) for a in c['actions']},
        {a['id']:() for a in c['actions']},social_bindings={'help':SocialBinding('friend',1.)})

class ActorModelTests(unittest.TestCase):
    def test_counterfactual_blocked_root_does_not_disable_real_progress_control(self):
        c=scene();c['tick']=1
        purpose=PurposeRequest(0.,0.,{'help':Activity('idle','no-progress'),'work':Activity('attempt','real-option')})
        watch=ProgressWatch(c['scope'],ProgressConfig(grace=1));watch.observe(0,'help',purpose,PurposeFeedback(0.,0.))
        actor=DecisionLoop(c,policy=Policy(principle_priority='finite'),progress=watch)
        def req(cc):return Request(cc,{a['id']:Binding(a['id'],'same',()) for a in cc['actions']},{a['id']:() for a in cc['actions']},purpose=purpose)
        saved=actor.record();raw=req(c)
        model=ActorModel(actor,root_request=raw,request_for=req,feedback=lambda *args:ModelFeedback(PurposeFeedback(0.,0.)))
        def observe(t,state):
            cc=copy.deepcopy(c);cc.update(tick=t,state=state);return cc
        _,audit=rollout(c,1,observe=observe,advance=lambda t,k,s:(t+1,effect()),terminal=lambda t:t==3,
            horizon=2,observer=model,policy=actor.policy)
        self.assertEqual(audit['help'][0]['actions'],['help','work'])
        self.assertEqual(audit['work'][0]['actions'],['work','work'])
        self.assertEqual(actor.record(),saved)
        self.assertEqual(copy.deepcopy(actor).decide(raw,False)['decision']['action_id'],'work')

    def test_counterfactual_choice_cannot_force_illegal_or_known_failed_actions(self):
        for field,value in (('legal',False),('known_failure',True)):
            c=scene();c['actions'][0][field]=value;actor=DecisionLoop(c)
            raw=replace(request(c),social_bindings=None)
            model=ActorModel(actor,root_request=raw,request_for=lambda cc:replace(request(cc),social_bindings=None),feedback=lambda *a:ModelFeedback(None))
            model.prepare(None,c,prepared=True);model.preview();before=model.record()
            with self.assertRaisesRegex(ValueError,'viable'):model.commit_choice('help')
            self.assertEqual(model.record(),before)

    def test_explicit_retention_without_psychological_bias_is_also_checkpointed(self):
        c=scene();models=lambda cc:{'old':{'A':.9,'B':.1},'new':{'A':.1,'B':.9}}
        reader=CategoricalReader(c['scope'],('old','new'),models,lambda *a:effect(),belief_options={'retention':.7})
        record=reader.record();self.assertEqual(record['version'],'loop-predictor-v2')
        self.assertEqual(reader.restored(record).record(),record)
        changed=CategoricalReader(c['scope'],('old','new'),models,lambda *a:effect(),belief_options={'retention':.9})
        with self.assertRaises(ValueError):changed.restored(record)

    def test_support_readers_retain_disposition_and_independent_banks(self):
        c=scene();models=lambda cc:{'old':{'A':.9,'B':.1},'new':{'A':.1,'B':.9}}
        for cls in (SupportedCategoricalReader,ValidatedSupportedCategoricalReader,GuardedSupportedCategoricalReader):
            reader=cls(c['scope'],('old','new'),models,lambda *a:effect(),classes=('calm','contested'),
                opportunity_for=lambda cc:'calm',support_key='public-v1',belief_options=dict(inertia=.7,prior={'old':.8,'new':.2},assumption_trust=.6))
            updated,_=reader.updated(c,vector(effect()),{'revealed_action':'B'},'seen')
            self.assertEqual(updated.support.select('calm').tracker.observations,1)
            self.assertEqual(updated.support.select('contested').tracker.observations,0)
            actor=DecisionLoop(c,predictor=updated);record=actor.record()
            self.assertEqual(DecisionLoop.from_record(c,record,predictor=updated).record(),record)

    def test_composed_real_and_modeled_lifecycles_match_after_feedback_and_response_changes(self):
        c=scene();p=Policy(principle_priority='personal',commitments={character_key(c):.5})
        models=lambda cc:{'old':{'A':.9,'B':.1},'new':{'A':.1,'B':.9}}
        reader=CategoricalReader(c['scope'],('old','new'),models,lambda *a:effect(.1),
            belief_options=dict(inertia=.7,prior={'old':.8,'new':.2},assumption_trust=.6))
        original=DecisionLoop(c,policy=p,predictor=reader,pressure=NeedPressure(c['scope']),social=WithdrawalSocialMemory(c))
        def req(cc):
            return replace(request(cc),bindings={a['id']:Binding(a['id'],'public-condition',('objective',)) for a in cc['actions']})
        def feedback(before,after,key,cc):
            tick=cc['tick']
            return ModelFeedback(None,observed=vector(effect(.3 if tick<4 else -.2)),
                event={'revealed_action':'A' if tick<4 else 'B'},needs={'growth':0. if tick<4 else .2},
                social_events=(SocialEvent('friend',str(tick),tick,.4 if tick<4 else -.6),))
        actual=copy.deepcopy(original);saved=original.record()
        model=ActorModel(original,root_request=req(c),request_for=req,feedback=feedback)
        for tick in range(8):
            cc=copy.deepcopy(c);cc['tick']=tick
            if actual.state is not None:cc['state']=copy.deepcopy(actual.state)
            public,raw,_,_=model.prepare(tick,cc,prepared=tick==0)
            preview=model.preview();real=actual.decide(req(cc),False)
            self.assertEqual(preview,real['decision'])
            model.commit_choice(preview['action_id']);fb=feedback(tick,tick+1,preview['action_id'],cc)
            actual.observe(real['ticket'],fb.observed,event=fb.event,need_progress=fb.needs,social_events=fb.social_events)
            model.advance(tick,tick+1,preview['action_id'],public,raw)
            self.assertEqual(model.actor.record(),actual.record())
        self.assertEqual(original.record(),saved)
        self.assertEqual(model.actor.predictor.tracker.observations,8)
        self.assertGreater(len(model.actor.memory.entries),0)

    def test_relationship_checkpoint_keeps_projection_and_reflex_choice(self):
        c=scene();m=WithdrawalSocialMemory(c)
        for i in range(8):m.observe(SocialEvent('friend',f'bad-{i}',i,-1.))
        c['tick']=8;loop=DecisionLoop(c,policy=Policy(principle_priority='finite'),social=m)
        saved=loop.record();restored=DecisionLoop.from_record(c,saved)
        self.assertIsInstance(restored.social,WithdrawalSocialMemory);self.assertEqual(restored.record(),saved)
        a=loop.decide(request(c),False);b=restored.decide(request(c),False)
        self.assertEqual(a,b);self.assertEqual(a['decision']['action_id'],'work')

    def test_forecast_root_appraisal_changes_social_choice_without_mutating_owner(self):
        c=scene();c['actions'][0]['outcomes'][0]=effect(.2);m=SocialMemory(c)
        for i in range(8):m.observe(SocialEvent('friend',str(i),i,-1.))
        c['tick']=8;p=Policy(principle_priority='finite');loop=DecisionLoop(c,policy=p,social=m)
        def planner(cs,ds):
            fc=copy.deepcopy(cs[0]);fc['actions']=[copy.deepcopy(a) for a in cs[0]['actions']]
            return JointForecast((fc,),{a['id']:(a['id'],) for a in fc['actions']},{'help':.2,'work':.2},1)
        selected=DecisionLoop.decide_batch([(loop,request(c))],False,planner)[0]
        self.assertEqual(selected['decision']['action_id'],'work');self.assertIn('social',selected)
        saved=loop.social.record();loop.abandon(selected['ticket']);self.assertEqual(loop.social.record(),saved)

    def test_social_validation_is_transactional_with_immediate_learning(self):
        c=scene();loop=DecisionLoop(c,social=SocialMemory(c));r=loop.decide(request(c),False)
        old=digest(loop.social.record());old_memory=copy.deepcopy(loop.memory.entries)
        with self.assertRaises(ValueError):loop.observe(r['ticket'],vector(effect()),social_events=(SocialEvent('friend','future',1,-1.),))
        self.assertIsNotNone(loop.pending);self.assertEqual(digest(loop.social.record()),old);self.assertEqual(loop.memory.entries,old_memory)

    def test_detached_actor_learns_selected_model_results_and_relationships_only_inside_branch(self):
        c=scene();c['state'].update(primary_need=None,mode_urgency=0.)
        p=Policy(principle_priority='personal',commitments={character_key(c):1.})
        actor=DecisionLoop(c,policy=p,social=WithdrawalSocialMemory(c))
        def req(cc):
            return replace(request(cc),bindings={a['id']:Binding(a['id'],'same',('objective',)) for a in cc['actions']})
        raw=req(c);saved=actor.record()
        def observe(t,state):
            cc=copy.deepcopy(c);cc['tick']=t;cc['state']=state;return cc
        def advance(t,key,seed):return t+1,effect(.1,values={'benevolence':.2} if key=='help' else {})
        def feedback(before,after,key,cc):
            return ModelFeedback(None,observed=vector(effect(.1)),social_events=(SocialEvent('friend',str(before),before,-1.),))
        model=ActorModel(actor,root_request=raw,request_for=req,feedback=feedback)
        paths,audit=rollout(c,0,observe=observe,advance=advance,terminal=lambda t:t==3,horizon=3,
            roots=('help',),seeds=(1,2),observer=model,policy=p)
        self.assertEqual(actor.record(),saved);self.assertEqual(model.actor.record(),saved)
        for trace in audit['help']:
            record=trace['modeled_observations'][-1]['checkpoint']['actor']
            self.assertEqual(record['social']['state']['last_tick'],2)
            self.assertGreater(len(record['memory']['entries']),0)
            self.assertEqual(trace['actions'],['help']*3)

    def test_preview_matches_real_immediate_actor_from_identical_state(self):
        c=scene();actor=DecisionLoop(c,social=WithdrawalSocialMemory(c));raw=request(c)
        model=ActorModel(actor,root_request=raw,request_for=request,feedback=lambda *a:ModelFeedback(None))
        cc,_,_,_=model.prepare(None,c,prepared=True);preview=model.preview()
        actual=copy.deepcopy(actor).decide(raw,False)['decision']
        self.assertEqual(preview,actual)
        committed=model.commit_choice(preview['action_id']);self.assertEqual(committed,actual)
        model.advance(None,None,committed['action_id'],cc,raw)
        self.assertEqual(actor.last_tick,-1)

    def test_hypothesis_disposition_checkpoint_through_real_actor(self):
        c=scene();models=lambda cc:{'old':{'A':.9,'B':.1},'new':{'A':.1,'B':.9}}
        reader=CategoricalReader(c['scope'],('old','new'),models,lambda *a:effect(),
            belief_options=dict(inertia=1.,prior={'old':.95,'new':.05},assumption_trust=.9))
        reader.tracker.observe(models(c),'B','seen')
        actor=DecisionLoop(c,predictor=reader);record=actor.record()
        self.assertEqual(record['predictor']['version'],'loop-predictor-v2')
        restored=DecisionLoop.from_record(c,record,predictor=reader);self.assertEqual(restored.record(),record)
        changed=CategoricalReader(c['scope'],('old','new'),models,lambda *a:effect())
        with self.assertRaises(ValueError):DecisionLoop.from_record(c,record,predictor=changed)

    def test_old_team_owner_branch_can_use_the_same_explicit_commitment(self):
        c=scene();c['actions'][0]['outcomes'][0]['objective']=-1.;c['actions'][1]['outcomes'][0]['objective']=1.
        p=Policy(principle_priority='personal',commitments={character_key(c):1.});d=p.choose(c,False)
        branch=OwnerBranch((0,),(c,),(d,),('help',),policy=p)
        def factory(w,who,profile,seed,state):
            cc=copy.deepcopy(c);cc.update(tick=1,state=state);return cc
        self.assertEqual(branch.choose((0,),None,factory)[0][0],'help')

    def test_wrong_policy_is_rejected_before_any_branch_is_simulated(self):
        c=scene();p=Policy(principle_priority='finite');actor=DecisionLoop(c,policy=p)
        raw=replace(request(c),social_bindings=None)
        model=ActorModel(actor,root_request=raw,request_for=lambda cc:replace(request(cc),social_bindings=None),feedback=lambda *a:ModelFeedback(None))
        with self.assertRaisesRegex(ValueError,'policy'):model.validate_policy(Policy())
