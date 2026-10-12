"""Recovery fallback must retain the same person's current relationships."""
import copy
from dataclasses import replace
import unittest
import numpy as np
from reflex.core import Policy,NEEDS,FEATURES,compile_batch
from reflex.examples import context,action,effect
from reflex.decision_loop import DecisionLoop,Request,Reading
from reflex.judgment import Binding
from reflex.social import SocialMemory,SocialBinding,SocialEvent
from reflex.social_projection import CoarseSocialMemory
from reflex.social_withdrawal import WithdrawalSocialMemory
from reflex.progress import ProgressWatch,ProgressConfig,PurposeRequest,PurposeFeedback,Activity
from reflex.actor_model import ActorModel
from reflex.model_observer import ModelFeedback


def fixture(npc, memory_type=SocialMemory, watched=True):
    help_=action('help',effect(.2));help_['target']='friend'
    c=context('composition',[help_,action('work',effect(.2))],needs={k:0. for k in NEEDS},
        values={'benevolence':1.},mode='principle')
    c['scope']['npc']=npc;c['tick']=8
    social=memory_type(c)
    for tick in range(8):social.observe(SocialEvent('friend',str(tick),tick,-1.))
    purpose=PurposeRequest(0.,0.,{'help':Activity('attempt','alternative'),'work':Activity('idle','no-progress')})
    watch=None
    if watched:
        watch=ProgressWatch(c['scope'],ProgressConfig(grace=1,proof_margin=.05))
    loop=DecisionLoop(c,policy=Policy(principle_priority='finite'),social=social,progress=watch)
    req=Request(c,{a['id']:Binding(a['id'],'same',()) for a in c['actions']},
        {a['id']:() for a in c['actions']},social_bindings={'help':SocialBinding('friend',1.)},
        purpose=purpose if watched else None)
    if watched:
        prior=copy.deepcopy(c);prior['tick']=7
        decision=loop.decide(replace(req,context=prior),False)
        assert decision['decision']['action_id']=='work'
        loop.abandon(decision['ticket'],purpose_feedback=PurposeFeedback(0.,0.))
    return c,loop,req


class ProgressSocialCompositionTests(unittest.TestCase):
    def test_waste_proof_requires_equal_appraisal_and_keeps_other_guards(self):
        from reflex.judgment import avoid_waste
        c=context('pruning',[action('A',effect(.2)),action('B',effect(.2,cost=.03))])
        b=compile_batch([c]);exact=np.ones(b.legal.shape+(len(FEATURES),),bool)
        for bias,expected in ((None,[[False,True]]),(np.zeros((1,2)),[[False,True]]),
                              (np.array([[.2,.2]]),[[False,True]]),(np.array([[0.,.2]]),[[False,False]])):
            self.assertEqual(avoid_waste(b,exact,appraisal=bias)[1].tolist(),expected)
        for bad in (np.zeros((2,2)),np.array([[0.,np.nan]]),np.array([[0.,2.]])):
            with self.assertRaises(ValueError):avoid_waste(b,exact,appraisal=bad)
        c['actions'][1]['known_failure']=True;failed=compile_batch([c])
        d=Policy().decide(avoid_waste(failed,exact,appraisal=np.array([[0.,1.]]))[0],False,appraisal=np.array([[0.,1.]]))
        self.assertEqual(d.records(failed)[0]['action_id'],'A')

    def test_completion_and_learned_arbitration_use_the_same_social_benefit(self):
        from reflex.root_completion import constrained_completion
        from reflex.paired_guard import ValueProposal,constrain
        from reflex.paired_value import fit
        c=context('pruning',[action('A',effect(.2)),action('B',effect(.2,cost=.03))])
        p=Policy(principle_priority='finite');b=compile_batch([c]);bias={'A':0.,'B':.2}
        incumbent=p.decide(b,False,appraisal=np.array([[0.,.2]])).records(b)[0]
        self.assertEqual(incumbent['action_id'],'B');seen=[]
        def complete(cs,roots,options):seen.append(options);return roots,{}
        constrained_completion([c],[incumbent],[{'A','B'}],np.ones(b.legal.shape+(len(FEATURES),),bool),
            p,complete,appraisal=[bias])
        self.assertEqual(len(seen),1);self.assertIn('B',seen[0][0])
        model=fit([([[s,0],[s,1]],[[.1],[.9]]) for s in (-1.,-.5,.5,1.)],alpha=.01)
        proposal=ValueProposal(model,[.01],{'A':[.5,0],'B':[.5,1]},'public','terminal-success')
        decision,audit=constrain(c,incumbent,proposal,p,{'A','B'},{'A':FEATURES,'B':FEATURES},appraisal=bias)
        self.assertEqual(audit['excluded'],[]);self.assertEqual(decision['action_id'],'B')

    def test_known_extra_cost_is_not_waste_when_relationship_benefit_differs(self):
        for reading in (False,True):
            cheap=action('cheap',effect(.2));cheap['target']='stranger'
            trusted=action('trusted',effect(.2,cost=.03));trusted['target']='friend'
            c=context('different-beneficiaries',[cheap,trusted],needs={k:0. for k in NEEDS},
                values={'benevolence':1.},mode='principle');c['tick']=8
            social=SocialMemory(c)
            for tick in range(8):social.observe(SocialEvent('friend',str(tick),tick,1.))
            loop=DecisionLoop(c,policy=Policy(principle_priority='finite'),social=social)
            reader=(lambda cc,cap:Reading(cc,2,'public',responses={'known':1.},known_conditionals=True,
                                         response_candidate={'known':1.})) if reading else None
            req=Request(c,{a['id']:Binding(a['id'],'same',()) for a in c['actions']},
                {a['id']:FEATURES for a in c['actions']},reader=reader,threatened=reading,
                social_bindings={'trusted':SocialBinding('friend',1.)})
            r=loop.decide(req,False)
            self.assertEqual(r['decision']['action_id'],'trusted')
            self.assertEqual(r['waste_removed'],[])
            if reading:self.assertTrue(r['reading']['requested'])

    def test_unproved_recovery_preserves_relationships_for_every_batched_actor(self):
        for memory_type in (SocialMemory,CoarseSocialMemory,WithdrawalSocialMemory):
            for stochastic in (False,True):
                for decline in (False,True):
                    with self.subTest(memory=memory_type.__name__,stochastic=stochastic,decline=decline):
                        items=[fixture('watched',memory_type),fixture('unrelated',memory_type,False)]
                        expected=[]
                        for c,loop,req in items:
                            ordinary=copy.deepcopy(loop);ordinary.progress=None
                            expected.append(ordinary.decide(replace(req,purpose=None),stochastic)['decision'])
                        self.assertEqual([d['action_id'] for d in expected],['work','work'])
                        seen=[]
                        def planner(cs,ds):seen.extend(ds);return None
                        results=DecisionLoop.decide_batch([(l,r) for _,l,r in items],stochastic,planner if decline else None)
                        self.assertEqual([r['decision'] for r in results],expected)
                        self.assertTrue(results[0]['progress']['unresolved'])
                        self.assertFalse(results[0]['progress']['recovery']['approved'])
                        if decline:self.assertEqual(seen,expected)

    def test_detached_future_and_checkpoint_keep_the_same_fallback(self):
        c,loop,req=fixture('owner');reference=copy.deepcopy(loop);reference.progress=None
        expected=reference.decide(replace(req,purpose=None),False)['decision'];before=loop.record()
        model=ActorModel(loop,root_request=req,request_for=lambda cc:replace(req,context=cc),
                         feedback=lambda *args:ModelFeedback(PurposeFeedback(0.,0.)))
        model.prepare(None,c,prepared=True);self.assertEqual(model.preview(),expected)
        model.commit_choice(expected['action_id']);model.advance(None,None,expected['action_id'],c,req)
        self.assertEqual(loop.record(),before)
        restored=DecisionLoop.from_record(c,before)
        self.assertEqual(restored.decide(req,False)['decision'],expected)


if __name__=='__main__':unittest.main()
