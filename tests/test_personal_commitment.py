import copy
from dataclasses import replace
import unittest
import numpy as np
from reflex.core import Policy,compile_batch,character_key as owner_key,digest
from reflex.examples import context,action,effect
from reflex.decision_loop import DecisionLoop,Request
from reflex.runtime import Population
from reflex.scaling import TiledPolicy
from reflex.contingent_search import search_forecast
from reflex.purpose_plan import Goal
from reflex.deliberation import select
from reflex.opponent_beliefs import HypothesisTracker
from reflex.progress import Activity,PurposeRequest,PurposeFeedback,ProgressWatch,ProgressConfig
from reflex.judgment import Binding

def scene(name='pledge'):
    return context(name,[action('escape',effect(1.)),action('protect',effect(-1.,values={'benevolence':.3},cost=.2))],
        values={'benevolence':1.},mode='principle',needs={'physiology':0.,'safety':0.})

def policy(c,r):return Policy(principle_priority='personal',commitments={owner_key(c):r})

class PersonalCommitmentTests(unittest.TestCase):
    def test_same_values_different_commitment_and_permutation_batch(self):
        a=scene('flexible');b=scene('unyielding');saved=[digest(c) for c in (a,b)]
        p=Policy(principle_priority='personal',commitments={owner_key(a):0.,owner_key(b):1.})
        for cs,want in (([a,b],['escape','protect']),([b,a],['protect','escape'])):
            result=p.decide(compile_batch(cs),False).records(compile_batch(cs))
            self.assertEqual([r['action_id'] for r in result],want)
        self.assertEqual([digest(c) for c in (a,b)],saved)

    def test_zero_endpoint_matches_finite_and_no_value_has_no_veto(self):
        for t in range(8):
            c=scene();c['tick']=t
            a=policy(c,0.).decide(compile_batch([c]),False);b=Policy(principle_priority='finite').decide(compile_batch([c]),False)
            np.testing.assert_array_equal(a.action,b.action);np.testing.assert_array_equal(a.scores,b.scores)
        c['values']={k:0. for k in c['values']}
        self.assertEqual(policy(c,1.).choose(c,False)['action_id'],'escape')

    def test_urgent_need_does_not_break_an_explicit_unyielding_pledge(self):
        c=scene();c['needs']['safety']['deficit']=1.;c['state'].update(mode='need',primary_need='safety',mode_urgency=1.)
        self.assertEqual(policy(c,1.).choose(c,False)['action_id'],'protect')
        self.assertEqual(policy(c,0.).choose(c,False)['action_id'],'escape')

    def test_commitment_never_relaxes_legality_or_understood_failure(self):
        for field in ('legal','known_failure'):
            c=scene();c['actions'][1][field]=field=='known_failure'
            self.assertEqual(policy(c,1.).choose(c,False)['action_id'],'escape')

    def test_pledge_preserved_but_broken_method_replaced(self):
        c=scene();c['actions']=[action('old',effect(-.5,values={'benevolence':.3})),action('new',effect(.4,values={'benevolence':.3}))]
        c['state'].update(intent_action='old',age=9)
        self.assertEqual(policy(c,1.).choose(c,False)['action_id'],'new')

    def test_registry_cannot_be_mutated_and_wrong_owner_rejected(self):
        c=scene();r={owner_key(c):1.};p=Policy(principle_priority='personal',commitments=r)
        r[owner_key(c)]=0.;p.commitments[owner_key(c)]=0.
        self.assertEqual(p.choose(c,False)['action_id'],'protect')
        cc=copy.deepcopy(c);cc['values']['benevolence']=.5
        with self.assertRaisesRegex(ValueError,'owner'):p.choose(cc,False)
        for v in (True,float('nan'),-.1,1.1):
            with self.assertRaises(ValueError):policy(c,v)

    def test_checkpoint_and_shared_batch_contract(self):
        c=scene();p=policy(c,1.);loop=DecisionLoop(c,policy=p)
        restored=DecisionLoop.from_record(c,loop.record());self.assertEqual(restored.record(),loop.record())
        with self.assertRaisesRegex(ValueError,'policy'):DecisionLoop.from_record(c,loop.record(),policy=policy(c,.2))
        cc=scene('second');q=policy(cc,0.);other=DecisionLoop(cc,policy=q)
        def request(cs):return Request(cs,{a['id']:Binding(a['id'],'same',()) for a in cs['actions']},
            {a['id']:() for a in cs['actions']})
        results=DecisionLoop.decide_batch([(loop,request(c)),(other,request(cc))],False)
        self.assertEqual([r['decision']['action_id'] for r in results],['protect','escape'])
        self.assertEqual(len(loop.policy.commitments),1)

    def test_disposition_survives_goal_change_and_single_actor_checkpoint_is_small(self):
        c=scene();cc=scene('second');p=Policy(principle_priority='personal',commitments={owner_key(c):1.,owner_key(cc):0.})
        loop=DecisionLoop(c,policy=p);record=loop.record();self.assertEqual(len(record['policy_commitments']),1)
        altered=copy.deepcopy(c);altered['objective']='different goal'
        self.assertEqual(p.choose(altered,False)['action_id'],'protect')
        restored=DecisionLoop.from_record(altered,record,policy=p);self.assertEqual(restored.record(),record)

    def test_intermediate_commitment_changes_multioption_tradeoff_without_action_rules(self):
        c=scene();c['state'].update(primary_need='belonging',mode_urgency=.1)
        c['actions']=[action(f'option-{i:02d}',effect(1-2*i/10,values={'benevolence':i/10},cost=i/10)) for i in range(11)]
        selected=[int(policy(c,r).choose(c,False)['action_id'].split('-')[1]) for r in (0.,.25,.5,.75,1.)]
        self.assertEqual(selected,sorted(selected));self.assertGreater(len(set(selected)),2)

    def test_population_tiling_preserves_individual_commitment(self):
        cs=[scene(str(i)) for i in range(8)];registry={owner_key(c):float(i%2) for i,c in enumerate(cs)}
        a=Population(cs,Policy(principle_priority='personal',commitments=registry))
        b=Population(cs,TiledPolicy(3,principle_priority='personal',commitments=registry))
        for _ in range(3):
            x=a.step(False);y=b.step(False);np.testing.assert_array_equal(x.action,y.action);np.testing.assert_array_equal(x.scores,y.scores)

    def test_future_and_root_use_same_commitment_without_mutating_state(self):
        c=scene();p=policy(c,1.);saved=digest(c)
        def observe(t,state):
            cc=copy.deepcopy(c);cc['tick']=t;cc['state']=state;return cc
        def advance(t,key,seed):return t+1,copy.deepcopy(next(a for a in c['actions'] if a['id']==key)['outcomes'][0])
        f=search_forecast(c,0,observe=observe,advance=advance,terminal=lambda t:t>=2,
            assess=lambda t:Goal('mission','failure',-1.).record(),horizon=2,seeds=(1,),width=1,revisions=0,
            patch_budget=1,target='mission',policy=p,max_regret=2.)
        self.assertTrue(all(t['actions'][1]=='protect' for r in f.audit['contingent_search']['proposals'].values() for t in r['branches']))
        d,_=select([c],f,p);self.assertEqual(d[0]['action_id'],'protect');self.assertEqual(digest(c),saved)

    def test_progress_uses_actual_flexibility_not_legacy_tier(self):
        c=scene();w=ProgressWatch(c['scope'],ProgressConfig(grace=1,preserve_persona_tier=True))
        req=PurposeRequest(0.,0.,{'protect':Activity('idle','unsupported repetition'),'escape':Activity('attempt','alternative')})
        w.observe(0,'protect',req,PurposeFeedback(0.,0.))
        allowed,audit=w.mask(c,req,policy=policy(c,0.));self.assertTrue(audit['applied']);self.assertEqual(allowed,{'escape'})
        allowed,audit=w.mask(c,req,policy=policy(c,1.));self.assertFalse(audit['applied']);self.assertIn('protect',allowed)
        # Once another means preserves the same principle, even the unyielding
        # actor can discard an unsupported method without changing its value.
        c['actions'][0]['outcomes'][0]['values']['benevolence']=.3
        allowed,audit=w.mask(c,req,policy=policy(c,1.));self.assertTrue(audit['applied']);self.assertEqual(allowed,{'escape'})

class SubjectiveBeliefTests(unittest.TestCase):
    models={'old':{'A':.9,'B':.1},'new':{'A':.1,'B':.9}}
    def tracker(self,inertia):
        return HypothesisTracker(('old','new'),inertia=inertia,prior={'old':.95,'new':.05},assumption_trust=.9)
    def test_prior_affects_prediction_without_fabricating_evidence(self):
        s=self.tracker(1.).snapshot();self.assertEqual(s.observations,0);self.assertEqual(s.evidence_confidence,0.)
        self.assertGreater(s.predict(self.models)['A'],.5)
        self.assertIn('subjective',s.record()['prediction_basis'])

    def test_stubborn_updates_slower_but_can_relearn_changed_opponent(self):
        a=self.tracker(0.);b=self.tracker(1.)
        a.observe(self.models,'B','0');b.observe(self.models,'B','0')
        self.assertGreater(b.snapshot().weights[0],a.snapshot().weights[0])
        for i in range(1,24):b.observe(self.models,'B',str(i))
        self.assertGreater(b.snapshot().weights[1],.9)
        self.assertGreater(b.snapshot().predict(self.models)['B'],.5)

    def test_single_failed_prediction_does_not_erase_prior_but_repeated_failures_downgrade_trust(self):
        t=self.tracker(1.);t.observe(self.models,'B','1');self.assertGreater(t.snapshot().weights[0],.5)
        for i in range(2,5):t.observe(self.models,'B',str(i))
        self.assertLess(t.snapshot().assumption_trust,.2)

    def test_forced_response_does_not_create_evidence_or_revise_prior(self):
        t=self.tracker(1.);before=t.snapshot().record()
        t.observe({'old':{'only':1.},'new':{'only':1.}},'only','forced')
        self.assertEqual(t.snapshot().record(),before)
        with self.assertRaises(ValueError):t.observe(self.models,'A','forced')

    def test_invalid_assumptions_rejected(self):
        for params in ({'inertia':True},{'inertia':1.1},{'assumption_trust':.5},{'prior':{'old':1.,'new':0.}}):
            with self.assertRaises(ValueError):HypothesisTracker(('old','new'),**params)

    def test_concentrated_prior_and_uninformative_events_are_not_evidence(self):
        t=self.tracker(1.);flat={'old':{'A':.8,'B':.2},'new':{'A':.8,'B':.2}}
        for i in range(12):t.observe(flat,'A',str(i))
        self.assertEqual(t.snapshot().evidence_confidence,0.)
        self.assertEqual(t.snapshot().evidence_weights,(.5,.5))
        self.assertNotEqual(t.snapshot().weights,t.snapshot().evidence_weights)

    def test_evidence_state_is_independent_of_subjective_prior_and_inertia(self):
        a=self.tracker(0.);b=self.tracker(1.)
        for i in range(6):
            a.observe(self.models,'B',str(i));b.observe(self.models,'B',str(i))
            self.assertEqual(a.snapshot().evidence_weights,b.snapshot().evidence_weights)
            self.assertEqual(a.snapshot().evidence_confidence,b.snapshot().evidence_confidence)
        self.assertNotEqual(a.snapshot().weights,b.snapshot().weights)

    def test_diagnostic_world_randomness_is_not_given_to_actor(self):
        from reflex.commitment_experiment import pledge_episode
        a=pledge_episode(0,1.,1.,turns=2);b=pledge_episode(1,1.,1.,turns=2)
        self.assertEqual(a['trace'][0]['context_hash'],b['trace'][0]['context_hash'])
        self.assertEqual(a['trace'][0]['decision'],b['trace'][0]['decision'])
        self.assertTrue(all(k.startswith('pledge:') for t in a['trace'] for k in t['belief_after']['ids']))

    def test_checkpoint_preserves_prior_inertia_and_actual_future_updates(self):
        t=self.tracker(1.);t.observe(self.models,'B','1');u=HypothesisTracker.from_record(t.record())
        self.assertEqual(t.record(),u.record())
        for i in range(2,8):
            t.observe(self.models,'B',str(i));u.observe(self.models,'B',str(i));self.assertEqual(t.record(),u.record())
        bad=t.record();bad['logs'][0]=float('nan')
        with self.assertRaises(ValueError):HypothesisTracker.from_record(bad)
