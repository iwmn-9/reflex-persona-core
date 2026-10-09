import copy
import json
import unittest
import numpy as np
from reflex.core import Policy, compile_batch, digest
from reflex.examples import context, action, effect
from reflex.social import SocialBinding, SocialEvent, SocialMemory, SocialPopulation


def fixture(npc='care'):
    acts=[action('help',effect(.35,values={'benevolence':.6,'universalism':.4},style={'agreeableness':.8})),
          action('self',effect(.5)),action('revenge',effect(-.8))]
    acts[0]['target']='partner';acts[2]['target']='partner'
    c=context('social-test',acts,values={'benevolence':.85,'universalism':.5},traits={'agreeableness':.95},mode='principle')
    c['scope']['npc']=npc
    return c


class SocialTests(unittest.TestCase):
    def test_default_and_zero_appraisal_are_exactly_compatible(self):
        c=fixture();b=compile_batch([c]);p=Policy(principle_priority='finite')
        for stochastic in (True,False):
            a=p.decide(b,stochastic);z=p.decide(b,stochastic,appraisal=np.zeros_like(b.legal,float))
            self.assertEqual(a.records(b),z.records(b));np.testing.assert_equal(a.scores,z.scores)

    def test_repeated_exploitation_changes_means_without_changing_values(self):
        c=fixture();original=digest(c);m=SocialMemory(c);p=Policy(principle_priority='finite');b=compile_batch([c])
        bindings={'help':SocialBinding('partner',1),'revenge':SocialBinding('partner',-1)}
        self.assertEqual(p.decide(b,False).records(b)[0]['action_id'],'help')
        for t in range(12):m.observe(SocialEvent('partner',str(t),t,-1))
        bias,_=m.scores(b.ids[0],b.targets[0],bindings,12)
        d=p.decide(b,False,appraisal=bias[None,:])
        self.assertEqual(d.records(b)[0]['action_id'],'self')
        self.assertEqual(digest(c),original);self.assertEqual(c['values'],m.values)
        self.assertFalse(d.eligible[0,b.ids[0].index('revenge')])

    def test_grievance_is_recipient_specific_and_repairable(self):
        c=fixture();m=SocialMemory(c)
        for t in range(8):m.observe(SocialEvent('partner',str(t),t,-1))
        self.assertLess(m.status('partner',8)['attitude'],-.6)
        self.assertEqual(m.status('innocent',8)['attitude'],0)
        for t in range(8,16):m.observe(SocialEvent('partner',str(t),t,1))
        self.assertGreater(m.status('partner',16)['attitude'],.5)
        self.assertLess(abs(m.status('partner',160)['attitude']),.02)

    def test_bad_luck_is_not_betrayal_and_duplicate_feedback_is_rejected(self):
        m=SocialMemory(fixture());m.observe(SocialEvent('partner','lottery',0,-1,agency=False))
        self.assertEqual(m.status('partner',0)['attitude'],0)
        saved=m.record()
        with self.assertRaises(ValueError):m.observe(SocialEvent('partner','lottery',0,-1))
        self.assertEqual(saved,m.record())

    def test_distress_preserves_universal_concern_and_no_emotion_overrides_masks(self):
        c=fixture();c['actions'][2]['legal']=False;m=SocialMemory(c);b=compile_batch([c]);p=Policy(principle_priority='finite')
        for t in range(12):m.observe(SocialEvent('partner',str(t),t,-1))
        ordinary,_=m.scores(b.ids[0],b.targets[0],{'help':SocialBinding('partner',1)},12)
        urgent,_=m.scores(b.ids[0],b.targets[0],{'help':SocialBinding('partner',1,1)},12)
        self.assertGreater(urgent[b.ids[0].index('help')],ordinary[b.ids[0].index('help')])
        bias=np.full(b.legal.shape,1.5);d=p.decide(b,False,appraisal=bias)
        self.assertFalse(d.eligible[0,b.ids[0].index('revenge')])

    def test_checkpoint_and_bounded_eviction_preserve_ownership_and_forgetting(self):
        c=fixture();m=SocialMemory(c,capacity=2)
        for t in range(4):m.observe(SocialEvent(str(t),str(t),t,-1))
        self.assertEqual(m.status('0',4)['attitude'],0)
        restored=SocialMemory.from_record(c,json.loads(json.dumps(m.record())))
        self.assertEqual(m.record(),restored.record())
        with self.assertRaises(ValueError):SocialMemory.from_record(fixture('other'),m.record())

    def test_population_uses_observed_feedback_and_feedback_batch_is_atomic(self):
        cs=[fixture('one'),fixture('two')];p=SocialPopulation(cs,Policy(principle_priority='finite'))
        bindings=[{'help':SocialBinding('partner',1)}]*2
        for t in range(12):
            p.step(False,bindings=bindings)
            p.observe_batch([[SocialEvent('partner',str(t),t,-1)]]*2)
        before=[m.record() for m in p.social];ticks=p.ticks.copy()
        with self.assertRaises(ValueError):p.step(bindings=[bindings[0],{'help':SocialBinding('wrong',1)}])
        np.testing.assert_equal(p.ticks,ticks)
        with self.assertRaises(ValueError):p.observe_batch([[SocialEvent('partner','fresh',11,1)],[SocialEvent('partner','future',12,1)]])
        self.assertEqual(before,[m.record() for m in p.social])
        d=p.step(False,bindings=bindings)
        self.assertEqual(p.action_ids(d),['self','self'])

    def test_numeric_regret_and_appraisal_contracts_reject_invalid_inputs(self):
        b=compile_batch([fixture()]);p=Policy(principle_priority='finite')
        for bias,purpose in (([[float('nan')]*3],None),([[2]*3],None),([[0]*3],[[2]*3])):
            with self.assertRaises(ValueError):p.decide(b,appraisal=bias,purpose=purpose)
        with self.assertRaises(ValueError):p.decide(b,purpose=np.zeros_like(b.legal,float))

    def test_experiment_stops_dead_actor_and_replays_deterministically(self):
        from reflex.social_experiment import run
        a,trace=run(0,'patrol','exploitative',False)
        other,replayed=run(0,'patrol','exploitative',False)
        self.assertEqual(a,other);self.assertEqual(trace,replayed)
        ended=False
        for row in (r for r in trace if r['npc']=='care'):
            if ended:
                self.assertEqual(row['action'],'finished');self.assertEqual(row['before'],row['after'])
            ended|=row['terminal']
        self.assertTrue(ended)

    def test_personality_sacrifice_remains_and_innocent_recipient_is_not_punished(self):
        c=fixture();c['actions'][0]['outcomes'][0]['objective']=.1
        c['actions'][1]['outcomes'][0]['objective']=.9
        innocent=copy.deepcopy(c['actions'][0]);innocent.update(id='help_innocent',target='innocent')
        c['actions'].append(innocent);b=compile_batch([c]);p=Policy(principle_priority='finite')
        self.assertIn(p.decide(b,False).records(b)[0]['action_id'],('help','help_innocent'))
        m=SocialMemory(c)
        for t in range(12):m.observe(SocialEvent('partner',str(t),t,-1))
        bias,_=m.scores(b.ids[0],b.targets[0],{'help':SocialBinding('partner',1),'help_innocent':SocialBinding('innocent',1)},12)
        d=p.decide(b,False,appraisal=bias[None,:])
        self.assertEqual(d.records(b)[0]['action_id'],'help_innocent')

    def test_reordered_social_batch_has_same_actor_choices(self):
        cs=[fixture('one'),fixture('two')];policy=Policy(principle_priority='finite')
        together=SocialPopulation(cs,policy);reversed_pop=SocialPopulation(list(reversed(cs)),policy)
        bindings=[{'help':SocialBinding('partner',1)}]*2
        for t in range(12):
            a=together.step(bindings=bindings);b=reversed_pop.step(bindings=bindings)
            self.assertEqual(together.action_ids(a),list(reversed(reversed_pop.action_ids(b))))
            events=[[SocialEvent('partner',str(t),t,benefit)] for benefit in (-1,1)]
            together.observe_batch(events);reversed_pop.observe_batch(list(reversed(events)))
        self.assertEqual([m.record() for m in together.social],list(reversed([m.record() for m in reversed_pop.social])))


if __name__=='__main__':unittest.main()
