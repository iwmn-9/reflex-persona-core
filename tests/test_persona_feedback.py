import copy
import json
import unittest
import numpy as np
from reflex.core import Policy,compile_batch,character_key,FEATURES
from reflex.examples import action,context,effect
from reflex.decision_loop import DecisionLoop,Request,Reading
from reflex.judgment import Binding
from reflex.planning import vector


def fixture():
    c=context('feedback',[action('keep',effect(.45)),action('help',effect(.1,values={'benevolence':.8}))],
        values={'benevolence':.9},mode='principle')
    c['state']['primary_need']='physiology'
    return c


class PersonaFeedbackTests(unittest.TestCase):
    def test_outcome_decomposition_preserves_full_original_scores_and_choice(self):
        c=fixture();c['actions'][1]['outcomes']=[effect(.8,p=.3,values={'benevolence':.9}),effect(-.4,p=.7,values={'benevolence':-.5})]
        c['actions'][0]['switch_cost']=.12;c['actions'][1]['confidence']=.6
        b=compile_batch([c]);bias=np.array([[.11,-.07]])
        for priority in ('lexicographic','finite','personal'):
            p=Policy(np.arange(8)*.005,principle_priority=priority,**({'commitments':{character_key(c):.5}} if priority=='personal' else {}))
            for appraisal in (None,bias):
                audit={};old=p.decide(b,False,appraisal=appraisal);new=p.decide(b,False,appraisal=appraisal,outcome_audit=audit)
                for k in old.__dataclass_fields__:np.testing.assert_array_equal(getattr(old,k),getattr(new,k))
                np.testing.assert_allclose(np.einsum('nak,nak->na',b.probability,audit['scores']),new.scores,atol=1e-12)

    def test_audit_is_frozen_and_does_not_alias_decisions(self):
        b=compile_batch([fixture()]);audit={};p=Policy();d=p.decide(b,False,outcome_audit=audit)
        scores=d.scores.copy();audit['expected'][:]=100;audit['scores'][:]=100
        np.testing.assert_array_equal(d.scores,scores)

    def test_tiled_batch_preserves_full_outcome_audit(self):
        from reflex.scaling import TiledPolicy
        cs=[fixture() for _ in range(7)]
        for i,c in enumerate(cs):c['scope']['npc']=str(i);c['needs']['physiology']['deficit']=i/7
        b=compile_batch(cs);a={};z={};bias=np.full(b.legal.shape,.1)
        x=Policy().decide(b,False,appraisal=bias,outcome_audit=a)
        y=TiledPolicy(2).decide(b,False,appraisal=bias,outcome_audit=z)
        for k in x.__dataclass_fields__:np.testing.assert_array_equal(getattr(x,k),getattr(y,k))
        for k in ('scores','expected'):np.testing.assert_array_equal(a[k],z[k])
        np.testing.assert_allclose((b.probability*z['scores']).sum(2),y.scores,atol=1e-12)

    def test_known_rule_persona_cost_is_not_a_purpose_failure(self):
        def reader(c,cap):
            rows={'keep':[effect(.45),effect(.45)],
                'help':[effect(.1,values={'benevolence':.8}),effect(-.5,values={'benevolence':-.8})]}
            for a in c['actions']:
                a['outcomes']=copy.deepcopy(rows[a['id']])
                for r in a['outcomes']:r['p']=.5
            return Reading(c,4,'help-forecast',responses={'yes':.5,'no':.5},known_conditionals=True,response_candidate={'yes':.95,'no':.05})
        runs={}
        for metric in ('purpose','persona'):
            c=fixture();c['actions'][1]['outcomes']=[effect(-.5,values={'benevolence':-.8})]
            loop=DecisionLoop(c,policy=Policy(principle_priority='finite'),decision_metric=metric);records=[]
            for tick in range(12):
                c['tick']=tick
                req=Request(c,{a['id']:Binding(a['id'],'same',('objective','benevolence')) for a in c['actions']},{a['id']:('cost',) for a in c['actions']},reader=reader,threatened=True)
                r=loop.decide(req,False)
                observed=effect(.1,values={'benevolence':.8}) if r['decision']['action_id']=='help' else effect(.45)
                update=loop.observe(r['ticket'],vector(observed),{'revealed_action':'yes'})
                records.append(update['decision_comparison'])
            runs[metric]=(loop,records)
        personal=[r for r in runs['persona'][1] if r]
        self.assertTrue(personal)
        self.assertTrue(all(r['purpose_delta']<0 and r['persona_delta']>0 for r in personal))
        self.assertTrue(any(e['state']=='active' for k,e in runs['persona'][0].evidence.entries.items() if k.endswith(':persona-v1')))
        self.assertFalse(any(e['state']=='active' for k,e in runs['purpose'][0].evidence.entries.items() if k.endswith(':decision')))
        for r in personal:
            self.assertAlmostEqual(r['purpose_delta'],r['planned_purpose_delta']+r['purpose_surprise_delta'])
            self.assertGreater(r['candidate_rationale']['expected_purpose_cost'],0)
            self.assertGreater(r['candidate_rationale']['persona_advantage'],0)
        self.assertNotIn(('help','same'),runs['purpose'][0].memory.entries)

    def test_frozen_appraisal_keeps_negative_surprise(self):
        c=fixture();c['actions'][1]['outcomes']=[effect(.8,p=.9),effect(-.8,p=.1)]
        audit={};b=compile_batch([c]);Policy().decide(b,False,outcome_audit=audit)
        j=b.ids[0].index('help');before=audit['scores'].copy()
        c['values']['benevolence']=0;c['needs']['physiology']['deficit']=1
        np.testing.assert_array_equal(audit['scores'],before)
        self.assertLess(audit['scores'][0,j,1],0)

    def test_rule_contradiction_cannot_update_persona_evidence(self):
        c=fixture();loop=DecisionLoop(c,decision_metric='persona')
        def reader(c,cap):
            for a in c['actions']:a['outcomes']=[effect(.3,p=.5),effect(-.4,p=.5)]
            return Reading(c,4,'rule',responses={'a':.5,'b':.5},known_conditionals=True,response_candidate={'a':.8,'b':.2})
        req=Request(c,{a['id']:Binding(a['id'],'s',FEATURES) for a in c['actions']},{a['id']:() for a in c['actions']},reader=reader,threatened=True)
        r=loop.decide(req,False);saved=copy.deepcopy(loop.evidence.entries)
        with self.assertRaisesRegex(ValueError,'contradicts'):loop.observe(r['ticket'],vector(effect(.9)),{'revealed_action':'a'})
        self.assertEqual(loop.evidence.entries,saved);self.assertIsNotNone(loop.pending)
        loop.observe(r['ticket'],vector(effect(.3)),{'revealed_action':'a'})

    def test_checkpoint_retains_metric_and_rejects_reinterpretation(self):
        c=fixture();p=Policy(principle_priority='finite');loop=DecisionLoop(c,p,decision_metric='persona')
        saved=json.loads(json.dumps(loop.record()));restored=DecisionLoop.from_record(c,saved)
        self.assertEqual(restored.decision_metric,'persona');self.assertEqual(saved,json.loads(json.dumps(restored.record())))
        with self.assertRaisesRegex(ValueError,'metric changed'):DecisionLoop.from_record(c,saved,decision_metric='purpose')
        old=DecisionLoop(c,p).record();self.assertNotIn('decision_metric',old)
        self.assertEqual(DecisionLoop.from_record(c,old).decision_metric,'purpose')
        with self.assertRaises(ValueError):DecisionLoop(c,p,decision_metric='guessed')

    def test_mixed_batch_uses_each_owners_metric(self):
        # No reader: diagnostic mode cannot change ordinary reflex decisions.
        cs=[fixture(),fixture()];cs[1]['scope']['npc']='other'
        loops=[DecisionLoop(c,decision_metric=m) for c,m in zip(cs,('purpose','persona'))]
        reqs=[Request(c,{a['id']:Binding(a['id'],'same',()) for a in c['actions']},{a['id']:FEATURES for a in c['actions']}) for c in cs]
        singles=[DecisionLoop(c,decision_metric=m).decide(r,False) for c,r,m in zip(cs,reqs,('purpose','persona'))]
        self.assertEqual(DecisionLoop.decide_batch(list(zip(loops,reqs)),False),singles)


if __name__=='__main__':unittest.main()
