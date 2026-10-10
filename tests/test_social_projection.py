import copy
import unittest
import numpy as np
from reflex.core import Policy, compile_batch
from reflex.social import SocialBinding, SocialEvent, SocialMemory, SocialPopulation
from reflex.social_projection import CoarseSocialMemory, CoarseSocialPopulation, stance
from test_social import fixture


class SocialProjectionTests(unittest.TestCase):
    def test_weak_evidence_and_within_stance_nuance_are_omitted(self):
        self.assertEqual([stance(x) for x in (-1,-.6,-.15,-.149,0,.149,.15,.6,1)],
                         [-.5,-.5,-.5,0,0,0,.5,.5,.5])
        c=fixture();b=compile_batch([c]);m=CoarseSocialMemory(c)
        bindings={'help':SocialBinding('partner',1)}
        m.observe(SocialEvent('partner','first',0,-1))
        self.assertEqual(m.scores(b.ids[0],b.targets[0],bindings,0)[0].sum(),0)
        values=[]
        for t in range(1,9):
            m.observe(SocialEvent('partner',str(t),t,-1))
            score,audit=m.scores(b.ids[0],b.targets[0],bindings,t)
            if audit['help']['stance']==-.5:values.append(score[b.ids[0].index('help')])
        self.assertGreater(len(values),2);np.testing.assert_allclose(values,values[0],rtol=0,atol=1e-15)

    def test_evidence_is_retained_for_repair_forgetting_and_checkpoint(self):
        c=fixture();continuous=SocialMemory(c);coarse=CoarseSocialMemory(c)
        for t in range(16):
            e=SocialEvent('partner',str(t),t,-1 if t<8 else 1)
            continuous.observe(e);coarse.observe(e)
            self.assertEqual(continuous.record(),coarse.record())
        self.assertEqual(CoarseSocialMemory.from_record(c,coarse.record()).record(),coarse.record())
        self.assertGreater(stance(coarse.status('partner',16)['attitude']*coarse.status('partner',16)['confidence']),0)
        self.assertEqual(stance(coarse.status('partner',160)['attitude']*coarse.status('partner',160)['confidence']),0)

    def test_recipient_distress_and_purpose_floor_survive_projection(self):
        c=fixture();innocent=copy.deepcopy(c['actions'][0]);innocent.update(id='innocent',target='other')
        c['actions'].append(innocent);b=compile_batch([c]);m=CoarseSocialMemory(c)
        for t in range(8):m.observe(SocialEvent('partner',str(t),t,-1))
        ordinary,_=m.scores(b.ids[0],b.targets[0],{'help':SocialBinding('partner',1),'innocent':SocialBinding('other',1)},8)
        urgent,_=m.scores(b.ids[0],b.targets[0],{'help':SocialBinding('partner',1,1)},8)
        self.assertEqual(ordinary[b.ids[0].index('innocent')],0)
        self.assertGreater(urgent[b.ids[0].index('help')],ordinary[b.ids[0].index('help')])
        d=Policy(principle_priority='finite').decide(b,False,appraisal=ordinary[None,:])
        self.assertNotEqual(d.records(b)[0]['action_id'],'revenge')
        self.assertEqual(d.records(b)[0]['action_id'],'innocent')

    def test_coarse_population_zero_path_and_reordering_remain_compatible(self):
        cs=[fixture('one'),fixture('two')];policy=Policy(principle_priority='finite')
        baseline=SocialPopulation(cs,policy);coarse=CoarseSocialPopulation(cs,policy)
        reverse=CoarseSocialPopulation(list(reversed(cs)),policy)
        for t in range(12):
            a=baseline.step(bindings=[{},{}]);b=coarse.step(bindings=[{},{}]);r=reverse.step(bindings=[{},{}])
            self.assertEqual(baseline.action_ids(a),coarse.action_ids(b))
            self.assertEqual(coarse.action_ids(b),list(reversed(reverse.action_ids(r))))
            events=[[SocialEvent('partner',str(t),t,v)] for v in (-1,1)]
            coarse.observe_batch(events);reverse.observe_batch(list(reversed(events)))
        b=coarse.step(bindings=[{'help':SocialBinding('partner',1)}]*2)
        r=reverse.step(bindings=[{'help':SocialBinding('partner',1)}]*2)
        self.assertEqual(coarse.action_ids(b),list(reversed(reverse.action_ids(r))))

    def test_historical_continuous_patrol_result_is_preserved(self):
        from reflex.social_experiment import run
        result,_=run(0,'patrol','exploitative',True)
        # Published before the projection hook: e4acf14 social_appraisal evidence.
        care=next(a for a in result['actors'] if a['npc']=='care')
        self.assertEqual(care['aid_by_phase'],[4,1,1])
        self.assertEqual(care['score'],1.6);self.assertEqual(care['health'],3.5)
        expected=dict(attitude=-.48539235003269604,confidence=.6135117904356907,evidence=4.762203155904599)
        for key,value in expected.items():self.assertAlmostEqual(care['final_relation'][key],value,places=12)


if __name__=='__main__':unittest.main()
