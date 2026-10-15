import copy,json,unittest
from reflex.core import FEATURES,digest,Policy
from reflex.examples import context,action,effect
from reflex.loop_predictor import CategoricalReader
from reflex.decision_loop import DecisionLoop,Request
from reflex.judgment import Binding,ReadControl


class FixedForecasts(CategoricalReader):
    def forecasts(self,c):return {'A':.1,'B':.9},{'A':.9,'B':.1}


class ChoiceAdoptionTests(unittest.TestCase):
    def fixture(self,adoption):
        c=context('decision-value',[action('safe',effect(.1)),action('variable',effect(1.,p=.1),effect(-.05,p=.9))])
        outcome=lambda cc,k,z:effect(.1 if k=='safe' else (1. if z=='A' else -.05))
        r=FixedForecasts(c['scope'],('first','second'),lambda cc:{'first':{'A':.9,'B':.1},'second':{'A':.1,'B':.9}},outcome,known_conditionals=True)
        a=DecisionLoop(c,policy=Policy(principle_priority='finite'),predictor=r,read_control=ReadControl(max_nodes=64),
            decision_metric='persona',response_adoption=adoption)
        return c,a

    def test_less_accurate_probability_can_support_better_personal_choice_and_be_revoked(self):
        records={}
        for adoption in ('joint','persona_choice'):
            c,a=self.fixture(adoption);choices=[]
            for tick,symbol in enumerate('ABBAB'*8+'B'*20):
                c['tick']=tick
                req=Request(c,{k:Binding(k,'same',FEATURES) for k in ('safe','variable')},{k:() for k in ('safe','variable')},threatened=True)
                r=a.decide(req,False);choices.append(r['decision']['action_id'])
                a.abandon(r['ticket'],event={'revealed_action':symbol})
            records[adoption]=choices
            self.assertFalse(a.memory.entries)
            self.assertFalse(a.evidence.accepted('response:'+digest(a.predictor.key)))
            self.assertEqual(choices[-5:],['safe']*5)
        self.assertEqual(records['joint'],['safe']*60)
        self.assertIn('variable',records['persona_choice'][10:40])

    def test_adoption_is_owned_checkpointed_and_cannot_use_purpose_only_evidence(self):
        c,a=self.fixture('persona_choice');record=json.loads(json.dumps(a.record()))
        self.assertEqual(DecisionLoop.from_record(c,record,predictor=a.predictor).response_adoption,'persona_choice')
        with self.assertRaises(ValueError):DecisionLoop.from_record(c,record,predictor=a.predictor,response_adoption='joint')
        with self.assertRaises(ValueError):DecisionLoop(c,decision_metric='purpose',response_adoption='persona_choice')
        c,old=self.fixture('joint');record=old.record();self.assertNotIn('response_adoption',record)
        self.assertEqual(DecisionLoop.from_record(c,record,predictor=old.predictor).response_adoption,'joint')


if __name__=='__main__':unittest.main()
