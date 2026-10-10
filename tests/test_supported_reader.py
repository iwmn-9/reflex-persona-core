import copy
import json
import unittest
from reflex.decision_loop import DecisionLoop
from reflex.loop_predictor import SupportedCategoricalReader
from reflex.examples import effect
from reflex.planning import vector
from tests.test_decision_loop import fixture,request


def reader(c,support_key='resource-opportunity-v1'):
    return SupportedCategoricalReader(c['scope'],('x-model','y-model'),
        lambda c:{'x-model':{'x':.95,'y':.05},'y-model':{'x':.05,'y':.95}},
        lambda c,a,r:effect(.8 if r=='x' else -.8),
        classes=('continuing','settling'),opportunity_for=lambda c:c['facts']['opportunity'],support_key=support_key)


class SupportedReaderTests(unittest.TestCase):
    def test_observed_lifecycle_batch_and_checkpoint_keep_support_local(self):
        # Different application domains use the same reader and lifecycle.
        contexts=[fixture('combat-agent',negative=True),fixture('auction-agent',negative=True)]
        for c in contexts:c['facts']['opportunity']='settling'
        originals=[reader(c) for c in contexts];loops=[DecisionLoop(c,predictor=r) for c,r in zip(contexts,originals)]
        for tick in range(8):
            for c in contexts:c['tick']=tick
            results=DecisionLoop.decide_batch([(loop,request(c,threatened=True)) for loop,c in zip(loops,contexts)],False)
            for loop,r in zip(loops,results):loop.observe(r['ticket'],vector(effect(.8)),{'revealed_action':'x'})
        for loop,c,original in zip(loops,contexts,originals):
            self.assertEqual(original.support.select('settling').tracker.observations,0)
            self.assertEqual(loop.predictor.support.select('settling').tracker.observations,8)
            self.assertEqual(loop.predictor.support.select('continuing').tracker.observations,0)
            settling=loop.predictor.forecasts(c)
            c['facts']['opportunity']='continuing';self.assertEqual(loop.predictor.forecasts(c),original.forecasts(c))
            self.assertNotEqual(settling,loop.predictor.forecasts(c))
            saved=json.loads(json.dumps(loop.record()));restored=DecisionLoop.from_record(c,saved,predictor=original)
            c['tick']=8
            self.assertEqual(loop.decide(request(c,threatened=True),False),restored.decide(request(c,threatened=True),False))
            self.assertEqual(loop.personality,c['personality']);self.assertEqual(loop.values,c['values'])

    def test_reader_trust_keys_do_not_transfer_between_opportunity_classes(self):
        c=fixture(negative=True);r=reader(c);c['facts']['opportunity']='settling';a=r(c,16)
        c['facts']['opportunity']='continuing';b=r(c,16)
        self.assertNotEqual(a.key,b.key);self.assertEqual(a.nodes,b.nodes)
        self.assertIsNone(r(c,1))

    def test_unknown_support_and_foreign_owner_cannot_change_memory(self):
        c=fixture();r=reader(c);before=r.record();c['facts']['opportunity']='undeclared'
        with self.assertRaises(ValueError):r.updated(c,None,{'revealed_action':'x'},'event')
        self.assertEqual(before,r.record())
        c['facts']['opportunity']='continuing';c['scope']['npc']='foreign'
        with self.assertRaises(ValueError):r.forecasts(c)
        self.assertEqual(before,r.record())

    def test_checkpoint_rejects_changed_support_definition_and_corrupt_bank_atomically(self):
        c=fixture();r=reader(c);before=r.record()
        with self.assertRaises(ValueError):reader(c,'changed-v2').restored(before)
        bad=copy.deepcopy(before);bad['banks']['settling']['logs']=[float('nan')]*2
        with self.assertRaises(ValueError):r.restored(bad)
        self.assertEqual(r.record(),before)


if __name__=='__main__':unittest.main()
