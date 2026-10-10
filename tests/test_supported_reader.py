import copy
import json
import unittest
from reflex.decision_loop import DecisionLoop
from reflex.loop_predictor import SupportedCategoricalReader,ValidatedSupportedCategoricalReader,GuardedSupportedCategoricalReader
from reflex.examples import effect
from reflex.planning import vector
from tests.test_decision_loop import fixture,request


def reader(c,support_key='resource-opportunity-v1',cls=SupportedCategoricalReader):
    return cls(c['scope'],('x-model','y-model'),
        lambda c:{'x-model':{'x':.95,'y':.05},'y-model':{'x':.05,'y':.95}},
        lambda c,a,r:effect(.8 if r=='x' else -.8),
        classes=('continuing','settling'),opportunity_for=lambda c:c['facts']['opportunity'],support_key=support_key)


class SupportedReaderTests(unittest.TestCase):
    def test_observed_lifecycle_batch_and_checkpoint_keep_support_local(self):
        # Two independently named owners use the same reader and lifecycle;
        # these fixtures are not complete game strength comparisons.
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

    def test_validated_reader_borrows_only_with_local_proof_and_replays_real_lifecycle(self):
        c=fixture(negative=True);c['facts']['opportunity']='continuing'
        template=reader(c,cls=ValidatedSupportedCategoricalReader);loop=DecisionLoop(c,predictor=template)
        for tick in range(12):
            c['tick']=tick;c['facts']['opportunity']='continuing' if tick<8 else 'settling'
            r=loop.decide(request(c,threatened=True),False)
            loop.observe(r['ticket'],vector(effect(.8)),{'revealed_action':'x'})
        local=loop.predictor.support.select('settling');self.assertEqual(local.tracker.observations,4)
        self.assertEqual(loop.predictor.shared.tracker.observations,12)
        saved=json.loads(json.dumps(loop.record()));restored=DecisionLoop.from_record(c,saved,predictor=template)
        c['tick']=12
        self.assertEqual(loop.decide(request(c,threatened=True),False),restored.decide(request(c,threatened=True),False))
        self.assertEqual(template.shared.tracker.observations,0)

    def test_transfer_revocation_and_checkpoint_cannot_authorize_a_different_class(self):
        c=fixture();c['facts']['opportunity']='continuing';r=reader(c,cls=ValidatedSupportedCategoricalReader)
        for _ in range(4):r.transfer.categorical('continuing',{'x':.5,'y':.5},{'x':.9,'y':.1},'x')
        self.assertIs(r.selected(c),r.shared)
        c['facts']['opportunity']='settling';self.assertIs(r.selected(c),r.support.select('settling'))
        for _ in range(4):r.transfer.categorical('continuing',{'x':.5,'y':.5},{'x':.9,'y':.1},'y')
        c['facts']['opportunity']='continuing';self.assertIs(r.selected(c),r.support.select('continuing'))
        before=r.record();bad=copy.deepcopy(before);bad['transfer']['entries'][0]['gains']=[float('nan')]
        with self.assertRaises(ValueError):r.restored(bad)
        self.assertEqual(r.record(),before)

    def test_guarded_specialization_is_locally_earned_and_checkpoint_preserves_incumbent(self):
        c=fixture(negative=True);c['facts']['opportunity']='continuing'
        template=reader(c,cls=GuardedSupportedCategoricalReader);loop=DecisionLoop(c,predictor=template)
        self.assertIs(template.selected(c),template.shared)
        for tick in range(24):
            c['tick']=tick;c['facts']['opportunity']='continuing' if tick<8 or tick%2 else 'settling'
            reveal='x' if c['facts']['opportunity']=='continuing' else 'y';r=loop.decide(request(c,threatened=True),False)
            loop.observe(r['ticket'],vector(effect(.8 if reveal=='x' else -.8)),{'revealed_action':reveal})
        self.assertTrue(loop.predictor.transfer.accepted('settling'))
        c['facts']['opportunity']='settling'
        self.assertIs(loop.predictor.selected(c),loop.predictor.support.select('settling'))
        saved=json.loads(json.dumps(loop.record()));restored=DecisionLoop.from_record(c,saved,predictor=template)
        with self.assertRaises(ValueError):reader(c,cls=ValidatedSupportedCategoricalReader).restored(saved['predictor'])
        c['tick']=24
        self.assertEqual(loop.decide(request(c,threatened=True),False),restored.decide(request(c,threatened=True),False))


if __name__=='__main__':unittest.main()
