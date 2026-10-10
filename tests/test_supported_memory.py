import copy
from dataclasses import replace
import unittest
import numpy as np
from reflex.board_models import ThanksPosition
from reflex.goofspiel import Position
from reflex.strong_search import PublicMemory, _thanks_simulate
from reflex.supported_memory import SupportBanks, SupportedPublicMemory


class SupportedMemoryTests(unittest.TestCase):
    def test_generic_support_is_finite_and_isolates_mutable_state(self):
        banks=SupportBanks(('opportunity','settling'),list)
        banks.select('settling').append('experience')
        self.assertEqual(banks.select('opportunity'),[])
        with self.assertRaises(ValueError):banks.select('undeclared')
        shared=[]
        with self.assertRaises(ValueError):SupportBanks(('a','b'),lambda:shared)

    def test_settlement_learning_cannot_rewrite_continuing_predictions(self):
        early=ThanksPosition.start(4,35);late=replace(early,remaining=0)
        m=SupportedPublicMemory('no_thanks',1);cold=copy.deepcopy(m)
        for i in range(16):m.observe(late,0,'PASS',f'late-{i}')
        self.assertEqual(m.predict(early,0),cold.predict(early,0))
        self.assertEqual(m.bank(early).record(),cold.bank(early).record())
        self.assertNotEqual(m.predict(late,0),cold.predict(late,0))
        # Learned terminal behavior remains available where it has support.
        self.assertEqual(m.rollout_banks()[1]['thresholds'][0],m.bank(late).recent_threshold(0))

    def test_continuing_learning_cannot_rewrite_settlement_or_other_owner(self):
        s=ThanksPosition.start(4,35);late=replace(s,remaining=0)
        m=SupportedPublicMemory('no_thanks',1);other=SupportedPublicMemory('no_thanks',2);before=other.record();cold=copy.deepcopy(m)
        for i in range(16):m.observe(s,0,'TAKE',f'early-{i}')
        self.assertEqual(m.predict(late,0),cold.predict(late,0));self.assertEqual(other.record(),before)
        self.assertNotEqual(m.predict(s,0),cold.predict(s,0))

    def test_duplicate_event_cannot_enter_another_bank_and_forced_move_is_uninformative(self):
        s=ThanksPosition.start(4,35);m=SupportedPublicMemory('no_thanks',1)
        m.observe(s,0,'TAKE','one')
        with self.assertRaises(ValueError):m.observe(replace(s,remaining=0),0,'TAKE','one')
        bank=copy.deepcopy(m.bank(s).record());forced=replace(s,chips=(0,11,11,11))
        r=m.observe(forced,0,'TAKE','forced');self.assertTrue(r['forced'])
        self.assertEqual(m.bank(s).record(),bank)

    def test_adaptation_inside_support_survives_and_bad_evidence_can_be_weakened(self):
        s=ThanksPosition.start(4,35);m=SupportedPublicMemory('no_thanks',1)
        for i in range(16):m.observe(s,0,'PASS',f'pass-{i}')
        before=m.predict(s,0)['TAKE']
        records=[m.observe(s,0,'TAKE',f'take-{i}') for i in range(16)]
        self.assertGreater(m.predict(s,0)['TAKE'],before)
        self.assertTrue(any(r['effective_retention']<.94 for r in records))
        self.assertLessEqual(len(m.ids[0]),64)

    def test_goof_single_bank_preserves_existing_predictions_and_learning(self):
        s=Position.start(4,3);old=PublicMemory('goofspiel',0);new=SupportedPublicMemory('goofspiel',0)
        for i in range(6):
            self.assertEqual(old.predict(s,1),new.predict(s,1))
            old.observe(s,1,'BID:3',str(i));new.observe(s,1,'BID:3',str(i))
        np.testing.assert_equal(old.rollout_banks(),new.rollout_banks())

    def test_virtual_future_uses_the_future_support_without_training_actual_memory(self):
        s=replace(ThanksPosition.start(4,5),remaining=0,chips=(22,22,0,0))
        m=SupportedPublicMemory('no_thanks',0)
        # Force only the recent hypothesis, matching a known exact chip ledger.
        for bank in m.support._banks.values():
            bank.weights=lambda actor,adaptive=True:np.eye(12)[10]
        m.bank(s).recent[1]=[(True,10),(False,12)]
        before=copy.deepcopy(m.record())
        sc,_=_thanks_simulate(s,0,m,True,('PASS',),(0,),16,np.random.default_rng(8))
        np.testing.assert_array_equal(sc[0],np.tile([-21,-18,0,0],(16,1)))
        self.assertEqual(m.record(),before)


if __name__=='__main__':unittest.main()
