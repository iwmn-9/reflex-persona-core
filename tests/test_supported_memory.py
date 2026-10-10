import copy
from dataclasses import replace
import unittest
import numpy as np
from reflex.board_models import ThanksPosition
from reflex.goofspiel import Position
from reflex.strong_search import PublicMemory, _thanks_simulate
from reflex.supported_memory import SupportBanks, SupportedPublicMemory,ValidatedPublicMemory


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

    def test_transfer_requires_local_evidence_and_revokes_when_it_fails(self):
        s=replace(ThanksPosition.start(4,35),remaining=0);m=ValidatedPublicMemory('no_thanks',1)
        local=m.support.select('current_card_only');key=m._key('current_card_only',0)
        self.assertIs(m.selected_bank('current_card_only',0),local)
        for i in range(4):m.transfer.categorical(key,{'TAKE':.5,'PASS':.5},{'TAKE':.9,'PASS':.1},'TAKE')
        self.assertIs(m.selected_bank('current_card_only',0),m.shared)
        self.assertIs(m.selected_bank('future_draws',0),m.support.select('future_draws'))
        for i in range(4):m.transfer.categorical(key,{'TAKE':.5,'PASS':.5},{'TAKE':.9,'PASS':.1},'PASS')
        self.assertFalse(m.transfer.accepted(key))

    def test_validated_transfer_preserves_local_bank_and_commits_atomically(self):
        early=ThanksPosition.start(4,35);late=replace(early,remaining=0);m=ValidatedPublicMemory('no_thanks',1)
        before=copy.deepcopy(m.support.select('future_draws').record())
        for i in range(8):
            predicted=m.predict(late,0)['TAKE'];record=m.observe(late,0,'TAKE',f'late-{i}')
            self.assertEqual(record['predicted_probability'],predicted)
            self.assertEqual(record['predicted_probability'],record['mixture_probability'])
        self.assertEqual(m.support.select('future_draws').record(),before)
        unchanged=copy.deepcopy(m.record())
        with self.assertRaises(ValueError):m.observe(late,0,'unknown','bad')
        with self.assertRaises(ValueError):m.observe(late,0,'TAKE','late-7')
        self.assertEqual(unchanged,m.record())


if __name__=='__main__':unittest.main()
