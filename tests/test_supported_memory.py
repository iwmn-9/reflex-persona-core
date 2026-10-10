import copy
from dataclasses import replace
import unittest
import numpy as np
from reflex.board_models import ThanksPosition
from reflex.goofspiel import Position
from reflex.strong_search import PublicMemory, _thanks_simulate, search,PERSONA
from reflex.supported_memory import SupportBanks, SupportedPublicMemory,ValidatedPublicMemory,GuardedPublicMemory


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
        for factory in (SupportedPublicMemory,GuardedPublicMemory):
            s=Position.start(4,3);old=PublicMemory('goofspiel',0);new=factory('goofspiel',0)
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

    def test_forced_responses_never_supply_transfer_trials_even_with_roundoff(self):
        m=ValidatedPublicMemory('no_thanks',1);s=replace(ThanksPosition.start(4,35),remaining=0,chips=(0,11,11,11))
        rng=np.random.default_rng(3)
        for i in range(100):
            m.shared.trackers[0].logs=list(rng.normal(size=12)*4)
            record=m.observe(s,0,'TAKE',f'forced-{i}')
            self.assertFalse(record['transfer']['scored'])
        self.assertEqual(m.transfer.entries,{})

    def test_guarded_specialization_keeps_incumbent_until_local_proof_and_revokes(self):
        s=replace(ThanksPosition.start(4,35),remaining=0);m=GuardedPublicMemory('no_thanks',1)
        local=m.support.select('current_card_only');key=m._key('current_card_only',0)
        self.assertIs(m.selected_bank('current_card_only',0),m.shared)
        for _ in range(4):m.transfer.categorical(key,{'TAKE':.5,'PASS':.5},{'TAKE':.9,'PASS':.1},'TAKE')
        self.assertIs(m.selected_bank('current_card_only',0),local)
        self.assertIs(m.selected_bank('future_draws',0),m.shared)
        for _ in range(4):m.transfer.categorical(key,{'TAKE':.5,'PASS':.5},{'TAKE':.9,'PASS':.1},'PASS')
        self.assertIs(m.selected_bank('current_card_only',0),m.shared)
        self.assertEqual(m.comparison({'TAKE':.1},{'TAKE':.9}),({'TAKE':.9},{'TAKE':.1}))

    def test_guarded_observations_score_shared_against_local_before_update(self):
        s=ThanksPosition.start(4,35);late=replace(s,remaining=0);m=GuardedPublicMemory('no_thanks',1)
        for i in range(8):m.observe(s,0,'PASS',f'early-{i}')
        local_before=copy.deepcopy(m.support.select('future_draws').record())
        for i in range(8):
            shared=m.shared.predict(late,0);local=m.support.select('current_card_only').predict(late,0)
            record=m.observe(late,0,'TAKE',f'late-{i}')
            self.assertEqual(record['shared_baseline_probability'],shared['TAKE'])
            if record['transfer']['scored']:
                loss=lambda f:float(np.mean([(v-(k=='TAKE'))**2 for k,v in f.items()]))
                self.assertAlmostEqual(record['transfer']['gain'],loss(shared)-loss(local))
        self.assertEqual(m.support.select('future_draws').record(),local_before)

    def test_unearned_specialization_preserves_entire_incumbent_rollout_and_search(self):
        for remaining in (1,0):
            s=replace(ThanksPosition.start(4,35),remaining=remaining,turn=1)
            old=PublicMemory('no_thanks',1);new=GuardedPublicMemory('no_thanks',1)
            for i in range(6):
                early=replace(s,remaining=8,turn=0)
                old.observe(early,0,'PASS',str(i));new.observe(early,0,'PASS',str(i))
            self.assertFalse(new._specialized(True));before=copy.deepcopy(new.record())
            for adaptive in (True,False):
                a=search('no_thanks',s,1,old,adaptive,PERSONA,np.random.default_rng(19))
                b=search('no_thanks',s,1,new,adaptive,PERSONA,np.random.default_rng(19))
                self.assertEqual(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1]);np.testing.assert_array_equal(a[2],b[2])
                self.assertEqual(a[3],b[3])
            self.assertEqual(new.record(),before)

    def test_specialization_reuses_shared_hypotheses_for_other_rivals_across_phases(self):
        m=GuardedPublicMemory('no_thanks',1)
        for _ in range(4):m.transfer.categorical(m._key('current_card_only',0),{'TAKE':.5,'PASS':.5},{'TAKE':.9,'PASS':.1},'TAKE')
        banks=m.rollout_banks();self.assertEqual(banks[1]['reuse_from'],[-1,0,0,0])
        self.assertEqual(len(m.rollout_banks(False)),1)
        np.testing.assert_array_equal(m.rollout_bank_indices(np.array([0,1]),False),[0,0])


if __name__=='__main__':unittest.main()
