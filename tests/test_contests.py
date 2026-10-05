import copy
from dataclasses import replace
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.contests import Public,awarded,settle,winners_vector,make_context,PublicBeliefs,target,decide,capital_price,evaluate
from reflex.contests_experiment import opening,run,replay
from reflex.core import Policy,compile_batch
from reflex.laboratory import profiles


class ContestTests(unittest.TestCase):
    def test_positive_negative_duplicate_and_zero_pots(self):
        s,_,_=opening('hagetaka',0,0)
        s=replace(s,prize=5)
        self.assertEqual(awarded(s,(15,15,12,7)),2)
        self.assertEqual(awarded(replace(s,prize=-5),(1,1,4,8)),2)
        self.assertEqual(awarded(replace(s,prize=-5,carry=5),(15,15,12,7)),2)
        self.assertIsNone(awarded(s,(15,15,7,7)))

    def test_carry_consumes_cards_then_awards_accumulated_pot(self):
        s,_,_=opening('hagetaka',0,0); s=replace(s,prize=5)
        after,winner=settle(s,(15,15,7,7))
        self.assertIsNone(winner); self.assertEqual(after.carry,5)
        self.assertNotIn(15,after.hands[0]); self.assertNotIn(7,after.hands[2])
        after=replace(after,prize=-3)
        last,winner=settle(after,(14,13,12,11))
        self.assertEqual(winner,0); self.assertEqual(last.scores,(2,0,0,0)); self.assertEqual(last.carry,0)

    def test_auction_rotating_tie_and_only_winner_pays(self):
        s,_,_=opening('auction',0,0); s=replace(s,round=2)
        after,winner=settle(s,(4,4,4,4))
        self.assertEqual(winner,2); self.assertEqual(after.budgets,(18,18,14,18))
        self.assertEqual(after.scores[2],s.prize)
        self.assertIsNone(awarded(s,(0,0,0,0)))
        poor=replace(s,budgets=(1,18,18,18))
        with self.assertRaises(ValueError): settle(poor,(2,1,1,1))

    def test_vector_referee_matches_scalar_for_3_to_5_players(self):
        rng=np.random.default_rng(57)
        for game in ('hagetaka','auction'):
            for n in (3,4,5):
                s,_,_=opening(game,2,0,n)
                for negative in (False,True):
                    if game=='hagetaka': s=replace(s,prize=-5 if negative else 5)
                    m=np.column_stack([rng.choice(s.legal(i),256) for i in range(n)])
                    expected=[-1 if (a:=awarded(s,tuple(map(int,row)))) is None else a for row in m]
                    self.assertEqual(winners_vector(s,m).tolist(),expected)

    def test_large_carry_keeps_shared_effects_bounded(self):
        s,_,_=opening('hagetaka',0,0)
        for carry in (55,-15):
            c,_=make_context(replace(s,prize=10,carry=carry),0,profiles()[0],2,0)
            compile_batch([c])
        with self.assertRaises(ValueError): replace(s,carry=100)

    def test_belief_reads_only_completed_unique_observations(self):
        s,_,_=opening('hagetaka',0,0); b=PublicBeliefs(4,0)
        before=b.forecast(s,1)
        self.assertEqual(b.trust(),0)
        b.reveal(s,(15,14,13,12))
        with self.assertRaises(ValueError): b.reveal(s,(15,14,13,12))
        self.assertEqual(b.trackers[1].snapshot().observations,1)
        self.assertEqual(before,b.forecast(s,1))  # too little evidence: retain fallback
        clone=PublicBeliefs(4,0); self.assertEqual(clone.forecast(s,1),before)

    def test_disadvantage_and_public_response_change_hypotheses(self):
        s,_,_=opening('auction',1,0); s=replace(s,prize=3)
        self.assertGreater(target(replace(s,scores=(12,0,0,0)),1,'pressure'),target(s,1,'pressure'))
        self.assertNotEqual(target(replace(s,last=(1,1,1,1)),1,'response'),target(replace(s,last=(6,6,6,6)),1,'response'))

    def test_observer_cannot_mix_foreign_game_or_roster(self):
        s,_,_=opening('hagetaka',0,0); b=PublicBeliefs(4,0); b.forecast(s,1)
        other,_,_=opening('auction',0,0)
        with self.assertRaises(ValueError): b.forecast(other,1)
        short,_,_=opening('hagetaka',0,0,3)
        with self.assertRaises(ValueError): b.forecast(short,1)

    def test_hidden_world_order_not_in_actor_and_batch_same_as_single(self):
        s,deck,_=opening('hagetaka',1,0); original=copy.deepcopy(s)
        rows=[make_context(s,0,p,5,0)[0] for p in profiles()]
        self.assertEqual(s,original)
        for c in rows:
            self.assertEqual(c['facts']['unseen_remaining_multiset'],str(tuple(sorted(deck[1:]))))
            self.assertNotIn('controllers',c['facts']); self.assertNotIn('current_bids',c['facts'])
        batch=compile_batch(rows)
        self.assertEqual(Policy().decide(batch).records(batch),[Policy().choose(c) for c in rows])

    def test_no_trust_gate_and_actual_complete_replay(self):
        s,_,_=opening('auction',0,0); beliefs=PublicBeliefs(4,0)
        _,d,stats=decide(s,0,profiles()[0],0,0,beliefs=beliefs)
        self.assertEqual(stats['reason'],'no_reliable_public_evidence')
        for game in ('hagetaka','auction'):
            r=run(game,profiles()[1],1,'responsive','gated')
            self.assertEqual(replay(r),30 if game=='hagetaka' else 24)
            for row in r['trace']:
                self.assertEqual(row['context']['facts']['budgets'],str(tuple(row['before']['budgets'])))
                self.assertTrue(all(x>=0 for x in row['after']['budgets']))

    def test_capital_future_value_uses_multiset_and_terminal_salvage(self):
        s,_,_=opening('auction',0,0); s=replace(s,prize=2)
        self.assertEqual(capital_price(replace(s,remaining=()),0),.25)
        self.assertGreater(capital_price(s,0),.25)
        self.assertEqual(capital_price(s,0),capital_price(replace(s,remaining=tuple(reversed(s.remaining))),0))
        # Same certain purchase: low-value early item gives less net development
        # than the identical purchase after future opportunities disappear.
        joint=np.zeros((32,4),dtype=int)
        early,_=evaluate(s,0,8,joint); late,_=evaluate(replace(s,remaining=()),0,8,joint)
        self.assertLess(early[0]['needs']['esteem'],late[0]['needs']['esteem'])
        compile_batch([make_context(s,0,profiles()[0],1,0)[0]])


if __name__=='__main__': unittest.main()
