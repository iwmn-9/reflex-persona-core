import copy
import unittest
from unittest.mock import patch
import numpy as np
from reflex.board_models import ThanksPosition
from reflex.goofspiel import Position, observe
from reflex.goofspiel_beliefs import PublicBidBeliefs
from reflex.laboratory import PROFILES
from reflex.monte_carlo import RolloutBudget
from reflex.opponent_beliefs import HypothesisTracker
from reflex.tabletop_trials import competitive, choose, play, thanks_hypotheses, thanks_observe, ThanksForecastModel
from reflex.board_models import ThanksAdapter


class TabletopTrialsTests(unittest.TestCase):
    def test_competition_is_not_a_kindness_event_and_input_is_preserved(self):
        c,_=observe(Position.start(4,3),0,PROFILES[2],'win_share',12,0,'public')
        saved=copy.deepcopy(c);r=competitive(c)
        self.assertEqual(c,saved);self.assertEqual(c['values'],r['values'])
        for a,b in zip(c['actions'],r['actions']):
            for x,y in zip(a['outcomes'],b['outcomes']):
                self.assertEqual(x['objective'],y['objective'])
                self.assertEqual(y['values']['benevolence'],0)
                self.assertEqual(y['values']['universalism'],0)

    def test_counterless_player_forces_take_without_creating_informative_evidence(self):
        s=ThanksPosition.start(4,30);s=__import__('dataclasses').replace(s,chips=(0,11,11,11))
        models=thanks_hypotheses(s)
        self.assertTrue(all(row=={'TAKE':1.} for row in models.values()))
        tracker=HypothesisTracker(tuple(models));d=tracker.observe(models,'TAKE','public-1')
        self.assertTrue(d['forced']);self.assertEqual(tracker.observations,0)

    def test_virtual_future_does_not_train_real_public_memory(self):
        s=ThanksPosition.start(4,30);trackers=tuple(HypothesisTracker(('uniform','tactical','reserve','accept')) for _ in range(4))
        before=[t.snapshot().record() for t in trackers]
        budget=RolloutBudget(samples=2,min_samples=1,max_nodes=5000,max_steps=512,rollout_policy='random')
        _,stats=thanks_observe(s,PROFILES[0],8,0,'fiction',None,budget,tuple(t.snapshot() for t in trackers))
        self.assertTrue(stats['used']);self.assertEqual(before,[t.snapshot().record() for t in trackers])

    def test_unrevealed_card_sampling_uses_only_public_exclusions(self):
        s=ThanksPosition.start(4,30).play('TAKE')
        model=ThanksForecastModel(ThanksAdapter(),s,PROFILES[0],8,0,'deck')
        for seed in range(20):
            drawn=model.sample(s,np.random.default_rng(seed))
            self.assertNotIn(drawn.card,s.seen);self.assertEqual(drawn.remaining,s.remaining-1)
            self.assertEqual(drawn.cards,s.cards);self.assertEqual(drawn.chips,s.chips)

    def test_no_public_evidence_learned_path_matches_same_budget_baseline(self):
        s=Position.start(4,3);beliefs=PublicBidBeliefs(4,0)
        a=choose('goofspiel',s,0,PROFILES[0],4,0,'same',None,'mc',beliefs)
        b=choose('goofspiel',s,0,PROFILES[0],4,0,'same',None,'learned',beliefs)
        self.assertEqual(a[0],b[0]);self.assertEqual(a[1],b[1]);self.assertFalse(b[2]['model_evaluated'])

    def test_complete_matches_all_four_owners_and_rotated_roster_are_reproducible(self):
        for game in ('goofspiel','no_thanks'):
            a=play(game,5001,3,'reflex');b=play(game,5001,3,'reflex')
            self.assertEqual(a,b);r,trace,p=a
            self.assertTrue(r['finished']);self.assertEqual(r['roster'][r['seat']],'ego')
            self.assertEqual(len(r['roster']),4)
            self.assertEqual({d['actor'] for t in trace for d in t['decisions']},{0,1,2,3})


if __name__=='__main__':unittest.main()
