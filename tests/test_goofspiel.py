import copy
from itertools import product
import unittest
import numpy as np
from reflex.core import Policy, compile_batch, digest
from reflex.laboratory import profiles
from reflex.monte_carlo import RolloutBudget
from reflex.goofspiel import (Position, Pending, RolloutModel, referee, immediate_outcomes,
                              observe, exact_two_rounds)
from reflex.goofspiel_experiment import play, exact_probe, population_probe
from reflex.board_planning import core_audit


class GoofspielTests(unittest.TestCase):
    def test_unique_high_ties_discard_and_all_cards_consumed(self):
        s=Position.start(3,3); bids=(3,2,1); a=s.play(bids); referee(s,bids,a)
        self.assertEqual(a.scores,(1,0,0)); self.assertEqual(a.hands,((1,2),(1,3),(2,3)))
        bids=(1,3,3); b=a.play(bids); referee(a,bids,b)
        self.assertEqual(b.scores,a.scores); self.assertEqual(b.discarded,2)
        end=b.play((2,1,2)); referee(b,(2,1,2),end)
        self.assertTrue(end.terminal); self.assertEqual(end.discarded,5)
        self.assertEqual(end.share(0),(1.,[0]))

    def test_invalid_setup_bid_reuse_and_terminal(self):
        for args in ((1,13),(11,13),(4,1),(4,14),(True,13)):
            with self.assertRaises(ValueError): Position.start(*args)
        s=Position.start(3,2)
        for bids in ((1,2),(1,2,3),(True,1,1)):
            with self.assertRaises(ValueError): s.play(bids)
        s=s.play((1,1,1))
        with self.assertRaises(ValueError): s.play((1,2,2))
        s=s.play((2,2,2)); self.assertAlmostEqual(s.share(0)[0],1/3)
        with self.assertRaises(ValueError): s.play((1,1,1))

    def test_analytic_distribution_against_independent_enumeration(self):
        s=Position.start(4,3)
        for own in s.hands[0]:
            points=[]
            for rest in product(*s.hands[1:]):
                bids=(own,)+rest; high=max(bids)
                points.append(s.prizes[0] if own==high and bids.count(high)==1 else 0)
            rows=immediate_outcomes(s,0,own,'score')
            self.assertAlmostEqual(sum(r['p'] for r in rows),1)
            self.assertAlmostEqual(sum(r['p']*r['objective'] for r in rows),np.mean(points)/sum(s.prizes))

    def test_exact_values_against_independent_referee(self):
        s=Position(((1,3),(2,3),(1,2)),(4,2,1),(1,2,3),1,0)
        exact=exact_two_rounds(s,0)
        for own in (1,3):
            scores=[]; shares=[]
            for other in product(*s.hands[1:]):
                tally=list(s.scores); first=(own,)+other
                last=tuple(next(c for c in h if c!=b) for h,b in zip(s.hands,first))
                for bids,prize in ((first,2),(last,3)):
                    if bids.count(max(bids))==1: tally[bids.index(max(bids))]+=prize
                winners=[i for i,p in enumerate(tally) if p==max(tally)]
                scores.append(tally[0]); shares.append(1/len(winners) if 0 in winners else 0)
            self.assertEqual(exact[f'BID:{own}']['mean_score'],np.mean(scores))
            self.assertEqual(exact[f'BID:{own}']['win_share'],np.mean(shares))

    def test_fixed_root_is_not_observed_by_other_simultaneous_bidders(self):
        s=Position.start(6,3)
        for policy in ('random','tactical','persona'):
            model=RolloutModel(s,2,profiles()[0],'score',policy,42,0,'hidden')
            model.begin_trial(); a=model.joint_bids(Pending(s,1),np.random.default_rng(17))
            model.begin_trial(); b=model.joint_bids(Pending(s,3),np.random.default_rng(17))
            self.assertEqual(a[:2]+a[3:],b[:2]+b[3:]); self.assertEqual(a[2],1); self.assertEqual(b[2],3)

    def test_public_context_and_persona_copies_are_isolated(self):
        s=Position.start(4); profile=profiles()[0]; before=copy.deepcopy(profile)
        model=RolloutModel(s,0,profile,'win_share','persona',2,0,'x')
        model.profile['values']['power']=.1
        self.assertEqual(profile,before)
        c=observe(s,0,profile,'score',2,0,'x')[0]; before=digest(c)
        compile_batch([c]); Policy().choose(c)
        self.assertEqual(digest(c),before)
        self.assertEqual(c['facts']['public_hand_0'],str(list(s.hands[0])))
        self.assertNotIn('joint_bids',c['facts'])
        self.assertFalse(c['needs']['growth']['enabled'])

    def test_replay_and_budget_fallback_preserve_reflex(self):
        s=Position.start(3,3); p=profiles()[0]
        budget=RolloutBudget(samples=8,rollout_policy='random',max_steps=3)
        a,stats=observe(s,0,p,'score',2,0,'x',budget=budget)
        b,other=observe(s,0,p,'score',2,0,'x',budget=budget)
        self.assertEqual(a,b); self.assertEqual(stats,other); self.assertEqual(stats['completed_samples'],8)
        base=observe(s,0,p,'score',2,0,'x')[0]
        fallback,stats=observe(s,0,p,'score',2,0,'x',budget=RolloutBudget(max_nodes=0))
        self.assertEqual(base,fallback); self.assertFalse(stats['used'])

    def test_terminal_goal_and_score_rewards_are_distinct(self):
        s=Position(((2,),(1,),(1,)),(0,10,0),(1,2),1,0)
        rewards=[]
        for goal in ('score','win_share'):
            model=RolloutModel(s,0,profiles()[0],goal,'random',2,0,'x'); model.begin_trial()
            final=model.sample(Pending(s,2),np.random.default_rng(2)); result=model.evaluate(final)
            rewards.append(result.outcome['objective']); self.assertEqual(result.game_score,2)
        self.assertEqual(rewards,[1.,-1.])

    def test_complete_multiplayer_both_goals_and_opponents(self):
        for players in (3,4,6):
            for goal in ('score','win_share'):
                for opponent in ('random','reserve'):
                    r=play(players,1,0,goal,'mc8',opponent,cards=3)
                    self.assertTrue(r['finished']); self.assertEqual(len(r['trace']),3)
                    self.assertTrue(all(t['samples']==8 for t in r['trace']))

    def test_exact_material_has_no_preferred_labels(self):
        targets,summary=exact_probe(2)
        self.assertEqual(len(targets),12)
        self.assertTrue(all('preferred' not in r and 'gold' not in r for r in targets))
        self.assertTrue(all(r['enumerated_joint_choices']==2**(len(r['observation']['hands'])-1) for r in targets))
        self.assertEqual(len(summary['summary']),2)

    def test_simultaneous_personality_population_batch_matches(self):
        r=population_probe(); self.assertEqual(r['batch_individual_mismatches'],0)
        self.assertTrue(all(g['finished'] for g in r['live_runs']))

    def test_shared_core_unchanged(self):
        result=core_audit()
        if result['unchanged_verified'] is not None: self.assertTrue(result['unchanged_verified'])

    def test_supported_player_boundaries_and_persona_continuation(self):
        for players in (2,10):
            s=Position.start(players,3); p=profiles()[0]
            c,stats=observe(s,0,p,'score',7,0,'boundary',budget=RolloutBudget(samples=4,min_samples=4,rollout_policy='persona',max_steps=3))
            self.assertEqual(stats['completed_samples'],4)
            self.assertIn(Policy().choose(c)['action_id'],[f'BID:{b}' for b in s.hands[0]])
            self.assertEqual(s.scores,(0,)*players)


if __name__=='__main__': unittest.main()
