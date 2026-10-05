import copy
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import MAX_ACTIONS, MAX_OUTCOMES, Policy, compile_batch
from reflex.laboratory import profiles
from reflex.reversi import (Position,PASS,flips,referee_flips,referee_legal,audit_move,move_id,parse_move,
    observe,forecasts,consequences,vector,compress_outcomes,require_capacity,opening,match,baseline,personality_probe)
from reflex.cross_games import core_hashes


class ReversiTests(unittest.TestCase):
    def test_official_initial_legal_moves_and_sample_c4_reply(self):
        position = Position()
        self.assertEqual({move_id(m) for m in position.legal()},{'c4','d3','e6','f5'})
        after = position.play(parse_move('c4'))
        self.assertEqual({move_id(m) for m in after.legal()},{'c3','e3','c5'})
        self.assertEqual(after.black.bit_count(),4)
        self.assertEqual(after.white.bit_count(),1)
        audit_move(position,parse_move('c4'),after)

    def test_all_eight_rays_must_flip_and_occupied_square_must_not_move(self):
        center = 27; black = 0; white = 0
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                if dx or dy:
                    white |= 1 << (center+dx+8*dy)
                    black |= 1 << (center+2*dx+16*dy)
        position = Position(black,white,1)
        self.assertEqual(flips(black,white,center).bit_count(),8)
        after = position.play(center)
        self.assertEqual(after.white,0)
        self.assertEqual(after.black.bit_count(),17)
        audit_move(position,center,after)
        with self.assertRaises(ValueError): after.play(center)

    def test_direction_shift_does_not_wrap_at_board_edges(self):
        position = Position(1 << 6,1 << 7,1)
        self.assertEqual(position.legal(),[])
        self.assertEqual(flips(position.black,position.white,8),0)

    def test_forced_pass_is_not_terminal_but_voluntary_pass_is_illegal(self):
        position = Position(1,2,-1)
        self.assertFalse(position.terminal())
        self.assertEqual(position.legal(),[])
        passed = position.play(PASS)
        audit_move(position,PASS,passed)
        self.assertEqual(passed.legal(),[2])
        finished = passed.play(2)
        self.assertTrue(finished.terminal())
        self.assertEqual((finished.black | finished.white).bit_count(),3)
        with self.assertRaises(ValueError): Position().play(PASS)
        with self.assertRaises(ValueError): finished.play(PASS)

    def test_fast_engine_matches_independent_referee_during_complete_games(self):
        for seed in range(3):
            position = Position()
            for ply in range(128):
                if position.terminal(): break
                self.assertEqual(position.legal(),referee_legal(position))
                own,rival = position.discs(position.side)
                for m in position.legal():
                    actual = [i for i in range(64) if flips(own,rival,m) & (1 << i)]
                    self.assertEqual(actual,referee_flips(position,m))
                m = baseline(position,'random',seed,ply)
                after = position.play(m); audit_move(position,m,after); position = after
            self.assertTrue(position.terminal())

    def test_capacity_overflow_is_reported_without_silent_move_pruning(self):
        require_capacity(list(range(MAX_ACTIONS)))
        with self.assertRaisesRegex(ValueError,'refusing to silently prune'):
            require_capacity(list(range(MAX_ACTIONS+1)))

    def test_observation_contains_exact_legal_actions_and_no_fabricated_needs(self):
        position,_ = opening(2)
        for p in profiles():
            c,_ = observe(position,p,2,'same-position',6,depth=2)
            self.assertEqual({parse_move(a['id']) for a in c['actions']},set(position.legal()))
            self.assertTrue(all(a['legal'] for a in c['actions']))
            for n in ('physiology','belonging','growth'):
                self.assertFalse(c['needs'][n]['enabled'])
                self.assertIsNone(c['needs'][n]['deficit'])
            compile_batch([c])

    def test_game_side_consequences_do_not_depend_on_profile_name_or_traits(self):
        position,_ = opening(1)
        rows = [observe(position,p,1,'same',6,depth=2)[0] for p in profiles()]
        self.assertTrue(all(c['actions'] == rows[0]['actions'] and c['facts'] == rows[0]['facts'] for c in rows))

    def test_one_ply_matches_actual_transition_and_two_ply_includes_all_replies(self):
        position,_ = opening(0)
        for m in position.legal():
            one,stats = forecasts(position,m,1)
            self.assertEqual(one,[consequences(position,position.play(m),position.side)])
            self.assertEqual(stats['nodes'],1)
            two,stats = forecasts(position,m,2)
            after = position.play(m)
            self.assertEqual(stats['replies'],len(after.legal()) or 1)
            self.assertLessEqual(len(two),MAX_OUTCOMES)
            self.assertAlmostEqual(sum(r['p'] for r in two),1)

    def test_quantization_preserves_mass_expectation_and_worst_branch(self):
        from reflex.examples import effect
        raw = [effect((i-10)/20,{'safety':(10-i)/20},p=(i+1)/210) for i in range(20)]
        short = compress_outcomes(raw)
        self.assertEqual(len(short),MAX_OUTCOMES)
        self.assertEqual(short[0],raw[0])
        self.assertAlmostEqual(sum(r['p'] for r in short),1)
        np.testing.assert_allclose(sum(r['p']*vector(r) for r in raw),sum(r['p']*vector(r) for r in short),atol=1e-12)

    def test_variable_board_candidates_batch_and_single_decisions_match(self):
        contexts = []
        for seed in range(3):
            position,_ = opening(seed,6+seed*4)
            for p in profiles(): contexts.append(observe(position,p,seed,'batch-check',0,depth=2)[0])
        policy = Policy(); b = compile_batch(contexts)
        self.assertEqual(policy.decide(b).records(b),[policy.choose(c) for c in contexts])
        rb = compile_batch(list(reversed(contexts)))
        self.assertEqual(policy.decide(b).records(b),list(reversed(policy.decide(rb).records(rb))))

    def test_complete_persona_match_preserves_profile_core_and_snapshot_choices(self):
        p = profiles()[0]; original = copy.deepcopy(p); hashes = core_hashes()
        game = match(p,'minimax2',0,1,2,True)
        self.assertEqual(p,original)
        self.assertEqual(core_hashes(),hashes)
        for row in game['traces']:
            if 'context' in row:
                choice = Policy().choose(row['context'])
                self.assertEqual(choice['action_id'],row['move'])
                self.assertEqual(choice['next_state'],row['next_state'])
                self.assertEqual(row['context']['personality'],game['traces'][0]['context']['personality'])
                self.assertNotIn('minimax',row['context']['scope']['episode'])

    def test_seed_replay_does_not_change_match_result_or_actions(self):
        p = profiles()[1]
        a,b = match(p,'greedy',1,-1,1),match(p,'greedy',1,-1,1)
        self.assertEqual(a['traces'],b['traces'])
        self.assertEqual((a['own_discs'],a['opponent_discs']),(b['own_discs'],b['opponent_discs']))

    def test_same_board_personality_probe_exposes_distinct_choices(self):
        diagnostic = personality_probe()
        self.assertGreater(diagnostic['different_choices'],0)
        self.assertTrue(all(set(r['choices']) == {p['id'] for p in profiles()} for r in diagnostic['rows']))


if __name__ == '__main__': unittest.main()
