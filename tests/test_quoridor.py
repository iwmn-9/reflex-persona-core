import copy
from dataclasses import replace
import unittest
from reflex.core import Policy, compile_batch, MAX_ACTIONS
from reflex.examples import action, context, effect
from reflex.laboratory import profiles
from reflex.quoridor import (Position, graph, point, referee_neighbors, referee_moves,
                            referee_distances, audit, snapshot, consequences, play_game,
                            personality_probe, core_change_audit, fits, referee_wall_valid)


class QuoridorTests(unittest.TestCase):
    def test_initial_rules_counts_and_stocks(self):
        for n in (2,4):
            p=Position.start(n)
            self.assertEqual(len(p.pawns),n)
            self.assertEqual(p.stock,(20//n,)*n)
            self.assertEqual(p.distances(),(8,)*n)
            self.assertEqual(len(p.legal()),131)
        with self.assertRaises(ValueError): Position.start(3)

    def test_no_cross_overlap_or_edge_wrap(self):
        for w in (('H',3,3),('H',2,3),('H',4,3),('V',3,3)):
            self.assertFalse(fits(w,(('H',3,3),)))
        self.assertTrue(fits(('H',5,3),(('H',3,3),)))
        self.assertTrue(fits(('V',4,3),(('H',3,3),)))
        self.assertFalse(fits(('H',8,0),()))
        for walls in ((),(('H',3,3),),tuple((k,x,y) for k,x,y in [('H',0,0),('V',2,2),('H',4,4)])):
            for cell,neighbors in enumerate(graph(walls)):
                self.assertEqual(sorted(point(i) for i in neighbors),sorted(referee_neighbors(point(cell),walls)))

    def test_jump_diagonal_and_third_pawn_never_chain(self):
        p=Position(((4,3),(4,4),(4,5),(3,4)),(0,1,2,3),(5,)*4)
        self.assertIn('M:5:4',p.moves())
        self.assertNotIn('M:4:6',p.moves())
        self.assertNotIn('M:3:4',p.moves())
        self.assertEqual(sorted(p.moves()),list(referee_moves(p)))
        p=Position(((4,3),(4,4)),(0,2),(10,10))
        self.assertIn('M:4:5',p.moves())
        self.assertNotIn('M:5:4',p.moves())
        p=replace(p,walls=(('H',4,4),),stock=(9,10))
        self.assertIn('M:5:4',p.moves()); self.assertIn('M:3:4',p.moves())
        p=Position(((4,7),(4,8)),(0,1),(10,10))
        self.assertIn('M:5:8',p.moves()); self.assertIn('M:3:8',p.moves())

    def test_all_players_must_keep_a_route(self):
        p=replace(Position.start(4),walls=tuple(('H',x,0) for x in (0,2,4,6)),stock=(4,)*4)
        self.assertNotIn('V:7:0',p.wall_candidates())
        maps=referee_distances(p)
        self.assertEqual(maps,p.distances())
        illegal=replace(p,walls=tuple(sorted(p.walls+(('V',7,0),))),stock=(3,4,4,4))
        self.assertEqual(referee_distances(illegal)[0],999)
        with self.assertRaises(ValueError): p.play('V:7:0')

    def test_full_wall_candidates_match_independent_referee(self):
        p=Position.start(4)
        for _ in range(4):
            expected=set()
            for kind in ('H','V'):
                for x in range(8):
                    for y in range(8):
                        if referee_wall_valid((kind,x,y),p): expected.add(f'{kind}:{x}:{y}')
            self.assertEqual(expected,set(p.wall_candidates()))
            p=p.play(sorted(expected)[len(expected)//2])

    def test_stock_conservation_empty_supply_and_terminal(self):
        p=Position.start(2)
        for _ in range(20):
            name=p.wall_candidates()[0]; after=p.play(name); audit(p,name,after); p=after
        self.assertEqual(p.stock,(0,0)); self.assertEqual(p.legal(),p.moves())
        with self.assertRaises(ValueError): p.play('H:7:7')
        terminal=Position(((4,8),(4,4)),(0,2),(10,10))
        self.assertEqual(terminal.winner(),0); self.assertFalse(terminal.legal())

    def test_full_candidate_capacity_and_unsupported_needs(self):
        c,_=snapshot(Position.start(4),profiles()[0],3,0,'initial')
        b=compile_batch([c]); self.assertEqual(b.legal.shape,(1,131))
        self.assertGreater(len(c['actions']),32); self.assertGreaterEqual(MAX_ACTIONS,131)
        for k in ('physiology','belonging','growth'):
            self.assertEqual(c['needs'][k],dict(supported=False,enabled=False,deficit=None))
        self.assertIsNone(c['opponent'])
        self.assertNotIn('profile',c['facts'])

    def test_generic_core_arbitrary_target_counts_and_batch_order(self):
        cases=[]
        for n in (3,6):
            acts=[action(f'choose-{i}',effect(i/n)) for i in range(n)]
            for i,a in enumerate(acts): a['target']=f'participant-{i}'
            cases.append(context(f'abstract-{n}-participants',acts))
        policy=Policy(); batch=compile_batch(cases); together=policy.decide(batch).records(batch)
        single=[policy.choose(c) for c in cases]
        self.assertEqual(together,single)
        reverse=compile_batch(cases[::-1]); self.assertEqual(policy.decide(reverse).records(reverse)[::-1],single)

    def test_two_four_player_contexts_mixed_and_separate_match(self):
        cases=[]
        for n in (2,4):
            for seat in range(n):
                c,_=snapshot(Position.start(n,seat),profiles()[seat],1,0,f'players-{n}')
                cases.append(c)
        p=Policy(); batch=compile_batch(cases)
        self.assertEqual(p.decide(batch).records(batch),[p.choose(c) for c in cases])
        reverse=compile_batch(cases[::-1])
        self.assertEqual(p.decide(reverse).records(reverse)[::-1],p.decide(batch).records(batch))

    def test_obvious_next_turn_finish_is_not_ignored_by_achievement(self):
        p=Position(((4,2),(1,4),(4,6),(2,4)),(0,1,2,3),(5,)*4)
        c,impacts=snapshot(p,profiles()[0],0,0,'immediate-threat')
        self.assertTrue(any(i['next_actor_can_finish'] for i in impacts.values()))
        self.assertTrue(any(not i['next_actor_can_finish'] for i in impacts.values()))
        d=Policy().choose(c,False)
        self.assertFalse(impacts[d['action_id']]['next_actor_can_finish'])

    def test_actual_effects_are_personality_independent(self):
        p=Position.start(4); rows=[snapshot(p,profile,7,0,'identical')[0] for profile in profiles()]
        for c in rows[1:]: self.assertEqual(c['actions'],rows[0]['actions'])
        _,impacts=snapshot(p,profiles()[0],7,0,'effect')
        self.assertTrue(any(len(i['affected'])>1 for i in impacts.values()))
        for name,i in impacts.items():
            after=p.play(name)
            self.assertEqual(i['distance_after'],list(referee_distances(after)))
            self.assertEqual(sum(after.stock)+len(after.walls),20)

    def test_near_finish_does_not_demand_interference(self):
        for row in personality_probe():
            if row['position']=='own_clear_lead':
                self.assertEqual(row['action'],'M:4:8')
                self.assertFalse(row['impact']['wall'])
            if row['position']=='own_advantage':
                self.assertGreater(row['impact']['self_progress'],0)
                self.assertFalse(row['impact']['wall'])

    def test_replay_persona_and_state_isolation(self):
        roster=profiles()
        old=copy.deepcopy(roster)
        a=play_game(roster,2,max_turns=5); b=play_game(roster,2,max_turns=5)
        a.pop('decision_ms'); b.pop('decision_ms')
        self.assertEqual(a,b); self.assertEqual(roster,old)
        self.assertTrue(a['truncated']); self.assertIsNone(a['winner'])
        self.assertEqual(len(a['trace']),5)

    def test_teacher_snapshot_has_no_future_or_reference_label(self):
        sink=[]; play_game(profiles(),1,max_turns=2,teacher_sink=sink)
        self.assertEqual(len(sink),2)
        for row in sink:
            self.assertEqual(row['quality'],'awaiting_generation')
            self.assertNotIn('preferred',row); self.assertNotIn('reference_action',row)
            self.assertNotIn('winner',row['context']['facts'])
            self.assertEqual(row['context']['tick'],int(row['case_id'].split('-')[-1]))

    def test_shared_logic_capacity_only_audit(self):
        result=core_change_audit()
        if result['capacity_only_verified'] is not None: self.assertTrue(result['capacity_only_verified'])


if __name__=='__main__': unittest.main()
