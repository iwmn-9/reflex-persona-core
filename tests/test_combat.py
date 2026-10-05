import copy
from dataclasses import replace
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import Policy,compile_batch
from reflex.routes import choose_route
from reflex.laboratory import profiles
from reflex.combat import (Battle,Unit,legal,los,hit_probability,resolve,preview,capture_tick,
    victory,terminal,route_context,make_context,battle_record,battle_from_record,zone_distance,progress)
from reflex.combat_experiment import run,replay,diagnostic_probe


def relocate(w,changes):
    roster=list(w.units)
    for i,fields in changes.items(): roster[i]=replace(roster[i],**fields)
    return replace(w,units=tuple(roster))


class CombatTests(unittest.TestCase):
    def test_movement_shot_legality_and_wall_los(self):
        w=Battle.start('choke')
        w=relocate(w,{0:dict(x=3,y=1),3:dict(x=5,y=1)})
        self.assertNotIn('move:4:1',legal(w,0)); self.assertNotIn('shoot:3',legal(w,0))
        self.assertFalse(los(w,(3,1),(5,1))); self.assertEqual(hit_probability(w,0,3),0)
        self.assertTrue(los(w,(3,2),(5,2)))
        self.assertGreater(zone_distance(w,(3,0)),zone_distance(w,(3,2)))

    def test_cover_guard_and_range_reduce_chance(self):
        w=relocate(Battle.start('open'),{0:dict(x=3,y=2),3:dict(x=5,y=2)})
        p=hit_probability(w,0,3)
        covered=replace(w,covers=((5,2),))
        self.assertLess(hit_probability(covered,0,3),p)
        self.assertLess(hit_probability(relocate(covered,{3:dict(guard=True)}),0,3),hit_probability(covered,0,3))
        self.assertEqual(hit_probability(relocate(w,{3:dict(x=8)}),0,3),0)

    def test_elimination_progress_includes_access_without_claiming_kill(self):
        w=Battle.start(goal='eliminate'); moved=preview(w,0,'move:2:0')
        self.assertGreater(progress(moved,0,'eliminate'),progress(w,0,'eliminate'))
        self.assertEqual(victory(moved),())
        self.assertTrue(all(u.hp==9 for u in moved.units))

    def test_same_destination_fails_for_everyone(self):
        w=relocate(Battle.start(),{0:dict(x=2,y=1),1:dict(x=2,y=3)})
        choices={i:'guard' for i in range(6)}; choices.update({0:'move:2:2',1:'move:2:2'})
        after,audit=resolve(w,choices,5)
        self.assertEqual(after.units[0].pos,(2,1)); self.assertEqual(after.units[1].pos,(2,3))
        self.assertEqual(audit['collisions'],2)
        self.assertEqual(len({u.pos for u in after.units if u.hp>0}),6)

    def test_simultaneous_volley_allows_committed_mutual_kill(self):
        w=Battle.start(goal='eliminate')
        w=relocate(w,{i:dict(hp=0) for i in (1,2,4,5)})
        w=relocate(w,{0:dict(x=3,y=2,hp=3),3:dict(x=4,y=2,hp=3)})
        results=[resolve(w,{0:'shoot:3',3:'shoot:0'},s) for s in range(12)]
        after,audit=next((a,b) for a,b in results if a.units[0].hp==a.units[3].hp==0)
        self.assertEqual(len(audit['shots']),2); self.assertTrue(terminal(after)); self.assertEqual(victory(after),())
        self.assertEqual(after.units[0].ammo,2); self.assertEqual(after.units[3].ammo,2)

    def test_out_of_range_after_move_still_spends_committed_shot(self):
        w=relocate(Battle.start(),{0:dict(x=3,y=2),3:dict(x=7,y=2)})
        # Unit 4 originally at 7,2: move it so the scene has distinct cells.
        w=relocate(w,{4:dict(x=7,y=1)})
        choices={i:'guard' for i in range(6)}; choices.update({0:'shoot:3',3:'move:8:2'})
        after,audit=resolve(w,choices,1)
        self.assertEqual(after.units[0].ammo,2); self.assertEqual(after.units[3].hp,9)
        self.assertEqual(audit['shots'][0]['probability'],0)

    def test_finite_medkit_can_help_ally_or_self_without_refill(self):
        w=relocate(Battle.start(),{0:dict(x=2,y=2),1:dict(x=2,y=1,hp=3)})
        self.assertIn('heal:1',legal(w,0))
        after=preview(w,0,'heal:1'); self.assertEqual(after.units[1].hp,7); self.assertEqual(after.units[0].kit,0)
        self.assertNotIn('heal:1',legal(after,0))
        injured=relocate(w,{0:dict(hp=3)})
        self_healed=preview(injured,0,'heal:0')
        self.assertEqual((self_healed.units[0].hp,self_healed.units[0].kit),(7,0))
        with self.assertRaises(ValueError): preview(w,0,'heal:3')

    def test_and_or_are_distinct_and_require_survival(self):
        base=Battle.start()
        killed=relocate(base,{i:dict(hp=0) for i in (3,4,5)})
        self.assertEqual(victory(replace(killed,goal='either')),(0,))
        self.assertEqual(victory(replace(killed,goal='both')),())
        self.assertEqual(victory(replace(killed,goal='both',secured=(True,False))),(0,))
        captured=replace(base,secured=(True,False))
        self.assertEqual(victory(replace(captured,goal='secure')),(0,))
        self.assertEqual(victory(replace(captured,goal='both')),())
        dead=relocate(captured,{i:dict(hp=0) for i in (0,1,2)})
        self.assertNotIn(0,victory(replace(dead,goal='secure')))

    def test_capture_needs_three_consecutive_majority_ticks_then_latches(self):
        w=relocate(Battle.start(goal='both'),{0:dict(x=4,y=2)})
        w=capture_tick(w); w=capture_tick(w)
        self.assertFalse(w.secured[0]); self.assertEqual(w.hold[0],2)
        contested=relocate(w,{3:dict(x=4,y=1)})
        self.assertEqual(capture_tick(contested).hold,(0,0))
        secured=capture_tick(w); self.assertTrue(secured.secured[0])
        self.assertTrue(capture_tick(relocate(secured,{0:dict(x=3,y=2)})).secured[0])

    def test_completed_and_subgoal_is_unavailable_to_generic_selector(self):
        w=replace(Battle.start(goal='both'),secured=(True,False))
        c=route_context(w,0,profiles()[0],1)
        self.assertEqual(choose_route(c)[0].chosen,'eliminate')
        eliminated=relocate(Battle.start(goal='both'),{i:dict(hp=0) for i in (3,4,5)})
        self.assertEqual(choose_route(route_context(eliminated,0,profiles()[3],1))[0].chosen,'secure')

    def test_context_is_prestate_no_actual_intents_or_hit_seed(self):
        w=relocate(Battle.start(),{0:dict(x=3,y=2),3:dict(x=5,y=2)})
        c=make_context(w,0,profiles()[0],1,'eliminate'); before=copy.deepcopy(w)
        choices={i:'guard' for i in range(6)}; choices[3]='shoot:0'
        resolve(w,choices,7); resolve(w,choices,8)
        self.assertEqual(make_context(w,0,profiles()[0],1,'eliminate'),c)
        self.assertEqual(w,before); self.assertNotIn('world_seed',c); self.assertNotIn('choices',c['facts'])
        batch=compile_batch([c]); self.assertEqual(Policy().decide(batch).records(batch)[0],Policy().choose(c))
        self.assertEqual(battle_from_record(battle_record(w)),w)

    def test_complete_episodes_replay_and_keep_traits(self):
        p=profiles()[0]; old=copy.deepcopy(p)
        for goal in ('eliminate','both'):
            r=run(p,0,'cover',goal); self.assertGreater(replay(r),0)
        self.assertEqual(p,old)
        self.assertEqual(diagnostic_probe()['batch_single_identity'],36)


if __name__=='__main__': unittest.main()
