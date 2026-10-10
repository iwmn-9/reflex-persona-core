import copy
import json
from dataclasses import replace
from unittest.mock import patch
import unittest
from reflex.strong_table import PlaybackStart, play, decide
from reflex.strong_search import PublicMemory, SearchBudget, random_stream
from reflex.goofspiel import Position
from reflex.board_models import ThanksPosition
from reflex.laboratory import PROFILES
from tools.intervene_goal_progress import position
import numpy as np


def plain(value):return json.loads(json.dumps(value))


class PolicyTeacherTests(unittest.TestCase):
    def test_resuming_actual_root_recreates_entire_remaining_match_and_keeps_inputs(self):
        budget=SearchBudget(8,16,16,1);seed=7055;bench=seed%4
        roster={a:p for a,p in zip([a for a in range(4) if a!=bench],PROFILES[:3])}
        for game in ('goofspiel','no_thanks'):
            original,rows=play(game,seed,0,bench,roster,'adaptive',[PublicMemory(game,a) for a in range(4)],strong=budget,persona=budget)
            index=next(i for i,t in enumerate(rows) if (t['before']['round']==10 if game=='goofspiel' else
                       t['before']['remaining']<=3 and t['before']['turn']!=bench and t['before']['chips'][t['before']['turn']]>0))
            memories=[PublicMemory(game,a) for a in range(4)];states=[None]*4
            for t in rows[:index]:
                s=position(game,t['before']);actors=range(4) if game=='goofspiel' else (s.turn,)
                for a,d in t['decisions'].items():
                    if int(a)!=bench:states[int(a)]=d['next_state']
                for observer in range(4):
                    for actor in actors:
                        if observer!=actor:memories[observer].observe(s,actor,t['moves'][actor],f'encounter-0-tick-{t["tick"]}-actor-{actor}')
            t=rows[index];s=position(game,t['before']);actor=min(roster) if game=='goofspiel' else s.turn
            if game=='goofspiel':deck=()
            else:
                cards=random_stream(seed,game,0,0,0,'world').permutation(np.arange(3,36))[:24]
                deck=tuple(int(c) for c in cards if c not in s.seen)
            snapshot=PlaybackStart(s,deck,tuple(states),t['tick']);saved=copy.deepcopy(snapshot.owner_states)
            prior=plain([m.record() for m in memories])
            resumed,_=play(game,seed,0,bench,roster,'adaptive',copy.deepcopy(memories),strong=budget,persona=budget,
                           start=snapshot,forced_root=(actor,t['moves'][actor]))
            for key in ('scores','credits','final','beliefs'):self.assertEqual(plain(resumed[key]),plain(original[key]))
            self.assertEqual(snapshot.owner_states,saved);self.assertEqual(plain([m.record() for m in memories]),prior)

    def test_sealed_root_is_not_revealed_to_peers_before_the_joint_move(self):
        budget=SearchBudget(8,16,16,1);s=Position.start(4,13);memories=[PublicMemory('goofspiel',a) for a in range(4)]
        roster={a:p for a,p in zip((1,2,3),PROFILES)};observed=[]
        def spy(*args,**kwargs):
            if args[6]==0:
                observed.append((args[2],args[1].round,args[1].hands,tuple(t.observations for t in args[7].trackers)))
            return decide(*args,**kwargs)
        with patch('reflex.strong_table.decide',side_effect=spy):
            play('goofspiel',2,0,0,roster,'adaptive',memories,strong=budget,persona=budget,
                 start=PlaybackStart(s,(),(None,)*4,0),forced_root=(1,'BID:13'))
        self.assertEqual({r[0] for r in observed},{1,2,3})
        self.assertTrue(all(r[1]==0 and r[2]==s.hands and r[3]==(0,0,0,0) for r in observed))

    def test_invalid_world_or_forced_action_is_rejected(self):
        s=replace(ThanksPosition.start(4,35),remaining=0,chips=(0,11,11,22))
        with self.assertRaises(ValueError):PlaybackStart(s,(35,),(None,)*4,0)
        snapshot=PlaybackStart(s,(),(None,)*4,0);memories=[PublicMemory('no_thanks',a) for a in range(4)]
        with self.assertRaises(ValueError):
            play('no_thanks',0,0,1,{0:PROFILES[0],2:PROFILES[1],3:PROFILES[2]},'adaptive',memories,
                 start=snapshot,forced_root=(0,'PASS'))


if __name__=='__main__':unittest.main()
