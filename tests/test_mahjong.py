import copy
import importlib.util
import unittest
import numpy as np
from reflex.core import Policy,compile_batch,digest
from reflex.laboratory import profiles

AVAILABLE=importlib.util.find_spec('riichienv') is not None


@unittest.skipUnless(AVAILABLE,'optional riichienv environment required')
class MahjongTests(unittest.TestCase):
    def setUp(self):
        from riichienv import RiichiEnv
        self.env=RiichiEnv(game_mode='4p-red-single',seed=2)
        self.obs=self.env.reset()[0]

    def test_known_hand_shanten_and_copy_normalization(self):
        from riichienv import parse_hand
        from reflex.mahjong_adapter import shanten
        for text,expected in [('123m456p789s11z',-1),('19m19p19s1234567z',0),('112233m445566p7s',0)]:
            hand,_=parse_hand(text);self.assertEqual(shanten(hand),expected)
        a,_=parse_hand('112233m445566p7s');b=[];used={}
        for t in a:
            k=t//4;i=used.get(k,0);b.append(k*4+3-i);used[k]=i+1
        self.assertEqual(shanten(a),shanten(b))

    def test_public_pool_uses_no_other_hand_or_wall(self):
        from reflex.mahjong_adapter import PublicView
        v=PublicView.from_observation(self.obs)
        self.assertEqual(len(v.unseen()),121)
        self.assertFalse(set(v.hand)&set(v.unseen()))
        self.assertFalse(set(v.dora)&set(v.unseen()))
        self.assertFalse(hasattr(v,'wall'));self.assertFalse(hasattr(v,'hands'))

    def test_duplicate_public_physical_tiles_rejected(self):
        from dataclasses import replace
        from reflex.mahjong_adapter import PublicView
        v=PublicView.from_observation(self.obs)
        with self.assertRaises(ValueError):replace(v,dora=(v.hand[0],)).unseen()

    def test_real_hidden_state_changes_cannot_change_decision(self):
        from reflex.mahjong_adapter import choose
        a,c,d,s=choose(self.obs,profiles()[0],2,0,'hidden-test',lookahead_samples=8)
        other=self.env.clone();h=other.hands;h[1],h[2]=h[2],h[1];other.hands=h
        other.wall=list(reversed(other.wall))
        a2,c2,d2,s2=choose(other.get_observation(0),profiles()[0],2,0,'hidden-test',lookahead_samples=8)
        self.assertEqual(digest(c),digest(c2));self.assertEqual(d,d2);self.assertEqual(s,s2)

    def test_all_engine_actions_logged_and_closed_protocol_explicit(self):
        from reflex.mahjong_adapter import action_id,candidates
        current={0:self.obs};found=False
        for tick in range(120):
            if self.env.done():break
            for o in current.values():
                v,rows=candidates(o)
                self.assertEqual({action_id(a) for a in o.legal_actions()},{r['id'] for r in rows})
                discards=[r for r in rows if r['type']=='Discard']
                if discards:
                    best=min(r['features']['shanten'] for r in discards)
                    kept=[r['features']['progress'] for r in discards if r['features']['shanten']==best]
                    lost=[r['features']['progress'] for r in discards if r['features']['shanten']>best]
                    if lost:self.assertLess(max(lost),min(kept))
                if any(not r['supported'] for r in rows):found=True
            if self.env.done():break
            from reflex.mahjong_adapter import reference
            current=self.env.step({p:reference(o,2,tick) for p,o in current.items()})
        self.assertTrue(found)

    def test_profile_and_unsupported_needs_preserved(self):
        from reflex.mahjong_adapter import choose
        p=profiles()[1];original=copy.deepcopy(p)
        a,c,d,s=choose(self.obs,p,2,0,'profile')
        self.assertEqual(p,original);self.assertEqual(c['values']['security'],p['values']['security'])
        self.assertTrue(all(not n['supported'] and not n['enabled'] and n['deficit'] is None for n in c['needs'].values()))
        a2,c2,d2,s2=choose(self.obs,p,2,1,'profile',d['next_state'])
        self.assertEqual(c['personality'],c2['personality']);self.assertEqual(c['values'],c2['values'])

    def test_genbutsu_is_type_specific_not_calibrated_probability(self):
        from dataclasses import replace
        from reflex.mahjong_adapter import PublicView
        v=PublicView.from_observation(self.obs)
        v=replace(v,riichi=(False,True,True,False),discards=((),(0,),(4,),()))
        self.assertEqual(v.exposure(3),.5);self.assertEqual(v.exposure(5),.5)
        self.assertEqual(v.exposure(10),1.)
        self.assertEqual(replace(v,riichi=(False,False,False,False)).exposure(10),0.)

    def test_equal_progress_avoids_unnecessary_public_exposure(self):
        from reflex.mahjong_adapter import PublicView,candidates,choose
        v,rows=candidates(self.obs);ds=[r for r in rows if r['type']=='Discard']
        best=max(r['features']['progress'] for r in ds)
        equal=[r for r in ds if r['features']['progress']==best]
        self.assertGreater(len(equal),1)
        safe=next(t for t in v.unseen() if t//4==equal[-1]['tile']//4)
        public=self.obs.to_dict();public['riichi_declared'][1]=True;public['discards'][1]=[safe]
        outer=self.obs
        class Fixture:
            def to_dict(self):return copy.deepcopy(public)
            def legal_actions(self):return outer.legal_actions()
        for p in profiles():
            a,c,d,s=choose(Fixture(),p,2,0,'no-free-risk')
            self.assertEqual(s['selected']['exposure'],0.)

    def test_same_state_personality_safety_tradeoff(self):
        from reflex.mahjong_adapter import candidates,choose,PublicView
        v,rows=candidates(self.obs);ds=[r for r in rows if r['type']=='Discard']
        minimum=min(r['features']['shanten'] for r in ds)
        bad=next(r for r in ds if r['features']['shanten']>minimum)
        public=self.obs.to_dict();safe=next(t for t in v.unseen() if t//4==bad['tile']//4)
        public['riichi_declared'][1]=True;public['discards'][1]=[safe]
        # Controlled public-information fixture, not claimed a reached match state.
        outer=self.obs
        class Fixture:
            def to_dict(self):return copy.deepcopy(public)
            def legal_actions(self):return outer.legal_actions()
        a,c,d,s=choose(Fixture(),profiles()[0],2,0,'tradeoff')
        a2,c2,d2,s2=choose(Fixture(),profiles()[1],2,0,'tradeoff')
        self.assertEqual(s['selected_shanten'],minimum)
        self.assertEqual(s2['selected']['exposure'],0.)
        self.assertGreater(s2['selected_shanten'],minimum)

    def test_minimum_hand_progress_without_declared_threat(self):
        from reflex.mahjong_adapter import choose
        for p in (profiles()[0],profiles()[1]):
            a,c,d,s=choose(self.obs,p,2,0,'progress')
            self.assertEqual(s['selected_shanten'],min(r['features']['shanten'] for r in s['features'] if r['type']=='Discard'))

    def test_lookahead_is_bounded_paired_and_replayable(self):
        from reflex.mahjong_adapter import choose,candidates
        with self.assertRaises(ValueError):candidates(self.obs,65)
        a,c,d,s=choose(self.obs,profiles()[0],2,0,'future',lookahead_samples=16)
        a2,c2,d2,s2=choose(self.obs,profiles()[0],2,0,'future',lookahead_samples=16)
        self.assertEqual(c,c2);self.assertEqual(d,d2);self.assertEqual(s,s2)
        for r in s['features']:
            self.assertLessEqual(len(r['outcomes']),8)
            self.assertAlmostEqual(sum(o['p'] for o in r['outcomes']),1.)
            self.assertTrue(all(-1<=o['objective']<=1 for o in r['outcomes']))

    def test_next_draw_completion_is_better_than_only_tenpai(self):
        from riichienv import parse_hand
        from reflex.mahjong_adapter import best_next_progress
        hand,_=parse_hand('123m12m456p789s11z')
        win=best_next_progress(hand,9);miss=best_next_progress(hand,120)
        self.assertEqual(win,1.);self.assertLess(miss,win);self.assertAlmostEqual(miss,.9)

    def test_same_state_batch_matches_individual_choices_and_memories(self):
        from reflex.mahjong_adapter import choose
        rows=[choose(self.obs,p,2,0,'batch') for p in profiles()]
        b=compile_batch([c for a,c,d,s in rows]);ds=Policy().decide(b,stochastic=False).records(b)
        self.assertEqual([d['action_id'] for d in ds],[d['action_id'] for a,c,d,s in rows])
        self.assertEqual([d['next_state'] for d in ds],[d['next_state'] for a,c,d,s in rows])

    def test_full_kyoku_finishes_and_real_wins_are_accepted(self):
        from reflex.mahjong_experiment import play
        r=play(2,0,'core_own_draw','efficiency')
        self.assertTrue(r['finished']);self.assertEqual(sum(r['scores'])+r['riichi_sticks']*1000,100000)
        for t in r['trace']:
            for d in t['decisions']:
                self.assertIn(d['action'],d['engine_legal'])
                if any(a.startswith(('Ron:','Tsumo:')) for a in d['engine_legal']):
                    self.assertTrue(d['action'].startswith(('Ron:','Tsumo:')))


if __name__=='__main__':unittest.main()
