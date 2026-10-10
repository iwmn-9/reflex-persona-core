import copy
from dataclasses import replace
from itertools import permutations, product
import unittest
import numpy as np
from reflex.goofspiel import Position, make_context
from reflex.board_models import ThanksPosition
from reflex.laboratory import PROFILES
from reflex.strong_search import (PublicMemory, SearchBudget, _goof_returns,
    _goof_scenarios, _thanks_simulate, _thanks_persona_take, search, reasonable_persona)
from reflex.strong_search import _thanks_persona_take_reference
from reflex.tabletop_trials import competitive, thanks_observe, score
from reflex.goal_guard import choose_with_goal
from reflex.examples import context, action, effect


class StrongTableTests(unittest.TestCase):
    def test_shared_allocations_equal_exhaustive_independent_scalar_games(self):
        s=Position.start(4,3);plans=np.array(list(permutations((1,2,3))))
        scenarios=np.zeros((216,4,3),int)
        for j,rivals in enumerate(product(permutations((1,2,3)),repeat=3)):
            for actor,plan in enumerate(rivals,1):scenarios[j,actor]=plan
        scores,shares,_=_goof_returns(s,0,plans,scenarios)
        for i,own in enumerate(plans):
            for j,rival in enumerate(scenarios):
                world=s
                for r in range(3):world=world.play(tuple([int(own[r])]+list(map(int,rival[1:,r]))))
                self.assertEqual(tuple(scores[i,j]),world.scores)
                self.assertEqual(shares[i,j],world.share(0)[0])

    def test_scenario_plans_do_not_reuse_or_invent_cards(self):
        s=Position.start(4,13).play((1,4,8,13));m=PublicMemory('goofspiel',1)
        scenarios=_goof_scenarios(s,1,m,False,200,np.random.default_rng(7))
        for row in scenarios:
            for actor in (0,2,3):self.assertEqual(sorted(row[actor]),list(s.hands[actor]))

    def test_terminal_take_and_pass_follow_exact_chip_and_score_rules(self):
        s=replace(ThanksPosition.start(4,35),remaining=0,chips=(11,0,11,11))
        m=PublicMemory('no_thanks',0)
        scores,shares=_thanks_simulate(s,0,m,False,('TAKE','PASS'),(0,0),64,np.random.default_rng(8))
        self.assertTrue(np.all(scores[0]==(24,0,-11,-11)))
        self.assertTrue(np.all(scores[1]==(-10,34,-11,-11)))
        self.assertTrue(np.all(shares==0))

    def test_virtual_search_cannot_train_public_memory_and_is_reproducible(self):
        budget=SearchBudget(8,16,16,1)
        for game,s in (('goofspiel',Position.start(4,3)),('no_thanks',replace(ThanksPosition.start(4,30),remaining=2))):
            m=PublicMemory(game,0);before=copy.deepcopy(m.record());recent=copy.deepcopy(m.recent)
            a=search(game,s,0,m,True,budget,np.random.default_rng(9))
            b=search(game,s,0,m,True,budget,np.random.default_rng(9))
            self.assertEqual(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1]);np.testing.assert_array_equal(a[2],b[2])
            self.assertEqual(a[3],b[3]);self.assertEqual(m.record(),before);self.assertEqual(m.recent,recent)

    def test_encounter_scoping_and_independent_owners(self):
        s=Position.start(4,3);a=PublicMemory('goofspiel',0);b=PublicMemory('goofspiel',2)
        before=b.record();a.observe(s,1,'BID:3','encounter-0-round-0');a.observe(s,1,'BID:3','encounter-1-round-0')
        self.assertEqual(a.trackers[1].observations,2);self.assertEqual(b.record(),before)
        self.assertIsNone(a.observe(s,0,'BID:3','own'));self.assertEqual(a.trackers[0].observations,0)
        with self.assertRaises(ValueError):a.observe(s,1,'BID:3','encounter-1-round-0')

    def test_forced_actions_do_not_identify_a_behavior_type(self):
        s=replace(ThanksPosition.start(4,35),chips=(0,11,11,11));m=PublicMemory('no_thanks',1)
        before=m.weights(0).copy();r=m.observe(s,0,'TAKE','forced')
        self.assertTrue(r['forced']);self.assertEqual(m.trackers[0].observations,0)
        np.testing.assert_array_equal(before,m.weights(0));self.assertEqual(m.recent[0],[])

    def test_bound_recent_history_and_tracker_ids_across_long_encounters(self):
        s=Position.start(4,3);m=PublicMemory('goofspiel',0)
        for i in range(100):m.observe(s,1,'BID:3',f'encounter-{i}-round-0')
        self.assertEqual(len(m.recent[1]),16);self.assertEqual(len(m.trackers[1].ids),64)
        self.assertTrue(all(np.isfinite(m.trackers[1].logs)));self.assertEqual(m.trackers[1].observations,100)

    def test_goal_guard_keeps_identity_and_unresolved_choices(self):
        c=competitive(make_context(Position.start(4,3),0,PROFILES[1],'win_share',3,0,'guard'))
        saved=copy.deepcopy(c);names=('BID:1','BID:2','BID:3')
        shares=np.array([[0.]*32,[1.]*32,[.9]*32])
        d,st=reasonable_persona(c,names,shares)
        self.assertEqual(c,saved);self.assertEqual(st['rejected'],['BID:1'])
        self.assertIn(d['action_id'],('BID:2','BID:3'))

    def test_generic_goal_guard_never_promotes_illegal_or_known_failure_moves(self):
        c=context('different-game',[action('forbidden',effect(1),legal=False),
            action('failed',effect(.9),failure=True),action('viable',effect(.2))])
        d,st=choose_with_goal(c,('forbidden','failed','viable'),np.array([[1.]*8,[1.]*8,[0.]*8]))
        self.assertEqual(d['action_id'],'viable');self.assertEqual(st['allowed'],['viable'])
        self.assertEqual(d['next_state']['intent_action'],'viable')

    def test_generic_guard_rejects_missing_and_nonfinite_evidence(self):
        c=context('goal',[action('a',effect(0)),action('b',effect(1))])
        for names,values in ((('a',),[[0,0]]),(('a','b'),[[0,float('nan')],[1,1]]),
                             (('a','b'),[[0],[1]]),(('a','b'),[[0,0],[1,2]])):
            with self.assertRaises(ValueError):choose_with_goal(c,names,values)

    def test_vector_persona_hypotheses_match_actual_common_policy_immediate_effects(self):
        for card,pot,stock,remaining in ((35,2,5,23),(10,4,2,4),(3,0,0,0),(20,6,11,0)):
            for actor in range(4):
                s=replace(ThanksPosition.start(4,card,actor),pot=pot,remaining=remaining,
                    chips=tuple(stock if a==actor else 11 for a in range(4)),
                    cards=((9,11),(18,19),(4,5),(22,)))
                n=4;turn=np.full(n,actor);cards=np.zeros((n,4,37),bool)
                for a,h in enumerate(s.cards):cards[:,a,list(h)]=True
                points=np.tile([__import__('reflex.board_models',fromlist=['card_points']).card_points(h) for h in s.cards],(n,1))
                kinds=np.zeros((n,4),int);kinds[:,actor]=np.arange(6,10)
                state=np.zeros((n,4,3));state[:,:,:2]=-1
                rows,takes=_thanks_persona_take(np.arange(n),turn,np.full(n,card),np.full(n,pot),
                    np.tile(s.chips,(n,1)),points,cards,kinds,np.full(n,remaining),state)
                self.assertEqual(list(rows),list(range(4)))
                for p,take in zip(PROFILES,takes):
                    c,_=thanks_observe(s,p,0,0,'hypothesis',None)
                    self.assertEqual(bool(take),score(c)[0]['action_id']=='TAKE',(card,pot,stock,actor,p['id']))

    def test_frozen_prediction_removes_recent_and_conditional_learning(self):
        for game,s in (('goofspiel',Position.start(4,3)),('no_thanks',ThanksPosition.start(4,35))):
            actor=1 if game=='goofspiel' else 0;viewer=0 if game=='goofspiel' else 1
            m=PublicMemory(game,viewer);cold=PublicMemory(game,viewer)
            actual='BID:3' if game=='goofspiel' else 'TAKE'
            for i in range(8):m.observe(s,actor,actual,f'encounter-{i}')
            self.assertEqual(m.predict(s,actor,False),cold.predict(s,actor,False))
            self.assertNotEqual(m.predict(s,actor,True),cold.predict(s,actor,True))

    def test_cached_virtual_kernel_matches_common_policy_with_history_and_endgames(self):
        rng=np.random.default_rng(370);n=3000;turn=rng.integers(4,size=n);card=rng.integers(3,36,size=n)
        pot=rng.integers(30,size=n);chips=np.array([rng.multinomial(44-p,[.25]*4) for p in pot])
        cards=np.zeros((n,4,37),bool)
        for i in range(n):
            available=[c for c in range(3,36) if c!=card[i]]
            for c in rng.choice(available,10,replace=False):cards[i,int(rng.integers(4)),c]=True
        from reflex.board_models import card_points
        points=np.array([[card_points(tuple(np.flatnonzero(cards[i,a]))) for a in range(4)] for i in range(n)])
        kinds=rng.integers(6,10,size=(n,4));remaining=rng.integers(2,size=n)
        state=np.zeros((n,4,3));state[:,:,0]=rng.integers(-1,2,size=(n,4))
        state[:,:,1]=rng.choice([-1,1,3],size=(n,4));state[:,:,2]=rng.choice([0,.1,.125,.25,.375,.5,.625,.75,.875,.9],size=(n,4))
        other=state.copy();a=_thanks_persona_take(np.arange(n),turn,card,pot,chips,points,cards,kinds,remaining,state)
        b=_thanks_persona_take_reference(np.arange(n),turn,card,pot,chips,points,cards,kinds,remaining,other)
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1]);np.testing.assert_array_equal(state,other)


if __name__=='__main__':unittest.main()
