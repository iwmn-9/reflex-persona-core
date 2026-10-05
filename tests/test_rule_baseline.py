import unittest
from unittest.mock import patch
from dataclasses import replace
import numpy as np
from reflex.core import Policy,compile_batch
from reflex.board_models import ConnectPosition,ConnectAdapter,ThanksPosition,ThanksAdapter
from reflex.goofspiel import Position,Pending
from reflex.laboratory import profiles
from reflex.monte_carlo import RolloutBudget
from reflex.rule_baseline import (Rewards,ConnectRules,GoofRules,ThanksRules,reward_effect,decide,observe)
from reflex.rule_baseline_experiment import play,setup,personality_probe
from reflex.cross_games import core_hashes

SMALL=RolloutBudget(samples=4,min_samples=4,max_nodes=8000,max_steps=512,rollout_policy='random')


class RuleBaselineTests(unittest.TestCase):
    def test_terminal_rewards_are_rule_facts_and_bounded(self):
        c=ConnectPosition()
        for col in (0,6,1,6,2,5,3):c=c.play(f'DROP:{col}')
        r=ConnectRules().rewards(c)
        self.assertEqual(r.credits,(1.,0.));self.assertEqual(r.returns,(1.,-1.))
        g=Position.start(3,2).play((2,1,1)).play((1,2,2))
        r=GoofRules(3).rewards(Pending(g))
        self.assertEqual(r.credits,(1.,0.,0.));self.assertEqual(r.scores,(1,0,0))
        t=ThanksPosition(cards=((3,4),(8,),()),chips=(2,3,28),turn=0,card=None,remaining=0)
        r=ThanksRules(3).rewards(t)
        self.assertEqual(r.scores,(1,5,-28));self.assertEqual(r.credits,(0.,0.,1.))
        self.assertTrue(all(-1<=x<=1 for x in r.returns))
        with self.assertRaises(ValueError):Rewards((1,1),(0,0),(0,0))

    def test_reward_mapping_shared_and_explicitly_limited(self):
        r=Rewards((0.,1.,0.),(-.5,.8,-.3),(2,4,1))
        bare=reward_effect(r,0,'rules_persona');rich=reward_effect(r,0,'reward_persona')
        self.assertEqual(bare['objective'],-1.);self.assertEqual(bare['values'],{'achievement':-1.})
        self.assertEqual(rich['values']['achievement'],-.5)
        self.assertEqual(rich['values']['security'],-.5)
        self.assertAlmostEqual(rich['values']['benevolence'],.25)
        self.assertEqual(rich['needs'],{});self.assertEqual(rich['style'],{})

    def test_no_positional_or_game_personality_heuristics_called(self):
        with (patch.object(ConnectAdapter,'consequence',side_effect=AssertionError('forbidden')),
             patch.object(ConnectAdapter,'context',side_effect=AssertionError('forbidden')),
             patch.object(ThanksAdapter,'consequence',side_effect=AssertionError('forbidden')),
             patch('reflex.goofspiel.consequence',side_effect=AssertionError('forbidden'))):
            for game,players in (('connect_four',2),('goofspiel',3),('no_thanks_basic',3)):
                rules,state,_=setup(game,players,0)
                c,d,s=decide(rules,state,0,profiles()[0],'reward_persona',SMALL,0,0,'isolated')
                self.assertTrue(s['used']);self.assertIn(d['action_id'],rules.legal(state,0))
                self.assertEqual(s['intermediate_evaluation'],'none')
                self.assertTrue(all(not n['supported'] for n in c['needs'].values()))

    def test_no_heuristic_fallback_on_budget_exhaustion(self):
        c,d,s=decide(ConnectRules(),ConnectPosition(),0,profiles()[0],'rules_persona',replace(SMALL,max_nodes=0),1,0,'cut')
        self.assertFalse(s['used']);self.assertEqual(len(c['actions']),7)
        self.assertTrue(all(a['outcomes'][0]['objective']==0 for a in c['actions']))
        self.assertEqual(len(set(s['persona_scores'].values())),1)
        self.assertIsNone(s['estimated_goal_regret'])

    def test_recognizes_rule_immediate_win_without_shape_evaluator(self):
        state=ConnectPosition()
        for col in (0,6,1,6,2,5):state=state.play(f'DROP:{col}')
        c,d,s=decide(ConnectRules(),state,0,profiles()[0],'rules_persona',SMALL,4,6,'win')
        self.assertEqual(s['actions']['DROP:3']['win_share'],1.)
        self.assertTrue(s['actions']['DROP:3']['exact'])
        self.assertEqual(d['action_id'],'DROP:3')

    def test_personalities_receive_same_terminal_samples(self):
        rules=GoofRules(3);state=Pending(Position.start(3,4));out=[]
        for p in profiles():
            c,d,s=decide(rules,state,0,p,'rules_persona',SMALL,5,0,'same')
            out.append((c,s))
        for c,s in out[1:]:
            self.assertEqual(c['actions'],out[0][0]['actions'])
            self.assertEqual(s['actions'],out[0][1]['actions'])
        a,_=observe(rules,state,0,profiles()[0],'rules_only',SMALL,5,0,'same')
        b,_=observe(rules,state,0,profiles()[3],'rules_only',SMALL,5,0,'same')
        self.assertEqual(a,b)

    def test_private_deck_not_in_forecast_or_observation(self):
        rules,s,deck=setup('no_thanks_basic',3,2)
        self.assertFalse(hasattr(rules,'deck'));self.assertNotIn('deck',rules.observation(s))
        c,d,stats=decide(rules,s,0,profiles()[0],'rules_persona',SMALL,2,0,'public')
        self.assertNotIn(str(list(deck)),str(c))
        public=replace(s,card=None)
        seen={rules.sample(public,np.random.default_rng(i),0).card for i in range(100)}
        self.assertTrue(seen.isdisjoint(s.seen));self.assertTrue(seen-set(deck))

    def test_closed_loop_games_referee_budget_and_replay(self):
        before=core_hashes()
        for game,players in (('connect_four',2),('goofspiel',3),('no_thanks_basic',3)):
            a=play(game,players,0,0,'rules_persona','random',SMALL)
            b=play(game,players,0,0,'rules_persona','random',SMALL)
            self.assertTrue(a['finished']);self.assertEqual(a['trace'],b['trace'])
            for t in a['trace']:
                if t['own']:self.assertLessEqual(t['stats']['additional_nodes'],SMALL.max_nodes)
        self.assertEqual(before,core_hashes())

    def test_profile_probe_control_and_common_contract(self):
        r=personality_probe(1,SMALL)
        self.assertTrue(r['rows'])
        self.assertTrue(all(not row['different'] for row in r['rows'] if row['method']=='rules_only'))
        for row in r['rows']:
            self.assertEqual(len(row['choices']),4)
            self.assertTrue(all(x is None or 0<=x<=1 for x in row['goal_regrets']))

    def test_invalid_configuration_rejected(self):
        args=(ConnectRules(),ConnectPosition(),0,profiles()[0])
        with self.assertRaises(ValueError):decide(*args,'bad',SMALL,0,0,'bad')
        with self.assertRaises(ValueError):decide(*args,'rules_only',replace(SMALL,rollout_policy='tactical'),0,0,'bad')
        with self.assertRaises(ValueError):decide(ConnectRules(),ConnectPosition(),1,profiles()[0],'rules_only',SMALL,0,0,'wrong-actor')


if __name__=='__main__':unittest.main()
