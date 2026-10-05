import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import unittest
import numpy as np
from reflex.core import Policy, compile_batch
from reflex.examples import effect
from reflex.planning import vector
from reflex.board_models import ConnectPosition, ConnectAdapter, ThanksPosition, ThanksAdapter
from reflex.board_planning import REFLEX, core_audit
from reflex.board_models import observe
from reflex.laboratory import profiles
from reflex.monte_carlo import RolloutBudget, TerminalEvaluation, evaluate_actions, wilson
from reflex.rollout_boards import BoardRolloutModel, observe_rollouts, connect_tactical
from reflex.monte_carlo_comparison import play_connect, play_thanks


class Toy:
    def begin_trial(self): pass
    def terminal(self,s): return s>=3
    def chance(self,s): return False
    def legal(self,s): return ('next',) if not self.terminal(s) else ()
    def step(self,s,a): return s+1
    def choose(self,s,rng,policy): return 'next'
    def evaluate(self,s): return TerminalEvaluation(effect(1),1.,True,False)


class MonteCarloTests(unittest.TestCase):
    def test_invalid_budgets_rejected(self):
        for kwargs in (dict(samples=0),dict(samples=True),dict(max_nodes=-1),dict(max_steps=0),
                       dict(min_samples=9,samples=8),dict(rollout_policy='oracle')):
            with self.assertRaises(ValueError): RolloutBudget(**kwargs)

    def test_terminal_roots_need_no_extra_transitions_and_are_exact(self):
        packed,s=evaluate_actions({'win':3,'win2':4},Toy(),RolloutBudget(samples=4,max_nodes=0),[1])
        self.assertTrue(s['used']); self.assertEqual(s['additional_nodes'],0)
        self.assertEqual(s['completed_samples'],4)
        self.assertEqual(s['actions']['win']['win_rate_interval95'],[1.,1.])
        self.assertEqual(packed['win'][0]['p'],1.)

    def test_incomplete_sample_round_is_discarded_for_all_actions(self):
        packed,s=evaluate_actions({'a':1,'b':2},Toy(),RolloutBudget(samples=4,max_nodes=5,min_samples=1),[1])
        self.assertEqual(s['completed_samples'],1); self.assertEqual(s['additional_nodes'],5)
        self.assertTrue(s['budget_exhausted']); self.assertEqual(s['discarded_terminal_samples'],1)
        self.assertEqual({v['samples'] for v in s['actions'].values()},{1})
        self.assertEqual(set(packed),{'a','b'})

    def test_length_cut_is_not_a_loss_and_below_floor_means_fallback(self):
        packed,s=evaluate_actions({'a':0,'b':2},Toy(),RolloutBudget(samples=4,max_steps=1),[1])
        self.assertFalse(s['used']); self.assertTrue(s['length_exhausted']); self.assertEqual(packed,{})
        self.assertIsNone(s['actions']['a']['win_rate'])
        _,s=evaluate_actions({'a':1,'b':2},Toy(),RolloutBudget(samples=4,max_nodes=5,min_samples=2),[1])
        self.assertFalse(s['used']); self.assertEqual(s['completed_samples'],1)

    def test_order_independence_and_common_trial_streams(self):
        class Chance(Toy):
            def chance(self,s): return s==0
            def sample(self,s,rng): return 3+int(rng.integers(2))
            def evaluate(self,s): return TerminalEvaluation(effect(1 if s==3 else -1),float(s==3),s==3,False)
        budget=RolloutBudget(samples=32,max_nodes=64)
        a,s=evaluate_actions({'left':0,'right':0},Chance(),budget,[3])
        b,t=evaluate_actions({'right':0,'left':0},Chance(),budget,[3])
        self.assertEqual(a,b); self.assertEqual(s,t); self.assertEqual(a['left'],a['right'])
        self.assertEqual(s['actions']['left'],s['actions']['right'])

    def test_invalid_terminal_result_rejected(self):
        class Bad(Toy):
            def evaluate(self,s): return TerminalEvaluation(effect(float('nan')),1.,True,False)
        with self.assertRaises(ValueError): evaluate_actions({'a':3},Bad(),RolloutBudget(),[1])

    def test_wilson_bounds_are_sampling_uncertainty_not_zero_at_small_samples(self):
        self.assertIsNone(wilson(0,0))
        self.assertGreater(wilson(0,8)[1],.3)
        self.assertLess(wilson(8,8)[0],.7)
        self.assertLess(wilson(8,32)[1]-wilson(8,32)[0],wilson(2,8)[1]-wilson(2,8)[0])

    def test_fallback_context_and_rng_are_identical_to_reflex(self):
        adapter=ConnectAdapter(); state=ConnectPosition(); profile=profiles()[0]
        a,_=observe(adapter,state,profile,REFLEX,2,0,'fallback')
        b,s=observe_rollouts(adapter,state,profile,RolloutBudget(max_nodes=0),2,0,'fallback')
        self.assertEqual(a,b); self.assertFalse(s['used']); self.assertEqual(Policy().choose(a),Policy().choose(b))
        c,_=observe_rollouts(adapter,state,profile,RolloutBudget(samples=4,min_samples=4),2,0,'fallback')
        np.testing.assert_array_equal(compile_batch([a]).rng,compile_batch([c]).rng)

    def test_known_immediate_win_and_obvious_reply_loss_are_recognized(self):
        p=ConnectPosition()
        for col in (0,6,1,6,2,5): p=p.play(f'DROP:{col}')
        c,s=observe_rollouts(ConnectAdapter(),p,profiles()[0],RolloutBudget(),1,6,'win')
        self.assertEqual(s['actions']['DROP:3']['win_rate'],1.)
        self.assertTrue(s['actions']['DROP:3']['exact'])
        self.assertEqual(Policy().choose(c,False)['action_id'],'DROP:3')
        p=ConnectPosition()
        for col in (6,0,6,1,5,2): p=p.play(f'DROP:{col}')
        c,s=observe_rollouts(ConnectAdapter(),p,profiles()[0],RolloutBudget(),1,6,'block')
        self.assertTrue(all(r['win_rate']==0 for a,r in s['actions'].items() if a!='DROP:3'))
        self.assertEqual(Policy().choose(c,False)['action_id'],'DROP:3')

    def test_fast_tactical_moves_match_independent_successor_legality(self):
        rng=np.random.default_rng(723)
        for _ in range(20):
            state=ConnectPosition()
            while state.legal():
                side=state.turn
                children={a:state.play(a) for a in state.legal()}
                wins=[a for a,s in children.items() if s.winner()==side]
                safe=[a for a,s in children.items() if not any(s.play(b).winner()==1-side for b in s.legal())]
                name=connect_tactical(state,rng)
                self.assertIn(name,wins or safe or list(children))
                state=children[name]

    def test_unknown_cards_sample_public_set_without_real_deck_or_roster(self):
        state=ThanksPosition.start(7,20); profile=profiles()[3]
        model=BoardRolloutModel(ThanksAdapter(),state,profile,1,0,'public')
        self.assertNotIn('deck',model.__dict__); self.assertNotIn('roster',model.__dict__)
        after=state.play('TAKE'); rng=np.random.default_rng(3)
        self.assertNotIn(model.sample(after,rng).card,state.seen)
        profile['values']['power']=0
        self.assertEqual(model.profile,profiles()[3])

    def test_tied_no_thanks_result_has_fractional_credit_and_raw_score(self):
        base=ThanksPosition.start(3,20)
        final=replace(base,cards=((3,),(4,),(5,)),chips=(1,2,0),card=None,remaining=0)
        model=BoardRolloutModel(ThanksAdapter(),base,profiles()[0],1,0,'tie')
        r=model.evaluate(final)
        self.assertTrue(r.won); self.assertEqual(r.win_share,.5); self.assertEqual(r.game_score,2.)
        self.assertEqual(r.outcome['objective'],0.)
        with self.assertRaises(ValueError): model.evaluate(base)

    def test_persona_continuation_uses_a_copy_of_own_memory(self):
        state=ThanksPosition.start(3,3)
        base,_=observe(ThanksAdapter(),state,profiles()[0],REFLEX,1,0,'persona')
        memory=Policy().choose(base)['next_state']; old=copy.deepcopy(memory)
        model=BoardRolloutModel(ThanksAdapter(),state,profiles()[0],1,0,'persona',memory)
        model.begin_trial(); choice=model.choose(state,np.random.default_rng(3),'persona')
        self.assertIn(choice,state.legal()); self.assertEqual(memory,old)
        model.begin_trial(); self.assertEqual(model.memories[0],old)

    def test_parallel_rollouts_are_isolated_and_reproducible(self):
        state=ThanksPosition.start(3,20); old=copy.deepcopy(state); adapter=ThanksAdapter()
        budget=RolloutBudget(samples=4,min_samples=4)
        def run(seed):
            c,s=observe_rollouts(adapter,state,profiles()[0],budget,seed,0,f'actor-{seed}')
            return Policy().choose(c),s
        expected=[run(i) for i in range(4)]
        with ThreadPoolExecutor(max_workers=4) as pool: actual=list(pool.map(run,range(4)))
        self.assertEqual(actual,expected); self.assertEqual(state,old)

    def test_persona_policy_completes_terminal_rollouts_without_mutating_personality(self):
        state=ThanksPosition.start(3,20); profile=profiles()[3]; old=copy.deepcopy(profile)
        c,s=observe_rollouts(ThanksAdapter(),state,profile,
                             RolloutBudget(samples=1,min_samples=1,rollout_policy='persona'),4,0,'full-persona')
        self.assertTrue(s['used']); self.assertEqual(s['completed_samples'],1)
        self.assertEqual(profile,old); self.assertIn(Policy().choose(c)['action_id'],state.legal())

    def test_compressed_terminal_effects_keep_the_reported_mean_return(self):
        c,s=observe_rollouts(ThanksAdapter(),ThanksPosition.start(5,20),profiles()[0],
                             RolloutBudget(samples=32),3,0,'moments')
        for a in c['actions']:
            self.assertLessEqual(len(a['outcomes']),8)
            self.assertAlmostEqual(sum(r['p'] for r in a['outcomes']),1.)
            self.assertAlmostEqual(sum(r['p']*vector(r)[0] for r in a['outcomes']),s['actions'][a['id']]['mean_return'])

    def test_complete_balanced_matches_replay_and_preserve_real_game_separation(self):
        budget=RolloutBudget(samples=4,min_samples=4)
        for run,args in ((play_connect,(2,0,0,'mc4',budget,'minimax2')),
                         (play_thanks,(2,3,1,0,'mc4',budget,'tactical'))):
            a=run(*args); b=run(*args); a.pop('decision_ms'); b.pop('decision_ms')
            self.assertEqual(a,b); self.assertTrue(a['finished'])
            for row in a['trace']:
                if row['evaluation'] is not None: self.assertLessEqual(row['evaluation']['additional_nodes'],budget.max_nodes)

    def test_shared_personality_core_and_persistent_runtime_unchanged(self):
        r=core_audit()
        if r['unchanged_verified'] is not None: self.assertTrue(r['unchanged_verified'])


if __name__=='__main__': unittest.main()
