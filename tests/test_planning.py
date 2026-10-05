import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import unittest
import numpy as np
from reflex.core import Policy, compile_batch
from reflex.planning import PlanningBudget, refine, compress_outcomes, vector
from reflex.examples import effect
from reflex.board_models import (ConnectPosition, ConnectAdapter, connect_referee, ThanksPosition, ThanksAdapter,
                                card_points, thanks_referee_score, observe, neutral, PersonaModel)
from reflex.board_planning import REFLEX, SHORT, CONNECT_FULL, run_connect, run_thanks, core_audit, preserve_teacher
from reflex.laboratory import profiles


class Tree:
    def terminal(self,s): return len(s)>=5
    def chance(self,s): return False
    def legal(self,s): return ('left','right')
    def order(self,s,m): return m
    def step(self,s,a): return s+(a,)
    def select(self,s,options): return max(options,key=lambda a:len(options[a][0][1])+int(a=='right'))


class PlanningTests(unittest.TestCase):
    def test_budget_configuration_rejects_invalid_controls(self):
        for kwargs in (dict(depth=0),dict(depth=True),dict(max_nodes=-1),dict(width=0),dict(chance_samples=9),dict(enabled=1)):
            with self.assertRaises(ValueError): PlanningBudget(**kwargs)

    def test_off_is_zero_additional_work_and_depth_one_is_identical(self):
        roots={'a':('a',),'b':('b',)}; tree=Tree()
        off,stats=refine(roots,tree,PlanningBudget(enabled=False,depth=6,max_nodes=20))
        one,s=refine(roots,tree,PlanningBudget(depth=1,max_nodes=20))
        self.assertEqual(off,one); self.assertEqual(stats['additional_nodes'],0)
        self.assertFalse(stats['used']); self.assertEqual(s['reached_depth'],1)

    def test_budget_cut_discards_partial_horizon_and_retains_all_roots(self):
        roots={'a':('a',),'b':('b',),'c':('c',)}; tree=Tree()
        result,s=refine(roots,tree,PlanningBudget(depth=3,max_nodes=8,width=2))
        expected,e=refine(roots,tree,PlanningBudget(depth=2,max_nodes=8,width=2))
        self.assertEqual(result,expected); self.assertEqual(set(result),set(roots))
        self.assertEqual(s['reached_depth'],2); self.assertTrue(s['budget_exhausted'])
        self.assertEqual(s['additional_nodes'],8)
        reverse,r=refine(dict(reversed(list(roots.items()))),tree,PlanningBudget(depth=3,max_nodes=8,width=2))
        self.assertEqual(result,reverse)

    def test_width_pruning_is_reported_but_does_not_prune_roots(self):
        roots={'a':('a',),'b':('b',)}
        result,s=refine(roots,Tree(),PlanningBudget(depth=3,max_nodes=8,width=1))
        self.assertEqual(set(result),set(roots)); self.assertGreater(s['beam_pruned_actions'],0)
        self.assertEqual(s['reached_depth'],3)

    def test_chance_mass_and_generation_count_are_bounded(self):
        class Chance(Tree):
            def chance(self,s): return len(s)==1
            def draws(self,s,k): return [(1/k,s+(f'draw-{i}',)) for i in range(k)]
        roots={'a':('a',),'b':('b',)}
        result,s=refine(roots,Chance(),PlanningBudget(depth=2,max_nodes=12,width=1,chance_samples=2))
        self.assertEqual(s['chance_draws'],4)
        for leaves in result.values(): self.assertAlmostEqual(sum(p for p,_ in leaves),1)
        fallback,s=refine(roots,Chance(),PlanningBudget(depth=2,max_nodes=1,chance_samples=2))
        self.assertEqual(s['additional_nodes'],0); self.assertEqual(s['reached_depth'],1)
        class Invalid(Chance):
            def draws(self,s,k): return [(float('nan'),s+('bad',))]
        with self.assertRaises(ValueError): refine(roots,Invalid(),PlanningBudget(depth=2,max_nodes=12))

    def test_outcome_compression_preserves_probability_and_first_moment(self):
        rows=[effect((i-6)/8,p=1/12,values={'power':i/12},cost=i/24) for i in range(12)]
        before=copy.deepcopy(rows); packed=compress_outcomes(rows)
        self.assertLessEqual(len(packed),8); self.assertAlmostEqual(sum(r['p'] for r in packed),1)
        np.testing.assert_allclose(sum(r['p']*vector(r) for r in packed),sum(r['p']*vector(r) for r in rows))
        self.assertEqual(rows,before)

    def test_outcome_compression_accepts_smaller_bounded_budgets(self):
        rows=[effect((i-6)/8,p=1/12,values={'power':i/12},cost=i/24) for i in range(12)]
        before=copy.deepcopy(rows);expected=sum(r['p']*vector(r) for r in rows)
        for limit in (1,2,4,8):
            packed=compress_outcomes(rows,max_outcomes=limit)
            self.assertLessEqual(len(packed),limit)
            self.assertAlmostEqual(sum(r['p'] for r in packed),1)
            np.testing.assert_allclose(sum(r['p']*vector(r) for r in packed),expected)
            if limit>1:self.assertEqual(packed[0],rows[0])
        self.assertEqual(rows,before)
        self.assertEqual(compress_outcomes(rows),compress_outcomes(rows,max_outcomes=8))
        for invalid in (0,9,-1,True,1.5,'2'):
            with self.assertRaises(ValueError):compress_outcomes(rows,max_outcomes=invalid)

    def test_off_and_one_have_same_rng_and_choice(self):
        adapter=ConnectAdapter(); p=ConnectPosition(); profile=profiles()[0]
        a,sa=observe(adapter,p,profile,REFLEX,2,0,'same')
        b,sb=observe(adapter,p,profile,PlanningBudget(depth=1),2,0,'same')
        self.assertEqual(a,b); self.assertEqual(Policy().choose(a),Policy().choose(b))
        c,sc=observe(adapter,p,profile,SHORT,2,0,'same')
        np.testing.assert_array_equal(compile_batch([a]).rng,compile_batch([c]).rng)

    def test_connect_official_gravity_horizontal_vertical_diagonal(self):
        p=ConnectPosition()
        for col in (0,6,1,6,2,5,3): p=p.play(f'DROP:{col}')
        self.assertEqual(p.winner(),0); self.assertFalse(p.legal())
        self.assertEqual(connect_referee(p)[0],0)
        with self.assertRaises(ValueError): p.play('DROP:4')
        p=ConnectPosition()
        for col in (0,1,0,1,0,2,0): p=p.play(f'DROP:{col}')
        self.assertEqual(p.winner(),0); self.assertEqual(connect_referee(p)[0],0)
        for direction in (1,-1):
            cells=[(i if direction==1 else 6-i,i) for i in range(4)]
            bits=sum(1<<(7*x+y) for x,y in cells)
            p=ConnectPosition((bits,0))
            self.assertEqual(p.winner(),0); self.assertEqual(connect_referee(p)[0],0)

    def test_connect_blocks_an_obvious_next_turn_loss_with_reading(self):
        p=ConnectPosition()
        for col in (6,0,6,1,5,2): p=p.play(f'DROP:{col}')
        c,s=observe(ConnectAdapter(),p,profiles()[0],SHORT,1,6,'block')
        self.assertEqual(Policy().choose(c,False)['action_id'],'DROP:3')
        self.assertGreaterEqual(s['reached_depth'],2)

    def test_full_four_plies_complete_with_the_declared_budget(self):
        p=ConnectPosition()
        for col in (5,5,1,1,6,0,3,3): p=p.play(f'DROP:{col}')
        c,s=observe(ConnectAdapter(),p,profiles()[0],CONNECT_FULL,3,8,'complete-horizon')
        self.assertEqual(s['reached_depth'],4); self.assertFalse(s['budget_exhausted'])
        self.assertLessEqual(s['additional_nodes'],4096)
        self.assertEqual(s['beam_pruned_actions'],0)

    def test_teacher_snapshots_keep_config_and_no_future_gold(self):
        c,s=observe(ConnectAdapter(),ConnectPosition(),profiles()[0],SHORT,1,0,'teacher')
        sink=[]; preserve_teacher(c,sink,s)
        self.assertEqual(sink[0]['planning']['reached_depth'],3)
        self.assertEqual(sink[0]['quality'],'awaiting_generation')
        self.assertNotIn('preferred',sink[0]); self.assertNotIn('reference_action',sink[0])
        self.assertIn('forecast_horizon',sink[0]['context']['facts'])

    def test_no_thanks_basic_inventory_forced_take_chain_and_keep_turn(self):
        for n,amount in ((3,11),(5,11),(6,9),(7,7)):
            p=ThanksPosition.start(n,20); self.assertEqual(p.chips,(amount,)*n)
        p=ThanksPosition.start(3,20); p=p.play('PASS')
        self.assertEqual(p.turn,1); self.assertEqual(p.pot,1); self.assertEqual(p.chips[0],10)
        p=p.play('TAKE'); self.assertEqual(p.turn,1); self.assertEqual(p.chips[1],12)
        self.assertEqual(card_points((13,15,16)),28)
        self.assertEqual(card_points((13,14,15,16)),13)
        self.assertEqual(thanks_referee_score((13,14,15,16),8),5)
        p=replace(ThanksPosition.start(3,35),chips=(0,11,11))
        self.assertEqual(p.legal(),('TAKE',))
        with self.assertRaises(ValueError): p.play('PASS')

    def test_unknown_draws_only_use_public_seen_set_and_stable_sample_prefix(self):
        p=ThanksPosition.start(3,20).play('TAKE'); adapter=ThanksAdapter(3)
        a=adapter.draws(p,2); b=adapter.draws(p,4)
        self.assertEqual([s.card for _,s in a],[s.card for _,s in b][:2])
        self.assertNotIn(20,[s.card for _,s in b]); self.assertNotIn('deck',p.__dataclass_fields__)
        self.assertEqual(adapter.draws(p,2),a)

    def test_denial_value_only_counts_an_actual_rival_chain_benefit(self):
        adapter=ThanksAdapter(); p=ThanksPosition.start(3,20)
        after=p.play('TAKE'); e=adapter.consequence(p,after,0)
        self.assertAlmostEqual(e['values']['power'],-20/35)
        self.assertEqual(e['values']['benevolence'],0)
        p=replace(p,cards=((),(19,21),()))
        e=adapter.consequence(p,p.play('TAKE'),0)
        self.assertLess(e['values']['benevolence'],0)

    def test_parallel_planning_uses_independent_states_and_random_streams(self):
        adapter=ThanksAdapter(3); state=ThanksPosition.start(3,20); old=copy.deepcopy(state)
        def choose(seed):
            c,s=observe(adapter,state,profiles()[0],SHORT,seed,0,f'isolated-{seed}')
            return Policy().choose(c),s
        expected=[choose(i) for i in range(4)]
        with ThreadPoolExecutor(max_workers=4) as pool: actual=list(pool.map(choose,range(4)))
        self.assertEqual(expected,actual); self.assertEqual(state,old)

    def test_persona_model_does_not_receive_opponent_roster(self):
        p=ThanksPosition.start(3,20); profile=profiles()[3]
        model=PersonaModel(ThanksAdapter(),0,profile,1,0,'episode',None)
        old=copy.deepcopy(profile)
        options={a:[(1,p.play(a))] for a in p.legal()}
        first=model.select(p,options); profile['values']['power']=0
        self.assertEqual(model.select(p,options),first)
        self.assertEqual(model.profile,old); self.assertFalse(hasattr(model,'roster'))

    def test_mixed_games_and_player_counts_share_policy_batch(self):
        examples=[]
        for n in (3,5,7):
            c,_=observe(ThanksAdapter(),ThanksPosition.start(n,20),profiles()[0],REFLEX,1,0,f'count-{n}')
            examples.append(c)
        c,_=observe(ConnectAdapter(),ConnectPosition(),profiles()[0],REFLEX,1,0,'board')
        examples.append(c); p=Policy(); batch=compile_batch(examples)
        self.assertEqual(p.decide(batch).records(batch),[p.choose(c) for c in examples])

    def test_complete_games_replay_budget_limits_and_input_preservation(self):
        a=run_connect(1,SHORT); b=run_connect(1,SHORT)
        a.pop('decision_ms'); b.pop('decision_ms'); self.assertEqual(a,b)
        c=run_thanks(1,3,SHORT); d=run_thanks(1,3,SHORT)
        c.pop('decision_ms'); d.pop('decision_ms'); self.assertEqual(c,d)
        self.assertFalse(c['truncated']); self.assertEqual(len(c['scores']),3)
        for r in (a,c):
            self.assertTrue(all(t['planning']['additional_nodes']<=SHORT.max_nodes for t in r['trace']))

    def test_personality_and_per_actor_compute_budget_are_independent(self):
        game=run_connect(0,(REFLEX,SHORT))
        self.assertEqual(game['roster'],['growth','ego'])
        self.assertTrue(all(t['planning']['additional_nodes']==0 for t in game['trace'] if t['actor']==0))
        self.assertTrue(any(t['planning']['additional_nodes']>0 for t in game['trace'] if t['actor']==1))

    def test_shared_reflex_code_and_runtime_unchanged(self):
        r=core_audit()
        if r['unchanged_verified'] is not None: self.assertTrue(r['unchanged_verified'])


if __name__=='__main__': unittest.main()
