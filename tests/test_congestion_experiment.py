"""Diagnostics contracts; no tuning on the predeclared heldout game outcomes."""
import copy
from dataclasses import replace
import unittest
from unittest.mock import patch
from reflex.combat import Battle,Unit,alive,legal,resolve,battle_record
from reflex.combat_planning import TacticalControl,forecast
from reflex.core import Policy
from reflex.laboratory import profiles
from reflex.congestion_experiment import (SCENES,jobs,authored_world,opportunities,
    enemy_hypotheses,trace_diagnostics,forced_probes,totals,paired_divergence,run_job,summarize)
from reflex.validation_experiment import combat,replay,replay_progress


def run_row(w,choices,adopted=False,seed=592):
    after,audit=resolve(w,choices,123)
    return dict(seed=seed,trace=[dict(before=battle_record(w),after=battle_record(after),choices=choices,audit=audit,
        selected_routes={i:'secure' for i in alive(w,seed%2)},planning=dict(adopted=adopted))])


class CongestionTests(unittest.TestCase):
    def test_matrix_complete_paired_and_profile_seat_balanced(self):
        work=jobs();self.assertEqual(len(work),128);self.assertEqual(len(set(work)),128)
        self.assertEqual(sum(j[0]=='controlled' for j in work),96)
        for i in range(0,len(work),2):
            self.assertEqual(work[i][:-1],work[i+1][:-1])
            self.assertEqual(tuple(j[-1] for j in work[i:i+2]),('baseline','root-completion'))
        for p in profiles():
            rows=[j for j in work if j[0]=='heldout' and j[4]==p['id'] and j[-1]=='baseline']
            self.assertEqual(sorted(j[5]%2 for j in rows),[0,0,1,1])

    def test_authored_states_legal_mirrored_nonterminal_and_bounded(self):
        for scene in SCENES:
            a,b=authored_world(scene,'both',0),authored_world(scene,'both',1)
            self.assertEqual(a.limit,24);self.assertEqual(a.tick,0)
            self.assertEqual(len({u.pos for u in a.units}),6)
            for team in (0,1):
                for i,j in zip(alive(a,team),alive(b,1-team)):
                    self.assertEqual((8-a.units[i].x,a.units[i].y),b.units[j].pos)
                    self.assertTrue(legal(a,i));self.assertTrue(legal(b,j))
            self.assertEqual(sorted((8-x,y) for x,y in a.walls),sorted(b.walls))
        with self.assertRaises(ValueError):authored_world('mixed_junction','both',2)

    def test_declared_targets_separate_legal_opportunities(self):
        for team in (0,1):
            self.assertGreater(opportunities(authored_world(SCENES[0],'both',team),team)['friendly_only'],0)
            self.assertGreater(opportunities(authored_world(SCENES[1],'both',team),team)['opponent_only'],0)
            self.assertGreater(opportunities(authored_world(SCENES[2],'both',team),team)['mixed'],0)

    def test_exposure_requires_selected_declined_friendly_conflict(self):
        w=authored_world('mixed_junction','both',0)
        choices={i:'move:4:2' if 'move:4:2' in legal(w,i) else 'guard' for i in range(6)}
        r=run_row(w,choices)
        d=trace_diagnostics(r)['diagnostics']
        self.assertEqual(d['declined_friendly_conflict_ticks'],1)
        self.assertEqual(d['declined_friendly_conflict_moves'],2)
        self.assertEqual(d['occupancy_actual_contests'],2)
        r['trace'][0]['planning']['adopted']=True
        d=trace_diagnostics(r)['diagnostics']
        self.assertEqual(d['declined_friendly_conflict_ticks'],0)
        self.assertEqual(d['adopted_plan_friendly_conflict_ticks'],1)

    def test_before_completion_roots_measure_exposure(self):
        w=authored_world('friendly_convergence','both',0)
        choices={i:'guard' for i in range(6)}
        r=run_row(w,choices)
        r['trace'][0]['planning']['root_completion']=dict(before=['move:2:2','move:2:2','guard'],reason='compatible friendly root completion',adapter=dict(feasible=1))
        self.assertEqual(trace_diagnostics(r)['diagnostics']['declined_friendly_conflict_moves'],2)

    def test_hypotheses_use_only_public_prestate(self):
        w=authored_world('mixed_junction','both',0);old=copy.deepcopy(w)
        with patch('reflex.combat.opponent',side_effect=AssertionError('actual controller consulted')),patch('reflex.combat.resolve',side_effect=AssertionError('actual RNG consulted')):
            hypotheses=enemy_hypotheses(w,0)
        self.assertEqual(w,old)
        for h in hypotheses:
            self.assertEqual(set(h),set(alive(w,1)))
            self.assertTrue(all(k in legal(w,i) for i,k in h.items()))

    def test_saved_forecast_hypotheses_match_first_step(self):
        from reflex.combat import make_context
        w=authored_world('enemy_convergence','secure',0);actors=alive(w,0)
        cs=[make_context(w,i,profiles()[0],592,'secure',survival_security=True) for i in actors]
        f=forecast(w,actors,cs,[Policy().choose(c) for c in cs],TacticalControl(horizon=1,samples=2,max_plans=2,trace_execution=True))
        hs=enemy_hypotheses(w,0)
        for plan in f.audit['plans'].values():
            for i,sample in enumerate(plan['execution_samples']):
                self.assertEqual(sample[0]['opponent'],{str(j):k for j,k in hs[i].items()})

    def test_realized_geometry_holds_enemy_prestate_fixed(self):
        w=authored_world('enemy_convergence','both',0)
        choices={i:'guard' for i in range(6)}
        choices[0]='move:4:1';choices[3]='move:4:1'
        r=run_row(w,choices)
        d=trace_diagnostics(r)['diagnostics']
        self.assertEqual(d['realized_secure_distance_gain'],0)
        self.assertEqual(d['realized_eliminate_distance_gain'],0)
        self.assertEqual(d['chosen_route_moves'],1)
        self.assertEqual(d['chosen_route_improving_moves'],1)
        w=authored_world('friendly_convergence','both',0)
        choices={i:'guard' for i in range(6)}
        choices.update({0:'move:2:2',1:'move:2:2',3:'move:6:0'})
        r=run_row(w,choices)
        self.assertNotEqual(r['trace'][0]['before']['units'][3],r['trace'][0]['after']['units'][3])
        self.assertEqual(trace_diagnostics(r)['diagnostics']['realized_eliminate_distance_gain'],0)

    def test_forced_probes_are_separate_and_persona_contract_is_not_rewritten(self):
        rows=forced_probes();self.assertEqual(len(rows),24)
        for r in rows:
            self.assertNotIn('diagnostics',r)
            self.assertIn('forced mechanism probe',r['scope'])
            self.assertGreater(r['actual_collisions'],0)
            self.assertTrue(all(v<=.025+1e-12 for v in r['completion']['persona_regret']))

    def test_zero_attempt_denominators_are_none(self):
        r=dict(movement={},diagnostics={},total_work={},selected_routes={},longest_observed_stall=0)
        t=totals([r]);self.assertIsNone(t['target_failure_rate']);self.assertIsNone(t['all_failure_rate'])
        self.assertIsNone(t['enemy_occupancy_brier']);self.assertEqual(t['exposure_games'],0)

    def test_failed_move_categories_partition_actual_failures(self):
        from reflex.root_collision_experiment import movement_and_completion
        targets=('move:2:2','move:4:1','move:4:2')
        for scene,key in zip(SCENES,targets):
            w=authored_world(scene,'both',0)
            choices={i:key if key in legal(w,i) else 'guard' for i in range(6)}
            r=run_row(w,choices);r.update(won=False,lost=False)
            m=movement_and_completion(r)['movement']
            self.assertEqual(m['failed_moves'],sum(m.get(k,0) for k in ('friendly_collision_moves','opponent_collision_moves','both_collision_moves','unclassified_failed_moves')))
            self.assertEqual(m.get('unclassified_failed_moves',0),0)
            self.assertEqual(m['move_attempts'],sum(choices[i].startswith('move:') for i in alive(w,0)))

    def test_all_wait_candidate_and_search_bound_fail_cleanly(self):
        rows=[]
        for partition in ('controlled','heldout'):
            for variant in ('baseline','root-completion'):
                rows.append(dict(partition=partition,variant=variant,profile='care',scenario='fixture',
                    movement=dict(move_attempts=4,failed_moves=2,friendly_collision_moves=2) if variant=='baseline' else {},
                    diagnostics={},total_work={},selected_routes={},longest_observed_stall=0,decisions=3,
                    completion_max_combinations=1729 if variant=='root-completion' else 0))
        result=summarize(rows)
        self.assertFalse(result['natural_exposure_sufficient'])
        self.assertIn('target failure reduction below20percent or unexposed',result['gate_failures'])
        self.assertIn('completion search bound exceeded',result['gate_failures'])

    def test_first_divergence_rejects_changed_enemy_on_same_state(self):
        w=authored_world('friendly_convergence','both',0)
        choices={i:'guard' for i in range(6)};choices.update({0:'move:2:2',1:'move:2:2'})
        a=run_row(w,choices);b=copy.deepcopy(a);b['trace'][0]['choices'][1]='guard'
        b['trace'][0]['planning']['root_completion']=dict(adopted=True,before=['move:2:2','move:2:2','guard'],after=['move:2:2','guard','guard'])
        self.assertTrue(paired_divergence(a,b)['same_revealed_enemy_choices'])
        b['trace'][0]['choices'][3]='move:6:0'
        with self.assertRaisesRegex(AssertionError,'opponent differed'):paired_divergence(a,b)

    def test_optional_harness_trace_does_not_change_actions_feedback_or_learning(self):
        w=Battle.start('open','both',limit=2);p=profiles()[0]
        kw=dict(profile=p,seed=172,map_name='open',goal='both',mode='baseline',learn=True,survival_security=True,
            progress_watch=True,initial_world=w,planner=TacticalControl(horizon=1,samples=1,max_plans=2))
        a=combat(**kw);b=combat(**kw,trace_roots=True)
        defaults={k:v for k,v in kw.items() if k!='initial_world'}
        with patch('reflex.combat.Battle.start',return_value=w):default=combat(**defaults)
        for row in b['trace']:
            self.assertEqual(set(row.pop('selected_routes')),set(alive(battle_from(row['before']),0)))
        for r in (a,b,default):
            for k in ('decision_p50_ms','decision_and_feedback_p50_ms'):r.pop(k)
        self.assertEqual(a,b);self.assertEqual(a,default);self.assertEqual(replay(a),2);self.assertGreater(replay_progress(a),0)


def battle_from(record):
    from reflex.combat import battle_from_record
    return battle_from_record(record)

if __name__=='__main__':unittest.main()
