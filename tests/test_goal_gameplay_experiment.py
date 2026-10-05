"""Fresh paired gameplay, complete outcome denominators and causal checks."""
from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reflex.combat import alive, legal, battle_record, battle_from_record, resolve
from reflex.combat_planning import TacticalControl
from reflex.congestion_experiment import authored_world, enemy_hypotheses
from reflex.core import digest
from reflex.goal_rollout import GoalRollout
from reflex.goal_gameplay_experiment import (VARIANTS, CONTRASTS, METRICS, MOVEMENT,
    jobs, timing_jobs, configuration, game_id, condition_id, preregister, check_frozen,
    experiment, verify_run, first_divergence, rates, totals, paired_delta, pair_summary,
    aggregate, semantic_signature)
from reflex.laboratory import profiles
from reflex.root_collision_experiment import movement_and_completion


def fixture(variant='fixed', ticks=1, changed=False):
    world = authored_world('enemy_convergence', 'both', 0, limit=ticks)
    run = dict(genre='combat', partition='controlled', scenario='enemy_convergence/both',
        profile='care', seed=2702, rival='reference', variant=variant, trace=[], ticks=ticks,
        won=False, lost=False, decisions=3*ticks, longest_observed_stall=0, learned_uses=0,
        game_seconds=1., total_work=dict(planner_seconds=.8, calls=ticks, model_transitions=24,
        recovery_calls=0), trace_execution=True)
    for tick in range(ticks):
        choices = {i: 'guard' for i in range(6)}
        if changed and tick == 0:
            choices[0] = 'move:4:1'
        after, audit = resolve(world, choices, int(digest(['combat-world', run['seed']])[:16], 16))
        run['trace'].append(dict(before=battle_record(world), after=battle_record(after), choices=choices,
            audit=audit, selected_routes={i: 'secure' for i in alive(world, 0)}, planning=dict(adopted=False)))
        world = after
    run.update(game=game_id(run), condition=condition_id(run))
    run.update(movement_and_completion(run))
    return run


def add_samples(run):
    for row in run['trace']:
        world = battle_from_record(row['before'])
        choices = (enemy_hypotheses(world, 0)*2 if run['variant'] == 'fixed' else
            [GoalRollout(world, 0, s, run['variant'], samples=4).choose(world, 0) for s in range(4)])
        row['planning']['execution'] = dict(samples=[[dict(opponent={str(i): k for i, k in choice.items()})] for choice in choices])
    return run


class GoalGameplayExperimentTests(unittest.TestCase):
    def test_exact_fresh_96_game_matrix_and_matched_triples(self):
        work = jobs()
        self.assertEqual(len(work), 96)
        self.assertEqual(len({game_id(j) for j in work}), 96)
        self.assertEqual(Counter(j['partition'] for j in work), {'controlled': 48, 'heldout': 48})
        self.assertEqual(len({condition_id(j) for j in work}), 32)
        self.assertEqual({j['seed'] for j in work}, {2701, 2702})
        self.assertFalse({j['seed'] for j in work} & {591, 592, 611, 612, 1701, 1702})
        for index in range(0, len(work), 3):
            triple = work[index:index+3]
            self.assertEqual(tuple(j['variant'] for j in triple), VARIANTS)
            self.assertEqual(len({condition_id(j) for j in triple}), 1)
        for profile in profiles():
            for partition in ('controlled', 'heldout'):
                seats = Counter(j['seed'] % 2 for j in work if j['profile'] == profile['id'] and j['partition'] == partition and j['variant'] == 'fixed')
                self.assertEqual(seats, {0: 2, 1: 2})

    def test_only_opponent_model_differs_and_default_remains_legacy(self):
        self.assertEqual(TacticalControl().opponent_model, 'fixed')
        from dataclasses import asdict
        controls = [asdict(configuration(v)) for v in VARIANTS]
        for control in controls:
            self.assertEqual((control['horizon'], control['samples'], control['max_plans'], control['recovery_plans']), (6, 4, 24, 48))
            self.assertTrue(control['recovery_options'])
            self.assertFalse(control['root_completion'])
            self.assertEqual(control['self_continuation'], 'objective')
            control.pop('opponent_model')
        self.assertEqual(controls[0], controls[1])
        self.assertEqual(controls[1], controls[2])
        with self.assertRaises(ValueError):
            configuration('adaptive')

    def test_four_timing_runs_separate_exact_fixed_fixture(self):
        timing = timing_jobs()
        self.assertEqual(len(timing), 4)
        self.assertEqual([j['variant'] for j in timing], ['fixed', 'goal-uniform', 'fixed', 'goal-uniform'])
        for job in timing:
            self.assertEqual((job['scene'], job['goal'], job['profile'], job['seed']), ('mixed_junction', 'both', 'care', 2702))
            self.assertFalse(job['trace_execution'])
            self.assertIn(job['game'], {j['game'] for j in jobs()})

    def test_preregistration_is_exclusive_and_freezes_negative_offline_evidence(self):
        with tempfile.TemporaryDirectory() as directory, patch('reflex.goal_gameplay_experiment.run_job', side_effect=AssertionError('game before freeze')):
            manifest = preregister(directory)
            self.assertEqual(len(manifest['jobs']), 96)
            self.assertEqual(len(manifest['timing_jobs']), 4)
            self.assertIn('evidence/goal_opponent/evaluation.json', manifest['file_hashes'])
            self.assertIn('tests/test_goal_gameplay_experiment.py', manifest['file_hashes'])
            self.assertEqual(manifest, check_frozen(directory))
            self.assertFalse(manifest['default_promotion'])
            with self.assertRaises(FileExistsError):
                preregister(directory)
            with patch('reflex.goal_gameplay_experiment.sources', return_value={}):
                with self.assertRaisesRegex(ValueError, 'frozen'):
                    check_frozen(directory)

    def test_run_requires_preregister_and_rejects_existing_partial_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                experiment(directory)
            preregister(directory)
            (Path(directory)/'trajectories.jsonl').write_text('{}\n')
            with self.assertRaises(FileExistsError):
                experiment(directory)

    def test_replay_all_variants_and_original_fixed_root_validation(self):
        for variant in VARIANTS:
            checks = verify_run(add_samples(fixture(variant)))
            self.assertEqual(checks['rule_transitions'], 1)
            self.assertEqual(checks['legal_actual_intents'], 6)
            self.assertEqual(checks['fixed_hypothesis_checks'], 4 if variant == 'fixed' else 0)
            self.assertEqual(checks['sampled_hypothesis_checks'], 0 if variant == 'fixed' else 4)

    def test_reject_incorrect_sampled_root_without_imposing_original_hypotheses(self):
        run = add_samples(fixture('goal-uniform'))
        sample = run['trace'][0]['planning']['execution']['samples'][0][0]['opponent']
        sample['3'] = 'not-a-real-action'
        with self.assertRaisesRegex(ValueError, 'sampled public first-step'):
            verify_run(run)

    def test_reject_discontinuous_truncated_and_illegal_traces(self):
        run = fixture(ticks=2)
        run['trace'][1]['before']['tick'] = 0
        with self.assertRaisesRegex(ValueError, 'discontinuous'):
            verify_run(run)
        run = fixture(ticks=2)
        run['trace'].pop()
        with self.assertRaisesRegex(ValueError, 'truncated'):
            verify_run(run)
        run = fixture()
        run['trace'][0]['choices'][0] = 'move:99:99'
        with self.assertRaisesRegex(ValueError, 'illegal'):
            verify_run(run)

    def test_first_difference_is_own_action_on_same_state_and_enemy(self):
        left, right = fixture(), fixture('goal-uniform', changed=True)
        result = first_divergence(left, right)
        self.assertEqual(result['tick'], 0)
        self.assertEqual(result['changed_actors'], [0])
        self.assertTrue(result['same_public_before'])
        self.assertTrue(result['same_actual_enemy'])
        self.assertEqual(result['left_choices']['0'], 'guard')
        self.assertEqual(result['right_choices']['0'], 'move:4:1')

    def test_first_difference_rejects_enemy_leak_or_unexplained_transition(self):
        left, right = fixture(), fixture('goal-uniform')
        right['trace'][0]['choices'][3] = 'move:4:1'
        with self.assertRaisesRegex(AssertionError, 'actual enemy differed'):
            first_divergence(left, right)
        right = fixture('goal-uniform')
        right['trace'][0]['after']['tick'] = 9
        with self.assertRaisesRegex(AssertionError, 'different real transition'):
            first_divergence(left, right)
        right = fixture('goal-uniform')
        right['seed'] = 2701
        with self.assertRaisesRegex(ValueError, 'matched'):
            first_divergence(left, right)

    def test_route_difference_is_separate_from_real_action_divergence(self):
        left, right = fixture(), fixture('goal-uniform')
        right['trace'][0]['selected_routes'][0] = 'eliminate'
        result = first_divergence(left, right)
        self.assertIsNone(result['tick'])
        self.assertTrue(result['same_trajectory'])
        self.assertEqual(result['first_route_change'], 0)

    def test_failure_categories_and_move_denominators(self):
        run = dict(movement=dict(move_attempts=10, failed_moves=6, opponent_collision_moves=2,
            friendly_collision_moves=1, both_collision_moves=3), decisions=20, unresolved_stop_selections=2)
        metrics = rates(run)
        self.assertEqual(metrics['failed_move_rate'], .6)
        self.assertEqual(metrics['enemy_only_failure_rate'], .2)
        self.assertEqual(metrics['enemy_contested_failure_rate'], .5)
        self.assertEqual(metrics['both_failure_rate'], .3)
        self.assertEqual(metrics['unresolved_stop_rate'], .1)
        self.assertTrue(all(v is None for v in rates(dict(movement={}, decisions=0)).values()))

    def test_equal_game_rates_do_not_hide_empty_games_or_long_game_weight(self):
        a, b, empty = fixture(), fixture(), fixture()
        a['movement'].update(move_attempts=1, failed_moves=1)
        b['movement'].update(move_attempts=9, failed_moves=0)
        empty['movement'] = {}
        result = totals([a, b, empty])
        self.assertEqual(result['pooled_rates']['failed_move_rate'], .1)
        self.assertEqual(result['equal_game_rates']['failed_move_rate']['mean'], .5)
        self.assertEqual(result['equal_game_rates']['failed_move_rate']['nonempty_games'], 2)
        self.assertEqual(result['equal_game_rates']['failed_move_rate']['empty_games'], 1)

    def test_all_contrasts_and_profile_regressions_are_retained(self):
        rows = [fixture(v) for v in VARIANTS]
        rows[-1]['won'] = False
        rows[0]['won'] = True
        rows[-1]['longest_observed_stall'] = 5
        rows[-1]['expired_stop_selections'] = 5
        rows[-1]['longest_unresolved_stop_run'] = 5
        rows[-1]['final_secure_progress'] = 0.
        rows[0]['final_secure_progress'] = .5
        groups, pairs, regressions = aggregate(rows)
        self.assertEqual(len(pairs), 3)
        self.assertEqual({(p['left'], p['right']) for p in pairs}, set(CONTRASTS))
        profile = next(g for g in groups if g['partition'] == 'controlled' and g['dimension'] == 'profile')
        self.assertEqual(set(profile['variants']), set(VARIANTS))
        adverse = next(r for r in regressions if r['dimension'] == 'profile' and r['left'] == 'fixed' and r['right'] == 'goal-uniform')
        self.assertIn('won decreased', adverse['reasons'])
        self.assertIn('final_secure_progress decreased', adverse['reasons'])
        self.assertIn('longest stall increased', adverse['reasons'])
        self.assertIn('expired_stop_selections increased', adverse['reasons'])
        self.assertIn('longest unresolved stop run increased', adverse['reasons'])

    def test_paired_delta_and_ties_report_direction_not_quality(self):
        left, right = fixture(), fixture('goal-uniform')
        right['final_eliminate_progress'] += .2
        pair = paired_delta(left, right)
        result = pair_summary([pair])
        self.assertAlmostEqual(result['metrics']['final_eliminate_progress']['mean_delta'], .2)
        self.assertEqual(result['metrics']['final_eliminate_progress']['higher'], 1)
        self.assertEqual(result['metrics']['won']['unchanged'], 1)
        self.assertEqual(result['rates']['failed_move_rate']['unavailable_games'], 1)

    def test_trace_off_cost_signature_ignores_only_tracing_and_timing(self):
        traced = add_samples(fixture('goal-uniform'))
        plain = copy.deepcopy(traced)
        plain['trace'][0]['planning'].pop('execution')
        plain['trace_execution'] = False
        plain['game_seconds'] = 123.
        plain['total_work']['planner_seconds'] = 120.
        self.assertEqual(semantic_signature(traced), semantic_signature(plain))
        plain['trace'][0]['choices'][0] = 'move:4:1'
        self.assertNotEqual(semantic_signature(traced), semantic_signature(plain))
        plain = copy.deepcopy(traced)
        plain['learned_uses'] += 1
        self.assertNotEqual(semantic_signature(traced), semantic_signature(plain))


if __name__ == '__main__':
    unittest.main()
