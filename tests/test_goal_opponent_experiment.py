"""Fresh preregistration, unbiased scoring domains and fail-closed live gate."""
from collections import Counter
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reflex.combat import alive, legal, battle_record, battle_from_record, resolve
from reflex.congestion_experiment import authored_world, jobs as old_jobs
from reflex.core import digest
from reflex.goal_opponent import GoalOpponentMemory, METHODS
from reflex.goal_opponent_experiment import (PARTITIONS, DOMAINS, PRIMARY_DOMAINS,
    jobs, game_id, configuration, preregister, check_frozen, run_baselines,
    evaluate, evaluate_run, verify_run, summarize, equal_game_summary,
    paired_differences, primary_gate, groups_for, binary_metrics, intent_metrics)
from reflex.laboratory import profiles


def fixture(scene='enemy_convergence', ticks=2):
    w = authored_world(scene, 'both', 0, limit=ticks)
    run = dict(genre='combat', partition='controlled', scenario=scene+'/both',
        profile='care', seed=1702, rival='reference', variant='baseline', trace=[], ticks=ticks)
    for _ in range(ticks):
        choices = {i: 'guard' for i in range(6)}
        after, audit = resolve(w, choices, int(digest(['combat-world', run['seed']])[:16], 16))
        run['trace'].append(dict(before=battle_record(w), after=battle_record(after), choices=choices, audit=audit))
        w = after
    return run


def artificial_groups(candidate=.1, control=.2):
    groups = []
    for partition in PARTITIONS:
        values = {m: {d: dict(brier=candidate if m == 'goal_adaptive' else control,
                             log_loss=candidate if m == 'goal_adaptive' else control)
                      for d in PRIMARY_DOMAINS} for m in METHODS}
        groups.append(dict(partition=partition, dimension='all', metrics=copy.deepcopy(values), equal_game=copy.deepcopy(values)))
    return groups


class GoalOpponentExperimentTests(unittest.TestCase):
    def test_fresh_complete_matrix_no_previous_ids_and_balanced_seats(self):
        work = jobs()
        self.assertEqual(len(work), 56)
        self.assertEqual(Counter(j['partition'] for j in work), dict(controlled=24, heldout=32))
        ids = {game_id(j) for j in work}
        old = {'|'.join(map(str, (p, s+'/'+g, profile, seed))) for p, s, g, _, profile, seed, v in old_jobs() if v == 'baseline'}
        self.assertEqual(len(ids), 56)
        self.assertFalse(ids & old)
        self.assertEqual({j['seed'] for j in work}, {1701, 1702})
        for p in profiles():
            for partition, n in (('controlled', 3), ('heldout', 4)):
                seats = Counter(j['seed'] % 2 for j in work if j['partition'] == partition and j['profile'] == p['id'])
                self.assertEqual(seats, {0: n, 1: n})
        for j in work:
            self.assertEqual(j['limit'], 24 if j['partition'] == 'controlled' else 40)

    def test_original_baseline_defaults_preserved(self):
        control = configuration()
        self.assertTrue(control.recovery_options)
        self.assertTrue(control.trace_execution)
        self.assertFalse(control.root_completion)
        self.assertEqual(control.self_continuation, 'objective')
        self.assertEqual((control.horizon, control.samples, control.max_plans), (6, 4, 24))

    def test_preregistration_precedes_games_and_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory, patch('reflex.goal_opponent_experiment.run_job', side_effect=AssertionError('game before freeze')):
            manifest = preregister(directory)
            self.assertEqual(len(manifest['jobs']), 56)
            self.assertIn('goal_opponent.py', manifest['source_hashes'])
            self.assertIn('tests/test_goal_opponent_experiment.py', manifest['file_hashes'])
            self.assertIn('evidence/controlled_congestion/design.json', manifest['file_hashes'])
            self.assertFalse(manifest['forecasts_enter_live_decisions'])
            self.assertEqual(manifest, check_frozen(directory))
            with self.assertRaises(FileExistsError):
                preregister(directory)
            with patch('reflex.goal_opponent_experiment.sources', return_value={}):
                with self.assertRaisesRegex(ValueError, 'changed after preregistration'):
                    check_frozen(directory)

    def test_missing_preregistration_blocks_baseline_run(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                run_baselines(directory)

    def test_partial_baselines_are_not_overwritten_or_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            preregister(directory)
            (Path(directory) / 'trajectories.jsonl').write_text('{}\n')
            with self.assertRaises(FileExistsError):
                run_baselines(directory)
            with self.assertRaises(FileNotFoundError):
                evaluate(directory)

    def test_rule_replay_complete_and_contiguous(self):
        run = fixture()
        checks = verify_run(run)
        self.assertEqual(checks['rule_transitions'], 2)
        self.assertEqual(checks['legal_intents'], 12)
        run['trace'][1]['before']['tick'] = 0
        with self.assertRaisesRegex(ValueError, 'discontinuous'):
            verify_run(run)

    def test_truncation_and_illegal_actual_intents_rejected(self):
        run = fixture()
        run['trace'].pop()
        with self.assertRaisesRegex(ValueError, 'truncated'):
            verify_run(run)
        run = fixture()
        run['trace'][0]['choices'][3] = 'move:99:99'
        with self.assertRaisesRegex(ValueError, 'illegal'):
            verify_run(run)

    def test_all_legal_cells_and_one_intent_per_opponent_not_per_observer(self):
        run = fixture(ticks=1)
        w = battle_from_record(run['trace'][0]['before'])
        occupancy, intents, beliefs, cost = evaluate_run(run)
        expected = sum(k.startswith('move:') for owner in alive(w, 0) for k in legal(w, owner))
        self.assertEqual(len(occupancy), expected)
        self.assertEqual(len(intents), 3)
        self.assertTrue(any(r['enemy_reachable'] for r in occupancy))
        self.assertTrue(any(not r['enemy_reachable'] for r in occupancy))
        self.assertTrue(all(not r['selected'] and not r['actual'] for r in occupancy))
        self.assertEqual(len(beliefs), 1)
        self.assertIn('weights', beliefs[0]['forecast'])
        self.assertIn('active', beliefs[0]['forecast'])
        self.assertEqual(cost['ticks'], 1)
        for row in intents:
            for method in METHODS:
                self.assertEqual(set(row['probabilities'][method]), set(legal(w, row['opponent'])))

    def test_forecasts_constructed_before_current_choice_access(self):
        run = fixture(ticks=1)
        observed = []
        class WatchedMemory(GoalOpponentMemory):
            def forecast(self, *args, **kwargs):
                result = super().forecast(*args, **kwargs)
                observed.append(kwargs['owner'])
                return result
        class WatchedChoices(dict):
            def items(self):
                if sorted(observed) != [0, 1, 2]:
                    raise AssertionError('current actions were read before all owner forecasts')
                return super().items()
        run['trace'][0]['choices'] = WatchedChoices(run['trace'][0]['choices'])
        with patch('reflex.goal_opponent_experiment.GoalOpponentMemory', WatchedMemory):
            evaluate_run(run)

    def test_offline_scoring_never_calls_actual_controller_or_rng(self):
        run = fixture()
        with patch('reflex.combat.opponent', side_effect=AssertionError('controller leak')), \
             patch('reflex.combat.reference', side_effect=AssertionError('controller leak')), \
             patch('reflex.combat.resolve', side_effect=AssertionError('future RNG leak')):
            occupancy, intents, _, _ = evaluate_run(run)
        self.assertGreater(len(occupancy), 0)
        self.assertGreater(len(intents), 0)

    def test_binary_loss_exact_infinity_not_hidden_by_clipping(self):
        metrics = binary_metrics([(0., 1), (1., 0), (.5, 1), (0., 0)])
        self.assertAlmostEqual(metrics['brier'], 2.25/4)
        self.assertIsNone(metrics['log_loss'])
        self.assertEqual(metrics['infinite_log_losses'], 2)
        self.assertEqual(metrics['zero_probability_misses'], 1)
        self.assertEqual(metrics['certain_false_positives'], 1)
        self.assertTrue(math.isfinite(metrics['clipped_log_loss']))
        self.assertEqual(sum(b['n'] for b in metrics['bins']), 4)

    def test_intent_brier_sums_legal_categories_before_average(self):
        rows = [dict(actual='a', probabilities={'m': {'a': .25, 'b': .75}}),
                dict(actual='b', probabilities={'m': {'a': .5, 'b': .5}})]
        metrics = intent_metrics(rows, 'm')
        self.assertAlmostEqual(metrics['brier'], (1.125+.5)/2)
        self.assertAlmostEqual(metrics['log_loss'], (-math.log(.25)-math.log(.5))/2)
        self.assertEqual(metrics['legal_action_negatives'], 2)

    def test_equal_game_mean_does_not_equal_pooled_event_weight(self):
        def row(count, p):
            o = [dict(actual=0, probabilities={m: p for m in METHODS}, enemy_reachable=True, selected=False)]*count
            return dict(metrics=summarize(o, []))
        rows = [row(1, 1.), row(9, 0.)]
        macro = equal_game_summary(rows)
        self.assertEqual(macro['goal_adaptive']['reachable_occupancy']['brier'], .5)
        self.assertIsNone(macro['goal_adaptive']['reachable_occupancy']['log_loss'])
        self.assertEqual(macro['goal_adaptive']['reachable_occupancy']['infinite_log_loss_games'], 1)
        empty = row(0, .5)
        macro = equal_game_summary([rows[0], empty])
        self.assertEqual(macro['uniform']['reachable_occupancy']['games'], 1)
        self.assertEqual(macro['uniform']['reachable_occupancy']['empty_games'], 1)
        self.assertEqual(macro['uniform']['reachable_occupancy']['brier'], 1.)

    def test_primary_gate_requires_all_domains_losses_controls_partitions_and_weights(self):
        groups = artificial_groups()
        self.assertTrue(primary_gate(groups)['primary_gate_passed'])
        for partition in range(2):
            for aggregation in ('metrics', 'equal_game'):
                for domain in PRIMARY_DOMAINS:
                    for metric in ('brier', 'log_loss'):
                        changed = copy.deepcopy(groups)
                        changed[partition][aggregation]['goal_adaptive'][domain][metric] = .2
                        self.assertFalse(primary_gate(changed)['primary_gate_passed'])
                        changed[partition][aggregation]['goal_adaptive'][domain][metric] = None
                        self.assertFalse(primary_gate(changed)['primary_gate_passed'])
        self.assertFalse(primary_gate(groups)['live_gameplay_performed'])
        self.assertFalse(primary_gate(groups)['integrate'])

    def test_equal_goal_ablation_does_not_get_conflated_with_primary_gate(self):
        groups = artificial_groups()
        for group in groups:
            for aggregation in ('metrics', 'equal_game'):
                for domain in PRIMARY_DOMAINS:
                    group[aggregation]['goal_uniform'][domain] = dict(brier=0., log_loss=0.)
        self.assertTrue(primary_gate(groups)['primary_gate_passed'])

    def test_groups_retain_every_profile_scenario_and_adaptive_ablation(self):
        run = fixture(ticks=1)
        o, i, _, _ = evaluate_run(run)
        per_game = [dict(game=game_id(run), partition=run['partition'], profile=run['profile'],
            scenario=run['scenario'], rival=run['rival'], metrics=summarize(o, i))]
        groups = groups_for(o, i, per_game)
        for dimension, value in [('profile', 'care'), ('scenario', run['scenario']), ('rival', 'reference')]:
            group = next(g for g in groups if g['partition'] == 'controlled' and g['dimension'] == dimension and g['value'] == value)
            self.assertIn('goal_uniform', group['adaptive_differences'])
            self.assertEqual(group['equal_game']['uniform']['intent']['games'], 1)
        deltas = paired_differences(per_game, 'goal_uniform')
        self.assertAlmostEqual(deltas['intent']['brier']['mean_delta'], 0.)


if __name__ == '__main__':
    unittest.main()
