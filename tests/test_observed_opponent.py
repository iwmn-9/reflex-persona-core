"""Offline public behavior inference boundaries and complete scoring domains."""
import copy
from dataclasses import replace
import json
import math
import unittest
from unittest.mock import patch
from reflex.combat import Battle, alive, legal, battle_record, resolve
from reflex.congestion_experiment import authored_world
from reflex.observed_opponent import PublicOpponentMemory, IntentForecast, METHODS, public_features
from reflex.observed_opponent_experiment import binary_metrics, intent_metrics, evaluate_run


def memory(w=None, owner=0, scope='game'):
    w = w or Battle.start('open', 'both', limit=100)
    return w, PublicOpponentMemory(scope, owner, alive(w, 1-w.units[owner].team))


def forecast(m, w, scope='game', owner=0):
    return m.forecast(w, scope=scope, owner=owner)


def reveal(m, w, actions=None, scope='game', owner=0):
    return m.reveal(w, actions or {a: 'guard' for a in alive(w, 1-w.units[owner].team)}, scope=scope, owner=owner)


class ObservedOpponentTests(unittest.TestCase):
    def test_cold_predictions_normalized_legal_and_full_support(self):
        for scene in ('friendly_convergence', 'enemy_convergence', 'mixed_junction'):
            for team in (0, 1):
                w = authored_world(scene, 'both', team)
                owner = alive(w, team)[0]
                _, m = memory(w, owner)
                f = forecast(m, w, owner=owner)
                for actor in alive(w, 1-team):
                    for method in METHODS:
                        p = f.probabilities(method, actor)
                        self.assertEqual(set(p), set(legal(w, actor)))
                        self.assertAlmostEqual(sum(p.values()), 1.)
                        if method != 'fixed':
                            self.assertTrue(all(v > 0 for v in p.values()))
                    for k, p in f.probabilities('cold_memory', actor).items():
                        self.assertAlmostEqual(p, f.probabilities('observed_memory', actor)[k])

    def test_complete_fixed_joint_dependence_not_product_of_marginals(self):
        f = IntentForecast('game', 0, 0, 'state', (((3, 'move:4:1'), (4, 'move:4:1')), ((3, 'guard'), (4, 'guard'))),
            ((3, (('guard', 1.),)), (4, (('guard', 1.),))), ((3, (('guard', 1.),)), (4, (('guard', 1.),))))
        self.assertEqual(f.occupancy('fixed', 'move:4:1'), .5)
        self.assertNotEqual(f.occupancy('fixed', 'move:4:1'), .75)

    def test_fixed_support_cannot_be_restored_by_reweighting(self):
        w = authored_world('enemy_convergence', 'both', 0)
        _, m = memory(w)
        f = forecast(m, w)
        self.assertEqual(f.occupancy('fixed', 'move:4:1'), 0.)
        self.assertGreater(f.occupancy('fixed_coverage', 'move:4:1'), 0.)
        for weight in (0., .3, 1.):
            self.assertEqual(sum(v*('move:4:1' in dict(h).values()) for v, h in zip((weight, 1-weight), f.fixed)), 0.)

    def test_forecast_never_calls_actual_policy_or_resolution_rng(self):
        w, m = memory()
        with patch('reflex.combat.opponent', side_effect=AssertionError('controller leak')), patch('reflex.combat.reference', side_effect=AssertionError('controller leak')), patch('reflex.combat.resolve', side_effect=AssertionError('RNG leak')):
            f = forecast(m, w)
            reveal(m, w)
            forecast(m, replace(w, tick=1))
        self.assertEqual(f.tick, 0)

    def test_forecast_snapshot_stays_immutable_after_reveal(self):
        w, m = memory()
        first = forecast(m, w)
        saved = copy.deepcopy(first)
        self.assertEqual(first, forecast(m, w))
        reveal(m, w)
        self.assertEqual(first, saved)
        self.assertNotEqual(first.probabilities('observed_memory', 3), forecast(m, replace(w, tick=1)).probabilities('observed_memory', 3))

    def test_owner_game_and_opponent_isolation(self):
        w, a = memory()
        _, same_owner_other_game = memory(scope='other')
        _, other_owner = memory(owner=1)
        before_a, before_b = same_owner_other_game.record(), other_owner.record()
        forecast(a, w)
        reveal(a, w, {3: 'guard', 4: 'move:6:2', 5: 'guard'})
        self.assertEqual(same_owner_other_game.record(), before_a)
        self.assertEqual(other_owner.record(), before_b)
        counts = a.record()['counts']
        self.assertNotEqual(counts['3'], counts['4'])
        self.assertEqual(counts['3'], counts['5'])
        for kwargs in ({'scope': 'other'}, {'owner': 1}):
            with self.assertRaises(ValueError): forecast(a, replace(w, tick=1), **kwargs)
        bad = replace(w, units=tuple(replace(u, team=0) if i == 3 else u for i, u in enumerate(w.units)))
        with self.assertRaises(ValueError): forecast(same_owner_other_game, bad, scope='other')

    def test_atomic_reject_invalid_or_partial_reveals(self):
        w, m = memory(); forecast(m, w)
        before = m.record()
        for actions in ({3: 'guard'}, {3: 'guard', 4: 'move:99:99', 5: 'guard'}, {3: 'guard', 4: 'guard', 5: 'guard', 0: 'guard'}):
            with self.assertRaises(ValueError): reveal(m, w, actions)
            self.assertEqual(m.record(), before)
        reveal(m, w)

    def test_no_reveal_before_forecast_or_duplicate_or_future_tick(self):
        w, m = memory()
        with self.assertRaises(ValueError): reveal(m, w)
        with self.assertRaises(ValueError): forecast(m, replace(w, tick=1))
        forecast(m, w)
        with self.assertRaises(ValueError): reveal(m, replace(w, tick=1))
        reveal(m, w)
        with self.assertRaises(ValueError): reveal(m, w)
        with self.assertRaises(ValueError): forecast(m, w)
        with self.assertRaises(ValueError): forecast(m, replace(w, tick=2))

    def test_pending_state_mismatch_and_detached_record(self):
        w, m = memory(); forecast(m, w)
        changed = replace(w, hold=(1, 0))
        before = m.record()
        with self.assertRaises(ValueError): reveal(m, changed)
        with self.assertRaises(ValueError): forecast(m, changed)
        self.assertEqual(m.record(), before)
        before['counts']['3'][0][0] = 123
        self.assertNotEqual(m.record(), before)

    def test_repeated_tactic_phase_reversal_and_recurrence(self):
        w, m = memory()
        w = replace(w, units=tuple(replace(u, ammo=1) for u in w.units))
        probabilities = []
        for tick in range(48):
            before = replace(w, tick=tick)
            f = forecast(m, before)
            probabilities.append(f.probabilities('observed_memory', 3))
            chosen = 'guard' if tick < 16 or tick >= 32 else 'reload'
            reveal(m, before, {a: chosen for a in (3, 4, 5)})
        self.assertGreater(probabilities[15]['guard'], probabilities[0]['guard'])
        self.assertGreater(probabilities[31]['reload'], probabilities[16]['reload'])
        self.assertLess(probabilities[31]['guard'], probabilities[16]['guard'])
        self.assertGreater(probabilities[47]['guard'], probabilities[32]['guard'])
        self.assertLessEqual(max(sum(sum(r) for r in matrix) for matrix in m.record()['counts'].values()), 10.+1e-12)
        self.assertEqual(sum(len(r) for matrix in m.record()['counts'].values() for r in matrix), 312)
        self.assertTrue(all(0 < p <= 1 for row in probabilities for p in row.values()))

    def test_only_observed_action_not_success_or_persona_is_learned(self):
        w, a = memory(); _, b = memory()
        for tick in range(8):
            before = replace(w, tick=tick)
            self.assertEqual(forecast(a, before), forecast(b, before))
            # No after-state/reward/persona field exists in the observation API.
            reveal(a, before); reveal(b, before)
        self.assertEqual(a.record(), b.record())
        with self.assertRaises(TypeError): a.reveal(replace(w, tick=8), {3: 'guard', 4: 'guard', 5: 'guard'}, scope='game', owner=0, success=True)

    def test_geometry_context_and_signature_bounds(self):
        for scene in ('friendly_convergence', 'enemy_convergence', 'mixed_junction'):
            for team in (0, 1):
                w = authored_world(scene, 'both', team)
                for a in range(6):
                    c, signatures = public_features(w, a)
                    self.assertTrue(0 <= c < 8)
                    self.assertTrue(all(0 <= s < 13 for s in signatures.values()))
                    self.assertEqual(set(signatures), set(legal(w, a)))

    def test_all_legal_occupancy_negatives_counted_not_only_selected(self):
        w = authored_world('mixed_junction', 'both', 0)
        choices = {a: 'guard' for a in range(6)}
        choices[3] = 'move:4:2'
        after, audit = resolve(w, choices, 91)
        run = dict(seed=592, partition='controlled', scenario='fixture/both', profile='care', rival='reference',
            trace=[dict(before=battle_record(w), after=battle_record(after), audit=audit, choices=choices)])
        o, i, _ = evaluate_run(run)
        expected = [(a, k) for a in alive(w, 0) for k in legal(w, a) if k.startswith('move:')]
        self.assertEqual(sorted((r['owner'], r['action']) for r in o), sorted(expected))
        self.assertEqual(len(i), 3)
        self.assertTrue(any(not r['actual'] for r in o))
        self.assertTrue(any(not r['enemy_reachable'] for r in o))
        self.assertTrue(any(r['actual'] for r in o))
        self.assertTrue(all(not r['selected'] for r in o))

    def test_binary_zero_support_infinity_is_not_hidden_by_clipping(self):
        result = binary_metrics([(0., 1), (1., 0), (.5, 1), (0., 0)])
        self.assertEqual(result['infinite_log_losses'], 2)
        self.assertIsNone(result['log_loss'])
        self.assertEqual(result['zero_probability_misses'], 1)
        self.assertEqual(result['certain_false_positives'], 1)
        self.assertAlmostEqual(result['brier'], 2.25/4)
        self.assertEqual(sum(b['n'] for b in result['bins']), 4)
        self.assertTrue(math.isfinite(result['clipped_log_loss']))
        self.assertIsNone(binary_metrics([])['brier'])

    def test_categorical_brier_uses_all_legal_negatives(self):
        r = intent_metrics([dict(actual='a', probabilities={'m': dict(a=.5, b=.3, c=.2)})], 'm')
        self.assertAlmostEqual(r['brier'], .25+.09+.04)
        self.assertAlmostEqual(r['log_loss'], math.log(2))
        self.assertEqual(r['legal_action_negatives'], 2)
        self.assertEqual(r['calibration']['n'], 3)

    def test_current_reveal_cannot_change_its_own_forecast(self):
        w, a = memory(); _, b = memory()
        fa, fb = forecast(a, w), forecast(b, w)
        reveal(a, w)
        reveal(b, w, {3: 'move:6:0', 4: 'move:6:2', 5: 'move:6:4'})
        self.assertEqual(fa, fb)
        self.assertNotEqual(forecast(a, replace(w, tick=1)), forecast(b, replace(w, tick=1)))

    def test_bounds_reject_invalid_registration(self):
        for kwargs in [dict(scope=''), dict(scope='a'*129), dict(owner=True), dict(opponents=(3, 3)), dict(opponents=(1, 2, 3, 4)), dict(opponents=(0,))]:
            args = dict(scope='game', owner=0, opponents=(3, 4, 5)); args.update(kwargs)
            with self.assertRaises(ValueError): PublicOpponentMemory(**args)


if __name__ == '__main__':
    unittest.main()
