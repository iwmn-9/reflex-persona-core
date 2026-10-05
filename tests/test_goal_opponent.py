"""Public-time, owner isolation, diversity and cross-game inference contracts."""
import copy
from dataclasses import replace
import math
import unittest
from unittest.mock import patch
from reflex.combat import Battle, alive, battle_record, legal
from reflex.congestion_experiment import authored_world
from reflex.goal_beliefs import GoalBeliefs, MAX_ACTIONS
from reflex.goal_opponent import GoalOpponentMemory, METHODS, goal_models


def make(w=None, owner=0, scope='game'):
    w = w or Battle.start('open', 'both', limit=100)
    return w, GoalOpponentMemory(scope, owner, alive(w, 1-w.units[owner].team))


def predict(m, w, owner=0, scope='game'):
    return m.forecast(w, scope=scope, owner=owner)


def reveal(m, w, actions=None, owner=0, scope='game'):
    actions = actions if actions is not None else {a: 'guard' for a in alive(w, 1-w.units[owner].team)}
    return m.reveal(w, actions, scope=scope, owner=owner)


class GoalOpponentTests(unittest.TestCase):
    def test_legal_normalized_full_support_and_equal_cold(self):
        for scene in ('friendly_convergence', 'enemy_convergence', 'mixed_junction'):
            for team in (0, 1):
                w = authored_world(scene, 'both', team)
                owner = alive(w, team)[0]
                _, memory = make(w, owner)
                f = predict(memory, w, owner)
                for actor in alive(w, 1-team):
                    for method in METHODS:
                        p = f.probabilities(method, actor)
                        self.assertEqual(set(p), set(legal(w, actor)))
                        self.assertAlmostEqual(sum(p.values()), 1.)
                        self.assertTrue(all(math.isfinite(x) and 0 <= x <= 1 for x in p.values()))
                        if method != 'fixed': self.assertTrue(all(x > 0 for x in p.values()))
                    self.assertEqual(f.probabilities('goal_uniform', actor), f.probabilities('goal_adaptive', actor))

    def test_public_goal_diversity_restores_central_move_beyond_coverage(self):
        w, m = make(authored_world('enemy_convergence', 'both', 0))
        f = predict(m, w)
        self.assertEqual(f.occupancy('fixed', 'move:4:1'), 0.)
        self.assertGreater(f.occupancy('goal_uniform', 'move:4:1'), f.occupancy('fixed_coverage', 'move:4:1'))
        self.assertNotEqual(goal_models(w, 3)['secure'], goal_models(w, 3)['eliminate'])

    def test_later_repeated_move_changes_goal_mass_only_for_that_opponent(self):
        w, m = make(authored_world('enemy_convergence', 'both', 0))
        probabilities = []
        for tick in range(8):
            before = replace(w, tick=tick)
            f = predict(m, before)
            probabilities.append(f.occupancy('goal_adaptive', 'move:4:1'))
            reveal(m, before, {3: 'move:4:1', 4: 'guard', 5: 'guard'})
        self.assertGreater(probabilities[-1], probabilities[0])
        r = m.record()['actors']
        self.assertGreater(r['3']['weights']['secure'], .5)
        self.assertNotEqual(r['3']['weights'], r['4']['weights'])
        self.assertEqual(f.probabilities('goal_uniform', 3), predict(make(w)[1], w).probabilities('goal_uniform', 3))

    def test_no_controller_rng_persona_or_future_outcome_access(self):
        w, m = make()
        with patch('reflex.combat.opponent', side_effect=AssertionError('actual opponent leak')), patch('reflex.combat.reference', side_effect=AssertionError('reference leak')), patch('reflex.combat.resolve', side_effect=AssertionError('world/RNG leak')):
            f = predict(m, w)
            reveal(m, w)
            predict(m, replace(w, tick=1))
        for extra in ('rival', 'seed', 'success', 'after', 'persona'):
            with self.assertRaises(TypeError): m.forecast(w, scope='game', owner=0, **{extra: 1})
        self.assertEqual(f.tick, 0)

    def test_fixed_joint_and_goal_product_have_distinct_declared_semantics(self):
        w, m = make()
        f = predict(m, w)
        # Same two enemies act together in half of F: occupancy .5, not .75.
        fixture = replace(f, fixed=(((3, 'move:4:1'), (4, 'move:4:1'), (5, 'guard')),
                                   ((3, 'guard'), (4, 'guard'), (5, 'guard'))))
        self.assertEqual(fixture.occupancy('fixed', 'move:4:1'), .5)
        for method in ('goal_uniform', 'goal_adaptive'):
            goal = 1-math.prod(1-f.goals.probabilities(str(a), adaptive=method=='goal_adaptive').get('move:6:1', 0.) for a in (3, 4, 5))
            uniform = 1-math.prod(1-dict(row).get('move:6:1', 0.) for _, row in f.uniform)
            self.assertAlmostEqual(f.occupancy(method, 'move:6:1'), .8*goal+.2*uniform)

    def test_owner_game_opponent_isolation_and_detached_records(self):
        w, m = make(); _, other_game = make(scope='other'); _, other_owner = make(owner=1)
        old = other_game.record(), other_owner.record()
        predict(m, w); reveal(m, w, {3: 'move:6:0', 4: 'guard', 5: 'guard'})
        self.assertEqual(old, (other_game.record(), other_owner.record()))
        record = m.record(); record['actors']['3']['logs'][0] = -1234
        self.assertNotEqual(record, m.record())
        for kwargs in (dict(scope='other'), dict(owner=1), dict(owner=True)):
            with self.assertRaises(ValueError): predict(m, replace(w, tick=1), **kwargs)

    def test_forecast_immutable_and_reveal_cannot_change_current_prediction(self):
        w, a = make(); _, b = make()
        fa, fb = predict(a, w), predict(b, w)
        before = copy.deepcopy(fa)
        self.assertEqual(fa, predict(a, w))
        reveal(a, w); reveal(b, w, {3: 'move:6:0', 4: 'move:6:2', 5: 'move:6:4'})
        self.assertEqual(fa, fb); self.assertEqual(fa, before)
        self.assertNotEqual(predict(a, replace(w, tick=1)), predict(b, replace(w, tick=1)))

    def test_atomic_rejection_partial_invalid_or_wrong_state(self):
        w, m = make(); predict(m, w)
        before = m.record()
        for actions in ({3: 'guard'}, {3: 'guard', 4: 'illegal', 5: 'guard'}, {0: 'guard', 3: 'guard', 4: 'guard', 5: 'guard'}):
            with self.assertRaises(ValueError): reveal(m, w, actions)
            self.assertEqual(before, m.record())
        changed = replace(w, hold=(1, 0))
        with self.assertRaises(ValueError): reveal(m, changed)
        with self.assertRaises(ValueError): predict(m, changed)
        self.assertEqual(before, m.record()); reveal(m, w)

    def test_atomic_rollback_unexpected_tracker_failure(self):
        w, m = make(); predict(m, w); before = m.record()
        count = 0
        # Use class hook; the first copy updates before the second throws.
        from reflex.opponent_beliefs import HypothesisTracker
        original_class = HypothesisTracker.observe
        def fail(self, *args, **kwargs):
            nonlocal count
            count += 1
            if count == 2: raise RuntimeError('injected failure')
            return original_class(self, *args, **kwargs)
        with patch.object(HypothesisTracker, 'observe', fail):
            with self.assertRaises(RuntimeError): reveal(m, w)
        self.assertEqual(before, m.record()); reveal(m, w)

    def test_no_duplicate_skipped_or_unpredicted_observation(self):
        w, m = make()
        with self.assertRaises(ValueError): reveal(m, w)
        with self.assertRaises(ValueError): predict(m, replace(w, tick=1))
        predict(m, w)
        with self.assertRaises(ValueError): reveal(m, replace(w, tick=1))
        reveal(m, w)
        for tick in (0, 2):
            with self.assertRaises(ValueError): predict(m, replace(w, tick=tick))
        with self.assertRaises(ValueError): reveal(m, w)

    def test_known_completed_goal_mask_without_hidden_goal(self):
        w, m = make(); w = replace(w, secured=(False, True))
        f = predict(m, w)
        self.assertEqual(set(tuple(x) for x in f.record()['active'].values()), {('eliminate',)})
        for a in (3, 4, 5): self.assertEqual(f.probabilities('goal_adaptive', a), f.probabilities('goal_uniform', a))

    def test_dead_observer_and_opponent_changes_rejected(self):
        w, m = make()
        dead = replace(w, units=(replace(w.units[0], hp=0),)+w.units[1:])
        with self.assertRaises(ValueError): predict(m, dead)
        wrong = replace(w, units=w.units[:3]+(replace(w.units[3], team=0),)+w.units[4:])
        with self.assertRaises(ValueError): predict(m, wrong)

    def test_no_live_default_import_or_configuration_change(self):
        from reflex.combat_planning import TacticalControl
        self.assertFalse(TacticalControl().root_completion)
        import inspect
        from reflex import core, runtime, decision_loop, combat_planning
        for module in (core, runtime, decision_loop, combat_planning):
            self.assertNotIn('goal_opponent', inspect.getsource(module))

    def test_invalid_registration_or_methods(self):
        for kwargs in (dict(scope=''), dict(owner=True), dict(opponents=(3, 3)), dict(opponents=(0,)), dict(opponents=(3, 4, 5, 6))):
            args = dict(scope='game', owner=0, opponents=(3, 4, 5)); args.update(kwargs)
            with self.assertRaises(ValueError): GoalOpponentMemory(**args)
        w, m = make(); f = predict(m, w)
        with self.assertRaises(ValueError): f.probabilities('unknown', 3)
        with self.assertRaises(ValueError): f.probabilities('uniform', 0)
        with self.assertRaises(ValueError): f.occupancy('uniform', 'guard')


class GenericGoalBeliefTests(unittest.TestCase):
    def test_noncombat_two_actor_adapter_adapts_reverses_and_reuses(self):
        belief = GoalBeliefs('market', 'observer', ('alice', 'bob'), ('save', 'spend'))
        models = {a: {'save': {'pass': .9, 'buy': .1}, 'spend': {'pass': .1, 'buy': .9}} for a in ('alice', 'bob')}
        rows = []
        for tick in range(300):
            f = belief.forecast(tick, str(tick), models, active={a: ('save', 'spend') for a in models}, scope='market', owner='observer')
            rows.append(f.probabilities('alice')['pass'])
            choice = 'pass' if tick < 100 or tick >= 200 else 'buy'
            belief.reveal(tick, str(tick), {'alice': choice, 'bob': 'buy'}, scope='market', owner='observer')
        self.assertGreater(rows[99], rows[0]); self.assertLess(rows[199], rows[100]); self.assertGreater(rows[299], rows[200])
        for a in belief.record()['actors'].values():
            self.assertLessEqual(a['retained_ids'], 64); self.assertLessEqual(a['retained_lifts'], 6)
            self.assertTrue(all(-80 <= n <= 0 for n in a['logs']))
            self.assertAlmostEqual(sum(a['weights'].values()), 1.)
        self.assertLess(belief.record()['actors']['bob']['weights']['save'], .01)

    def test_forced_action_is_not_goal_evidence(self):
        b = GoalBeliefs('g', 'o', ('a',), ('one', 'two'))
        models = {'a': {'one': {'forced': 1.}, 'two': {'forced': 1.}}}
        b.forecast(0, 'state', models, active={'a': ('one', 'two')}, scope='g', owner='o')
        b.reveal(0, 'state', {'a': 'forced'}, scope='g', owner='o')
        self.assertEqual(b.record()['actors']['a']['observations'], 0)
        self.assertEqual(b.record()['actors']['a']['weights'], {'one': .5, 'two': .5})

    def test_invalid_forecast_inputs_preserve_state(self):
        b = GoalBeliefs('g', 'o', ('a',), ('one', 'two')); old = b.record()
        invalid = [ {'one': {'x': float('nan')}, 'two': {'x': 1.}},
                    {'one': {'x': True}, 'two': {'x': 1.}},
                    {'one': {'x': 1.}, 'two': {'y': 1.}},
                    {'one': {'x': .5}, 'two': {'x': 1.}},
                    {g: {str(i): 1/(MAX_ACTIONS+1) for i in range(MAX_ACTIONS+1)} for g in ('one', 'two')} ]
        for models in invalid:
            with self.assertRaises(ValueError): b.forecast(0, 'state', {'a': models}, active={'a': ('one', 'two')}, scope='g', owner='o')
            self.assertEqual(old, b.record())

    def test_pending_forecast_models_are_frozen(self):
        b = GoalBeliefs('g', 'o', ('a',), ('one', 'two'))
        models = {'a': {'one': {'x': .9, 'y': .1}, 'two': {'x': .1, 'y': .9}}}
        f = b.forecast(0, 'state', models, active={'a': ('one', 'two')}, scope='g', owner='o')
        before = f.probabilities('a')
        models['a']['one'] = {'x': .1, 'y': .9}
        self.assertEqual(before, f.probabilities('a'))
        with self.assertRaises(ValueError): b.forecast(0, 'state', models, active={'a': ('one', 'two')}, scope='g', owner='o')
        b.reveal(0, 'state', {'a': 'x'}, scope='g', owner='o')
        self.assertGreater(b.record()['actors']['a']['weights']['one'], .5)


if __name__ == '__main__': unittest.main()
