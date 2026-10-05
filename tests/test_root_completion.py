"""Contract checks for declined-plan friendly root completion, not new seeds."""
import copy
from contextlib import ExitStack
from dataclasses import replace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from reflex.combat import Battle, alive, legal, make_context
from reflex.combat_planning import TacticalControl, forecast, no_own_collision
from reflex.core import FEATURES, Policy, compile_batch, digest
from reflex.decision_loop import DecisionLoop, Request
from reflex.deliberation import JointForecast, select
from reflex.examples import action, context, effect
from reflex.judgment import Binding, avoid_waste
from reflex.laboratory import profiles
from reflex.planning import vector
from reflex.progress import Activity, ProgressConfig, ProgressWatch, PurposeFeedback, PurposeRequest
from reflex.root_completion import combat_completion, constrained_completion


def fixture(world=None):
    """Two near-tied colliders and a third owner whose intention must be frozen."""
    world = world or Battle.start('open', 'either', limit=3)
    actors = alive(world, 0)
    contexts = []
    for actor in actors:
        c = make_context(world, actor, profiles()[1], 282, 'secure', survival_security=True)
        c['personality'] = {k: .5 for k in c['personality']}
        c['values'] = {k: (.8 if k == 'security' else 0.) for k in c['values']}
        for need in c['needs'].values():
            need.update(enabled=True, supported=True, deficit=.1)
        root = 'move:1:1' if actor in (0, 1) else 'guard'
        c['actions'] = [action(k, effect(.21 if k == root else .2, values={'security': .2}))
                        for k in legal(world, actor)]
        c['state'].update(mode='principle', primary_need='physiology', mode_urgency=.1,
                          intent_action=root, age=7)
        contexts.append(c)
    records = [Policy().choose(c, False) for c in contexts]
    return world, actors, contexts, records


def run_completion(contexts, callback, records=None, allowed=None, exact=None, blocked=None):
    batch = compile_batch(contexts)
    records = records if records is not None else Policy().decide(batch, False).records(batch)
    allowed = allowed if allowed is not None else [set(ids) for ids in batch.ids]
    exact = exact if exact is not None else np.zeros(batch.legal.shape + (len(FEATURES),), dtype=bool)
    return constrained_completion(contexts, records, allowed, exact, Policy(), callback, blocked)


def request(c, purpose=None, exact=()):
    return Request(c, {a['id']: Binding(a['id'], 'public', () if exact else ('objective',))
                       for a in c['actions']},
                   {a['id']: exact for a in c['actions']}, purpose=purpose)


def declined_forecast(cs, ds):
    """One-step, incompatible owner tiers force a genuine horizon decline."""
    futures = []
    for row, c in enumerate(cs):
        fc = copy.deepcopy(c)
        preferred = 'left' if row == 0 else 'right'
        fc['actions'] = [action(k, effect(.8, values={'security': 1. if k == preferred else -1.}))
                         for k in ('left', 'right')]
        futures.append(fc)
    return JointForecast(tuple(futures),
                         {'left': tuple(d['action_id'] for d in ds),
                          'right': tuple('guard' for _ in cs)},
                         {'left': .5, 'right': .5}, 1, audit={'nodes': 0})


def planner_with(callback):
    def planner(cs, ds):
        return declined_forecast(cs, ds)
    planner.complete_fallback = callback
    return planner


class ConstrainedCompletionTests(unittest.TestCase):
    def test_original_principle_tier_cannot_be_lowered_for_coordination(self):
        c = context('tier', [action('root', effect(.1, values={'security': .5})),
                             action('tempting', effect(1., values={'security': .47}))],
                    values={'security': .8}, mode='principle')
        c['state']['primary_need'] = 'physiology'
        seen = []
        def inspect(cs, roots, options):
            seen.extend(options)
            return None, {'reason': 'no compatible completion'}
        records, audit = run_completion([c], inspect)
        self.assertEqual(set(seen[0]), {'root'})
        self.assertEqual(records[0]['action_id'], 'root')
        self.assertFalse(audit['adopted'])
        with self.assertRaisesRegex(ValueError, 'tier, band'):
            run_completion([c], lambda *_: (('tempting',), {}))

    def test_options_stay_in_original_point_025_score_band(self):
        c = context('band', [action('root', effect(.5)),
                             action('near', effect(.5 - .024 / .65)),
                             action('far', effect(.5 - .026 / .65))])
        seen = []
        def choose_near(cs, roots, options):
            seen.extend(options)
            return ('near',), {}
        records, audit = run_completion([c], choose_near)
        self.assertEqual(set(seen[0]), {'root', 'near'})
        self.assertEqual(records[0]['action_id'], 'near')
        self.assertAlmostEqual(audit['persona_regret'][0], .024)
        with self.assertRaisesRegex(ValueError, 'tier, band'):
            run_completion([c], lambda *_: (('far',), {}))

    def test_out_of_band_original_is_not_reinterpreted_or_sent_to_adapter(self):
        c = context('unusual-root', [action('best', effect(.5)), action('old', effect(.1))])
        b = compile_batch([c])
        d = Policy().decide(b, False)
        old = replace(d, action=np.array([b.ids[0].index('old')])).records(b)
        callback = Mock(side_effect=AssertionError('unsupported root reached adapter'))
        completed, audit = run_completion([c], callback, records=old)
        self.assertIs(completed, old)
        self.assertFalse(audit['adopted'])
        self.assertIn('outside', audit['reason'])
        callback.assert_not_called()

    def test_final_progress_and_exact_waste_masks_are_enforced(self):
        c = context('masks', [action('root', effect(.5)),
                              action('waste', effect(.5, cost=.01)),
                              action('expired', effect(.51)),
                              action('illegal', effect(1), legal=False),
                              action('failed', effect(1), failure=True)])
        b = compile_batch([c])
        exact = np.ones(b.legal.shape + (len(FEATURES),), dtype=bool)
        allowed = [set(b.ids[0]) - {'expired'}]
        guarded, _ = avoid_waste(replace(b, legal=b.legal & np.array([[k in allowed[0] for k in b.ids[0]]])), exact)
        records = Policy().decide(guarded, False).records(guarded)
        seen = []
        def inspect(cs, roots, options):
            seen.extend(options)
            return roots, {}
        completed, _ = run_completion([c], inspect, records, allowed, exact)
        self.assertEqual(set(seen[0]), {'root'})
        self.assertEqual(completed, records)
        for bad in ('waste', 'expired', 'illegal', 'failed'):
            with self.subTest(root=bad), self.assertRaises(ValueError):
                run_completion([c], lambda *_, bad=bad: ((bad,), {}), records, allowed, exact)

    def test_blocked_alternatives_are_filtered_after_original_scoring(self):
        c = context('blocked-band', [action('top', effect(.52)),
                                     action('root', effect(.50)),
                                     action('far', effect(.48))])
        b = compile_batch([c])
        d = Policy().decide(b, False)
        old = replace(d, action=np.array([b.ids[0].index('root')])).records(b)
        seen = []
        def inspect(cs, roots, options):
            seen.extend(options)
            return roots, {}
        completed, audit = run_completion([c], inspect, records=old, blocked=[{'top', 'root'}])
        # far is within .025 of root, but outside .025 of the ORIGINAL top.
        self.assertEqual(set(seen[0]), {'root'})
        self.assertIs(completed[0], old[0])
        self.assertFalse(audit['adopted'])
        with self.assertRaises(ValueError):
            run_completion([c], lambda *_: (('top',), {}), records=old, blocked=[{'top'}])
        with self.assertRaisesRegex(ValueError, 'one blocked-alternative set'):
            run_completion([c], inspect, records=old, blocked=[])

    def test_risky_lottery_remains_available_inside_persona_band(self):
        c = context('risk-is-not-failure', [action('root', effect(.1)),
                                           action('risky', effect(.5, p=.5), effect(-.2, p=.5))])
        b = compile_batch([c])
        exact = np.ones(b.legal.shape + (len(FEATURES),), dtype=bool)
        seen = []
        def choose_risk(cs, roots, options):
            seen.extend(options)
            return ('risky',), {}
        records, audit = run_completion([c], choose_risk, exact=exact)
        self.assertIn('risky', seen[0])
        self.assertEqual(records[0]['action_id'], 'risky')
        self.assertTrue(audit['adopted'])
        self.assertLessEqual(audit['persona_regret'][0], .025)
        self.assertEqual(len(c['actions'][1]['outcomes']), 2)

    def test_callback_input_and_metadata_are_isolated_from_caller_mutation(self):
        w, actors, cs, ds = fixture()
        b = compile_batch(cs)
        allowed = [set(ids) for ids in b.ids]
        exact = np.zeros(b.legal.shape + (len(FEATURES),), dtype=bool)
        blocked = [set() for _ in cs]
        saved = copy.deepcopy((w, cs, ds, allowed, blocked))
        metadata = {'reason': 'no completion', 'nested': [1]}
        def mutate(copies, roots, options):
            copies[0]['personality']['openness'] = 0.
            copies[1]['actions'].clear()
            options[0].clear()
            return None, metadata
        completed, audit = run_completion(cs, mutate, ds, allowed, exact, blocked)
        metadata['nested'].append(2)
        self.assertEqual((w, cs, ds, allowed, blocked), saved)
        self.assertFalse(exact.any())
        self.assertIs(completed, ds)
        self.assertEqual(audit['adapter']['nested'], [1])

    def test_callback_cannot_inject_a_new_allowed_root(self):
        _, _, cs, ds = fixture()
        before = copy.deepcopy((cs, ds))
        def malicious(copies, roots, options):
            copies[0]['actions'].append(action('injected', effect(1)))
            options[0]['injected'] = 100.
            return ('injected',) + roots[1:], {}
        with self.assertRaisesRegex(ValueError, 'tier, band'):
            run_completion(cs, malicious, ds)
        self.assertEqual((cs, ds), before)

    def test_record_lengths_owners_and_context_hashes_are_checked_before_callback(self):
        _, _, cs, ds = fixture()
        b = compile_batch(cs)
        allowed = [set(ids) for ids in b.ids]
        exact = np.zeros(b.legal.shape + (len(FEATURES),), dtype=bool)
        variants = [(cs, ds[:-1], allowed), (cs, ds, allowed[:-1]),
                    ([], [], [])]
        duplicate = copy.deepcopy(cs)
        duplicate[1]['scope'] = copy.deepcopy(duplicate[0]['scope'])
        variants.append((duplicate, ds, allowed))
        stale = copy.deepcopy(cs)
        stale[0]['state']['age'] += 1
        variants.append((stale, ds, allowed))
        swapped = [ds[1], ds[0], ds[2]]
        variants.append((cs, swapped, allowed))
        for index, (contexts, records, masks) in enumerate(variants):
            callback = Mock(side_effect=AssertionError('invalid records reached adapter'))
            with self.subTest(variant=index), self.assertRaises(ValueError):
                constrained_completion(contexts, records, masks, exact, Policy(), callback)
            callback.assert_not_called()

    def test_adapter_must_return_one_tuple_root_per_owner(self):
        _, _, cs, ds = fixture()
        roots = tuple(d['action_id'] for d in ds)
        for proposal in (list(roots), roots[:-1], roots + ('guard',)):
            with self.subTest(proposal=proposal), self.assertRaisesRegex(ValueError, 'tier, band'):
                run_completion(cs, lambda *_, proposal=proposal: (proposal, {}), ds)

    def test_independent_episode_game_or_tick_is_rejected_before_callback(self):
        for field, value in (('episode', 'other-episode'), ('game', 'other-game'), ('tick', 1)):
            with self.subTest(field=field):
                _, _, cs, ds = fixture()
                if field == 'tick':
                    cs[1]['tick'] = value
                else:
                    cs[1]['scope'][field] = value
                callback = Mock()
                with self.assertRaisesRegex(ValueError, 'independent episodes or ticks'):
                    run_completion(cs, callback, ds)
                callback.assert_not_called()


class CombatCompletionTests(unittest.TestCase):
    def test_collision_repair_changes_only_one_collider_and_freezes_noncollider(self):
        w, actors, cs, ds = fixture()
        cs[2]['state']['age'] = 1000000
        ds[2] = Policy().choose(cs[2], False)
        before = copy.deepcopy((w, cs, ds))
        roots = tuple(d['action_id'] for d in ds)
        self.assertEqual(roots, ('move:1:1', 'move:1:1', 'guard'))
        completed, audit = run_completion(cs, lambda c, r, o: combat_completion(w, actors, c, r, o), ds)
        after = tuple(d['action_id'] for d in completed)
        self.assertTrue(audit['adopted'])
        self.assertEqual(audit['changed'], 1)
        self.assertEqual(audit['adapter']['colliding'], [0, 1])
        self.assertTrue(no_own_collision(dict(zip(actors, after))))
        self.assertIs(completed[2], ds[2])
        self.assertEqual(after[2], roots[2])
        self.assertEqual((w, cs, ds), before)
        for c, d, old, regret in zip(cs, completed, roots, audit['persona_regret']):
            self.assertEqual(d['context_hash'], digest(c))
            self.assertEqual(d['next_state']['intent_action'], d['action_id'])
            self.assertEqual(d['next_state']['age'], min(c['state']['age'] + 1, 1000000) if d['action_id'] == old else 0)
            self.assertLessEqual(regret, .025)

    def test_irreducible_collision_preserves_original_highest_value_tier(self):
        w, actors, cs, _ = fixture()
        for row, c in enumerate(cs):
            preferred = 'move:1:1' if row < 2 else 'guard'
            for a in c['actions']:
                a['outcomes'][0]['values']['security'] = 1. if a['id'] == preferred else -1.
        ds = [Policy().choose(c, False) for c in cs]
        completed, audit = run_completion(cs, lambda c, r, o: combat_completion(w, actors, c, r, o), ds)
        self.assertIs(completed, ds)
        self.assertFalse(audit['adopted'])
        self.assertEqual(audit['adapter']['feasible'], 0)
        self.assertIn('no compatible', audit['reason'])
        self.assertEqual(audit['before'], audit['after'])

    def test_noncollider_is_frozen_even_when_its_move_blocks_the_only_repair(self):
        w, actors, cs, _ = fixture()
        roots = ('move:1:1', 'move:1:1', 'move:1:3')
        options = [{'move:1:1': 1.}, {'move:1:1': 1., 'move:1:3': 1.},
                   {'move:1:3': 1., 'guard': 1.}]
        proposal, audit = combat_completion(w, actors, cs, roots, options)
        self.assertIsNone(proposal)
        self.assertEqual(audit['colliding'], [0, 1])
        self.assertEqual(audit['feasible'], 0)

    def test_no_collision_does_not_trigger_other_changes(self):
        w, actors, cs, _ = fixture()
        roots = ('guard', 'guard', 'guard')
        options = [{k: 10. if k.startswith('move:') else 0. for k in legal(w, i)} for i in actors]
        proposal, audit = combat_completion(w, actors, cs, roots, options)
        self.assertIsNone(proposal)
        self.assertEqual(audit['reason'], 'no friendly collision')
        self.assertEqual(audit['combinations'], 0)

    def test_explicit_bounded_same_team_group_and_owner_are_required(self):
        w, actors, cs, ds = fixture()
        roots = tuple(d['action_id'] for d in ds)
        options = [{k: 0. for k in legal(w, i)} for i in actors]
        for group in ((), (0, 1), (1, 0, 2), (0, 1, 3), (0, 0, 2), tuple(range(6))):
            with self.subTest(actors=group), self.assertRaises(ValueError):
                combat_completion(w, group, cs, roots, options)
        for change in ('owner', 'episode', 'seed', 'world', 'tick'):
            other = copy.deepcopy(cs)
            if change == 'owner':
                other[1]['scope']['npc'] = 'unit-0'
            elif change == 'episode':
                other[1]['scope']['episode'] = 'different'
            elif change == 'seed':
                other[1]['seed'] += 1
                other[1]['scope']['episode'] = f'{w.goal}-{other[1]["seed"]}'
            elif change == 'world':
                other[1]['facts']['public_unit_5'] = 'different world'
            else:
                other[1]['tick'] += 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                combat_completion(w, actors, other, roots, options)

    def test_public_world_mismatch_or_illegal_adapter_option_fails_closed(self):
        w, actors, cs, ds = fixture()
        roots = tuple(d['action_id'] for d in ds)
        options = [{k: 0. for k in legal(w, i)} for i in actors]
        changed = replace(w, units=(replace(w.units[0], hp=8),) + w.units[1:])
        with self.assertRaisesRegex(ValueError, 'owner/public world mismatch'):
            combat_completion(changed, actors, cs, roots, options)
        options[0]['move:8:4'] = 1.
        with self.assertRaisesRegex(ValueError, 'current legal roots'):
            combat_completion(w, actors, cs, roots, options)

    def test_completion_never_reads_actual_opponent_rng_experience_or_forecasts(self):
        w, actors, cs, ds = fixture()
        forbidden = ('reflex.combat.opponent', 'reflex.combat.reference', 'reflex.combat.resolve',
                     'numpy.random.default_rng', 'reflex.combat_planning.forecast',
                     'reflex.combat_planning.tactical_joint',
                     'reflex.persona_continuation.PersonaContinuation',
                     'reflex.decision_loop.DecisionLoop.decide_batch',
                     'reflex.judgment.OutcomeMemory.prepare', 'reflex.judgment.OutcomeMemory.commit',
                     'reflex.judgment.OutcomeMemory.observe')
        with ExitStack() as stack:
            for target in forbidden:
                stack.enter_context(patch(target, side_effect=AssertionError('forbidden access: ' + target)))
            completed, audit = run_completion(cs, lambda c, r, o: combat_completion(w, actors, c, r, o), ds)
        self.assertTrue(audit['adopted'])
        self.assertTrue(no_own_collision(dict(zip(actors, (d['action_id'] for d in completed)))))


class RootCompletionIntegrationTests(unittest.TestCase):
    def test_control_requires_explicit_coordination_and_is_off_by_default(self):
        self.assertFalse(TacticalControl().root_completion)
        for kwargs in ({'coordinate': False}, {'self_continuation': 'persona-band'},
                       {'root_completion': 1}, {'root_completion': None}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, 'explicit coordination'):
                TacticalControl(**{'root_completion': True, **kwargs})
        self.assertTrue(TacticalControl(root_completion=True).coordinate)

    def test_default_off_and_explicit_off_forecasts_are_identical(self):
        w, actors, cs, ds = fixture()
        control = TacticalControl(horizon=1, samples=1, max_plans=2)
        with patch('reflex.root_completion.combat_completion', side_effect=AssertionError('default enabled')):
            implicit = forecast(w, actors, cs, ds, control)
            explicit = forecast(w, actors, cs, ds, replace(control, root_completion=False))
        self.assertEqual(implicit, explicit)
        self.assertLessEqual(implicit.audit['nodes'], 2)

    def test_completion_flag_does_not_change_the_objective_horizon_forecast(self):
        w, actors, cs, ds = fixture()
        control = TacticalControl(horizon=1, samples=1, max_plans=2)
        plain = forecast(w, actors, cs, ds, control)
        enabled = forecast(w, actors, cs, ds, replace(control, root_completion=True))
        self.assertEqual((plain.contexts, plain.roots, plain.purpose),
                         (enabled.contexts, enabled.roots, enabled.purpose))
        self.assertEqual(plain.audit['nodes'], enabled.audit['nodes'])
        self.assertEqual(plain.audit['plans'], enabled.audit['plans'])
        self.assertNotIn('self_model', enabled.audit)

    def test_separate_reader_or_predictor_cannot_bypass_per_root_validation(self):
        class Reader:
            known_conditionals = False
            def __init__(self, scope):
                self.owner = digest(scope)
            def __call__(self, context, budget):
                return None
            def record(self):
                return {'owner': self.owner}
        for mode in ('request-reader', 'loop-predictor'):
            with self.subTest(mode=mode):
                _, _, cs, _ = fixture()
                loops = [DecisionLoop(c, predictor=Reader(c['scope']) if mode == 'loop-predictor' else None)
                         for c in cs]
                requests = [request(c) for c in cs]
                if mode == 'request-reader':
                    requests[-1] = replace(requests[-1], reader=Reader(cs[-1]['scope']))
                saved = [loop.record() for loop in loops]
                callback = Mock(side_effect=AssertionError('reader bypass reached completion'))
                with self.assertRaisesRegex(ValueError, 'without a separate reader'):
                    DecisionLoop.decide_batch(list(zip(loops, requests)), stochastic=False,
                                              planner=planner_with(callback))
                self.assertEqual([loop.record() for loop in loops], saved)
                callback.assert_not_called()

    def test_declined_planner_commits_completed_root_and_owner_scoped_immediate_trial(self):
        w, actors, cs, ds = fixture()
        loops = [DecisionLoop(c) for c in cs]
        planner = planner_with(lambda c, r, o: combat_completion(w, actors, c, r, o))
        self.assertIsNone(select(cs, planner(cs, ds))[0])
        results = DecisionLoop.decide_batch([(loop, request(c)) for loop, c in zip(loops, cs)],
                                           stochastic=False, planner=planner)
        self.assertTrue(results[0]['root_completion']['adopted'])
        self.assertTrue(all(not r['deliberation']['adopted'] for r in results))
        self.assertEqual(len({r['ticket'] for r in results}), len(loops))
        for loop, c, result, old in zip(loops, cs, results, ds):
            decision = result['decision']
            selected = decision['action_id']
            original = next(a for a in c['actions'] if a['id'] == selected)
            self.assertEqual(loop.pending['action'], selected)
            self.assertEqual(loop.pending['binding'].method, selected)
            self.assertEqual(loop.pending['prior'], original['outcomes'])
            self.assertEqual(loop.pending['candidate'], original['outcomes'])
            self.assertEqual(loop.memory.pending['action'], selected)
            self.assertEqual(loop.state, decision['next_state'])
            self.assertEqual(loop.state['age'], 8 if selected == old['action_id'] else 0)
            self.assertFalse(loop.memory.entries)
        with self.assertRaisesRegex(ValueError, 'matching outstanding ticket'):
            loops[0].observe(results[1]['ticket'], vector(effect(-.1)))
        for row, (loop, result) in enumerate(zip(loops, results)):
            observed = -.1 * (row + 1)
            loop.observe(result['ticket'], vector(effect(observed)))
            entry = loop.memory.entries[(result['decision']['action_id'], 'public')]
            self.assertEqual(entry['count'], 1)
            self.assertAlmostEqual(entry['samples'][0][0], observed)
            self.assertEqual(DecisionLoop.from_record(cs[row], loop.record()).record(), loop.record())

    def test_decision_loop_passes_exact_waste_mask_to_completion(self):
        w, actors, cs, _ = fixture()
        for c in cs[:2]:
            next(a for a in c['actions'] if a['id'] == 'guard')['outcomes'] = [
                effect(.21, values={'security': .2}, cost=.01)]
        loops = [DecisionLoop(c) for c in cs]
        seen = []
        def complete(contexts, roots, options):
            seen.extend(options)
            return combat_completion(w, actors, contexts, roots, options)
        results = DecisionLoop.decide_batch(
            [(loop, request(c, exact=FEATURES)) for loop, c in zip(loops, cs)],
            stochastic=False, planner=planner_with(complete))
        self.assertTrue(results[0]['root_completion']['adopted'])
        for row in (0, 1):
            self.assertIn('guard', results[row]['waste_removed'])
            self.assertNotIn('guard', seen[row])
            self.assertNotEqual(results[row]['decision']['action_id'], 'guard')

    def test_malicious_callback_aborts_all_actor_state_and_memory_atomically(self):
        _, _, cs, _ = fixture()
        loops = [DecisionLoop(c) for c in cs]
        saved = [loop.record() for loop in loops]
        inputs = copy.deepcopy(cs)
        def malicious(copies, roots, options):
            copies[0]['state']['age'] = 999
            options[-1]['illegal'] = 100.
            return roots[:-1] + ('illegal',), {}
        with self.assertRaisesRegex(ValueError, 'tier, band'):
            DecisionLoop.decide_batch([(loop, request(c)) for loop, c in zip(loops, cs)],
                                      stochastic=False, planner=planner_with(malicious))
        self.assertEqual([loop.record() for loop in loops], saved)
        self.assertEqual(cs, inputs)
        self.assertTrue(all(loop.pending is None and loop.memory.pending is None for loop in loops))

    def test_completion_hook_runs_only_after_an_explicit_forecast_decline(self):
        for mode in ('no-hook', 'no-forecast', 'adopted'):
            with self.subTest(mode=mode):
                _, _, cs, _ = fixture()
                loops = [DecisionLoop(c) for c in cs]
                callback = Mock(side_effect=AssertionError('completion should not run'))
                if mode == 'no-hook':
                    planner = declined_forecast
                elif mode == 'no-forecast':
                    planner = lambda cs, ds: None
                    planner.complete_fallback = callback
                else:
                    def planner(cs, ds):
                        fcs = copy.deepcopy(cs)
                        for fc in fcs:
                            fc['actions'] = [action('one', effect(.5))]
                        return JointForecast(tuple(fcs), {'one': tuple(d['action_id'] for d in ds)}, {'one': .5}, 1)
                    planner.complete_fallback = callback
                results = DecisionLoop.decide_batch([(loop, request(c)) for loop, c in zip(loops, cs)],
                                                   stochastic=False, planner=planner)
                self.assertTrue(all('root_completion' not in r for r in results))
                callback.assert_not_called()

    def test_duplicate_owner_and_cross_episode_batch_cannot_be_completed(self):
        _, _, cs, _ = fixture()
        for duplicate in (True, False):
            other = copy.deepcopy(cs)
            if duplicate:
                other[1]['scope'] = copy.deepcopy(other[0]['scope'])
            else:
                other[1]['scope']['episode'] = 'independent-world'
            loops = [DecisionLoop(c) for c in other]
            saved = [loop.record() for loop in loops]
            callback = Mock(side_effect=AssertionError('invalid group reached completion'))
            with self.subTest(duplicate=duplicate), self.assertRaises(ValueError):
                DecisionLoop.decide_batch([(loop, request(c)) for loop, c in zip(loops, other)],
                                          stochastic=False, planner=planner_with(callback))
            self.assertEqual([loop.record() for loop in loops], saved)
            callback.assert_not_called()

    def test_failed_recovery_proof_keeps_old_blocked_root_but_never_introduces_blocked_alternative(self):
        w, actors, cs, _ = fixture(replace(Battle.start('open', 'either', limit=4), tick=2))
        loops = []
        purposes = []
        for c in cs:
            p = PurposeRequest(0., 0., {a['id']: Activity('wait', 'prepare', patience=1, release='opening')
                if a['id'] == 'guard' else Activity('attempt' if a['id'] == 'move:1:1' else 'uncertain', 'advance')
                for a in c['actions']})
            watch = ProgressWatch(c['scope'], ProgressConfig(repeat_limit=1, proof_margin=.02))
            watch.observe(0, 'guard', p, PurposeFeedback(0., 0.))
            if 'move:1:1' in p.activities:
                watch.observe(1, 'move:1:1', p, PurposeFeedback(0., 0.))
            loops.append(DecisionLoop(c, progress=watch))
            purposes.append(p)
        captured = []
        def complete(contexts, roots, options):
            captured.append((roots, copy.deepcopy(options)))
            return combat_completion(w, actors, contexts, roots, options)
        results = DecisionLoop.decide_batch([(loop, request(c, p)) for loop, c, p in zip(loops, cs, purposes)],
                                           stochastic=False, planner=planner_with(complete))
        self.assertEqual(len(captured), 1)
        roots, options = captured[0]
        for row in (0, 1):
            self.assertEqual(roots[row], 'move:1:1')
            self.assertIn('move:1:1', options[row])
            self.assertNotIn('guard', options[row])
            self.assertFalse(results[row]['progress']['recovery']['approved'])
            self.assertTrue(results[row]['progress']['unresolved'])
        self.assertTrue(results[0]['root_completion']['adopted'])
        for row, result in enumerate(results):
            selected = result['decision']['action_id']
            self.assertTrue(selected == roots[row] or selected not in result['progress']['blocked'])


if __name__ == '__main__':
    unittest.main()
