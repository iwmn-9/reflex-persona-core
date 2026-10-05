"""Fixed-owner contracts for the opt-in approximate combat self model."""
import copy
from contextlib import ExitStack
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from reflex.combat import Battle, alive, legal, make_context
from reflex.combat_planning import TacticalControl, forecast
from reflex.core import Policy, compile_batch
from reflex.examples import action, effect
from reflex.laboratory import profiles
from reflex.persona_continuation import PersonaContinuation


def fixture(profile=None, world=None, seed=282, route='secure', coordinate=True):
    world = world or Battle.start('open', 'either', limit=5)
    actors = alive(world, 0)
    profile = profile or profiles()[1]
    contexts = [make_context(world, i, profile, seed, route, survival_security=True)
                for i in actors]
    for c in contexts:
        primary = max((k for k, n in c['needs'].items() if n['enabled']),
                      key=lambda k: c['needs'][k]['deficit'])
        c['state'].update(mode='principle', primary_need=primary,
                          mode_urgency=c['needs'][primary]['deficit'])
    decisions = [Policy().choose(c, False) for c in contexts]
    model = PersonaContinuation(actors, contexts, decisions, coordinate)
    states = model.start({i: 'guard' for i in actors})
    return world, actors, contexts, decisions, model, states


class PersonaContinuationTests(unittest.TestCase):
    def test_reconstructed_context_keeps_owner_axes_seed_and_perceived_route(self):
        w, actors, cs, ds, model, states = fixture()
        units = list(w.units)
        units[0] = replace(units[0], hp=3, ammo=0)
        later = replace(w, tick=1, units=tuple(units))
        roles = {i: 'eliminate' for i in actors}
        rebuilt = model.contexts(later, actors, roles, states)
        for source, c, i in zip(cs, rebuilt, actors):
            for key in ('personality', 'values', 'seed', 'scope', 'objective'):
                self.assertEqual(c[key], source[key])
            self.assertEqual(c['facts']['chosen_route'], 'secure')
            self.assertEqual(c['tick'], 1)
            self.assertEqual(c['state'], states[i])
            self.assertEqual({a['id'] for a in c['actions']}, set(legal(later, i)))
        self.assertGreater(rebuilt[0]['needs']['physiology']['deficit'],
                           cs[0]['needs']['physiology']['deficit'])

    def test_counterfactual_personalities_change_choice_on_identical_public_state(self):
        w = Battle.start('open', 'eliminate')
        units = list(w.units)
        units[0] = replace(units[0], x=3, y=2)
        units[1] = replace(units[1], x=3, y=1, hp=3)
        units[3] = replace(units[3], x=5, y=2, hp=3)
        w = replace(w, units=tuple(units))
        choices = {}
        for profile in profiles():
            _, _, _, _, model, states = fixture(profile, w, route='eliminate', coordinate=False)
            choices[profile['id']] = model.choose(replace(w, tick=1), 0, 'eliminate', states)[0][0]
        self.assertEqual(choices['care'], 'heal:1')
        self.assertEqual(choices['ego'], 'shoot:3')
        self.assertNotEqual(choices['care'], choices['ego'])

    def test_every_choice_remains_inside_original_tier_and_near_best_band(self):
        for profile in profiles():
            w, actors, _, _, model, states = fixture(profile)
            later = replace(w, tick=1)
            roles = {i: 'eliminate' if i == 1 else 'secure' for i in actors}
            choices, following = model.choose(later, 0, roles, states)
            b = compile_batch(model.contexts(later, actors, roles, states))
            d = Policy().decide(b, False)
            for row, i in enumerate(actors):
                j = b.ids[row].index(choices[i])
                best = np.max(np.where(d.eligible[row], d.scores[row], -np.inf))
                self.assertTrue(d.eligible[row, j])
                self.assertGreaterEqual(d.scores[row, j], best - .025 - 1e-12)
                self.assertEqual(following[i]['intent_action'], choices[i])
                expected_age = states[i]['age'] + 1 if states[i]['intent_action'] == choices[i] else 0
                self.assertEqual(following[i]['age'], expected_age)

    def test_each_proposed_root_seeds_its_own_actual_intent_and_age(self):
        w, actors, cs, ds, _, _ = fixture()
        cs[0]['state'].update(intent_action='guard', age=999999)
        cs[1]['state'].update(intent_action='guard', age=7)
        ds = [Policy().choose(c, False) for c in cs]
        model = PersonaContinuation(actors, cs, ds)
        first = {0: 'guard', 1: 'move:2:2', 2: 'guard'}
        states = model.start(first)
        self.assertEqual(states[0]['age'], 1000000)
        self.assertEqual(states[1]['age'], 0)
        for i in actors:
            self.assertEqual(states[i]['intent_action'], first[i])
        states[0]['age'] = 5
        self.assertEqual(model.start(first)[0]['age'], 1000000)
        self.assertEqual(model.start({i: 'guard' for i in actors})[1]['age'], 8)

    def test_tactical_override_carries_chosen_action_instead_of_policy_winner(self):
        w, actors, _, _, model, states = fixture()
        later = replace(w, tick=1)

        def near_ties(*args, **kwargs):
            c = make_context(*args, **kwargs)
            for a in c['actions']:
                # Guard is the strict Policy maximum, while every move is
                # still inside its unchanged near-best band.
                a['outcomes'] = [effect(.01 if a['id'] == 'guard' else 0.)]
            return c

        def prefer_move(world, actor, route):
            return {k: (1. if k.startswith('move:') else 0.) for k in legal(world, actor)}

        with patch('reflex.persona_continuation.make_context', side_effect=near_ties), \
                patch('reflex.combat_planning.tactical_scores', side_effect=prefer_move):
            b = compile_batch(model.contexts(later, actors, 'secure', states))
            numeric = Policy().decide(b, False)
            self.assertTrue(all(b.ids[i][j] == 'guard' for i, j in enumerate(numeric.action)))
            chosen, following = model.choose(later, 0, 'secure', states)
        for i, key in chosen.items():
            self.assertTrue(key.startswith('move:'))
            self.assertEqual(following[i]['intent_action'], key)
            self.assertEqual(following[i]['age'], 0)

    def test_coordination_cannot_lower_original_value_tier(self):
        w, actors, _, _, model, states = fixture()
        later = replace(w, tick=1)

        def forced_tier(world, actor, *args, **kwargs):
            c = make_context(world, actor, *args, **kwargs)
            preferred = 'move:1:1' if actor in (0, 1) else 'guard'
            self.assertIn(preferred, legal(world, actor))
            c['actions'] = [action(a['id'], effect(0., values={'security': 1. if a['id'] == preferred else -1.}))
                            for a in c['actions']]
            return c

        with patch('reflex.persona_continuation.make_context', side_effect=forced_tier):
            b = compile_batch(model.contexts(later, actors, 'secure', states))
            d = Policy().decide(b, False)
            self.assertTrue(all(d.mode == 1))
            for row in (0, 1):
                self.assertFalse(d.eligible[row, b.ids[row].index('guard')])
            choices, following = model.choose(later, 0, 'secure', states)
        self.assertEqual(choices[0], 'move:1:1')
        self.assertEqual(choices[1], 'move:1:1')
        self.assertEqual(model.conflicts, 1)
        self.assertEqual(following[1]['intent_action'], 'move:1:1')

    def test_cache_reuses_identical_inputs_but_not_new_world_intent_or_role(self):
        w, actors, cs, ds, model, states = fixture()
        later = replace(w, tick=1)
        before = copy.deepcopy((cs, ds, states))
        original = model.choose(later, 0, 'secure', states)
        scorings = model.scorings
        self.assertEqual(model.choose(later, 0, 'secure', states), original)
        self.assertEqual(model.scorings, scorings)
        changed = copy.deepcopy(states)
        changed[0]['age'] += 1
        model.choose(later, 0, 'secure', changed)
        self.assertEqual(model.scorings, scorings + len(actors))
        model.choose(later, 0, 'eliminate', states)
        self.assertEqual(model.scorings, scorings + 2 * len(actors))
        model.choose(replace(later, tick=2), 0, 'secure', states)
        self.assertEqual(model.scorings, scorings + 3 * len(actors))
        self.assertEqual((cs, ds, states), before)

    def test_cache_results_are_isolated_from_caller_mutation(self):
        w, _, _, _, model, states = fixture()
        later = replace(w, tick=1)
        choices, following = model.choose(later, 0, 'secure', states)
        expected = copy.deepcopy((choices, following))
        choices[0] = 'caller-mutated'
        following[0]['intent_action'] = 'caller-mutated'
        self.assertEqual(model.choose(later, 0, 'secure', states), expected)

    def test_dead_actors_are_omitted_and_empty_own_team_is_safe(self):
        w, actors, _, _, model, states = fixture()
        units = list(w.units)
        units[0] = replace(units[0], hp=0)
        later = replace(w, tick=1, units=tuple(units))
        choices, following = model.choose(later, 0, 'secure', states)
        self.assertEqual(set(choices), {1, 2})
        self.assertEqual(set(following), {1, 2})
        for i in actors:
            units[i] = replace(units[i], hp=0)
        self.assertEqual(model.choose(replace(later, units=tuple(units)), 0, 'secure', states), ({}, {}))

    def test_unsupported_context_contracts_fail_closed(self):
        w, actors, cs, ds, _, states = fixture()
        malformed = copy.deepcopy(cs)
        malformed[0]['facts'].pop('security_model')
        with self.assertRaisesRegex(ValueError, 'survival-security'):
            PersonaContinuation(actors, malformed, ds)
        with self.assertRaises(ValueError):
            PersonaContinuation(actors, cs, ds[:-1])
        malformed = copy.deepcopy(cs)
        malformed[0]['scope']['npc'] = 'different-owner'
        model = PersonaContinuation(actors, malformed, ds)
        with self.assertRaisesRegex(ValueError, 'actor/episode'):
            model.choose(replace(w, tick=1), 0, 'secure', states)

    def test_default_is_objective_and_does_not_construct_persona_model(self):
        w, actors, cs, ds, _, _ = fixture()
        control = TacticalControl(horizon=2, samples=1, max_plans=2)
        self.assertEqual(control.self_continuation, 'objective')
        with patch('reflex.persona_continuation.PersonaContinuation', side_effect=AssertionError('default changed')):
            implicit = forecast(w, actors, cs, ds, control)
            explicit = forecast(w, actors, cs, ds, replace(control, self_continuation='objective'))
        self.assertEqual(implicit, explicit)
        self.assertNotIn('self_model', implicit.audit)
        for value in (None, True, '', 'nested'):
            with self.assertRaises(ValueError):
                TacticalControl(self_continuation=value)

    def test_tracing_is_passive_including_persona_state_and_modeled_terminal(self):
        w, actors, cs, ds, _, _ = fixture(world=Battle.start('choke', 'eliminate', limit=2), route='eliminate')
        saved = copy.deepcopy((w, cs, ds))
        control = TacticalControl(horizon=4, samples=2, max_plans=2, self_continuation='persona-band')
        plain = forecast(w, actors, cs, ds, control)
        traced = forecast(w, actors, cs, ds, replace(control, trace_execution=True))
        self.assertEqual((plain.contexts, plain.roots, plain.purpose),
                         (traced.contexts, traced.roots, traced.purpose))
        self.assertEqual(plain.audit['nodes'], traced.audit['nodes'])
        self.assertEqual(plain.audit['self_model'], traced.audit['self_model'])
        self.assertEqual((w, cs, ds), saved)
        for label, roots in traced.roots.items():
            for sample in traced.audit['plans'][label]['execution_samples']:
                self.assertEqual(len(sample), 2)
                self.assertTrue(sample[-1]['terminal'])
                self.assertEqual(sample[0]['self'], {str(i): key for i, key in zip(actors, roots)})

    def test_forecast_never_calls_actual_rival_nested_planner_or_empirical_memory(self):
        w, actors, cs, ds, _, _ = fixture()
        control = TacticalControl(horizon=3, samples=2, max_plans=2, self_continuation='persona-band')
        forbidden = ('reflex.combat.opponent', 'reflex.combat.reference',
                     'reflex.combat_planning.forecast',
                     'reflex.decision_loop.DecisionLoop.decide_batch',
                     'reflex.judgment.OutcomeMemory.prepare',
                     'reflex.judgment.OutcomeMemory.commit',
                     'reflex.judgment.OutcomeMemory.observe')
        with ExitStack() as stack:
            for target in forbidden:
                stack.enter_context(patch(target, side_effect=AssertionError('forbidden access: ' + target)))
            # This imported entry point is the one authorized outer forecast;
            # patching its module name catches any nested call from inside it.
            result = forecast(w, actors, cs, ds, control)
        self.assertGreater(result.audit['self_model']['scorings'], 0)
        self.assertLessEqual(result.audit['nodes'], len(result.roots) * control.samples * control.horizon)


if __name__ == '__main__':
    unittest.main()
