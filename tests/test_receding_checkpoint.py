import copy
import json
import unittest
from dataclasses import replace
from reflex.core import digest, compile_batch
from reflex.receding_owner import RecedingOwner
from reflex.cumulative_preference import CumulativePreference
from reflex.sampling_check import DecisionSampling
from reflex.laboratory import PROFILES
from tools.continuing_games import ContinuingGame, GAMES


def controller(game, world, **overrides):
    actor = game.actor(game.context(world), learning='segmented')
    options = dict(observe=game.context, events=game.events, terminal=game.terminal,
        assess=lambda w: game.goal(w).record(), record=lambda w: w,
        request_for=game.request, feedback=game.feedback, model_id=game.model_id,
        target=game.target, tail_steps=1, seeds=(701,), world_model='posterior',
        purpose_estimator='conditional', tail_recovery=2)
    return RecedingOwner(actor, **dict(options, **overrides))


def resign(saved):
    saved['sha256'] = digest(saved['payload'])
    return saved


class RecedingCheckpointTests(unittest.TestCase):
    def test_four_worlds_keep_actual_learning_and_decisions_across_restarts(self):
        for family in GAMES:
            with self.subTest(family=family):
                game = ContinuingGame(family, PROFILES[1], turns=16)
                world = game.start()
                uninterrupted = controller(game, world)
                restarted = controller(game, world)
                initial = copy.deepcopy(uninterrupted.actor.personality)
                for tick in range(16):
                    if tick in (5, 12):
                        saved = json.loads(json.dumps(restarted.checkpoint(world)))
                        template = controller(game, world)
                        before = template.actor.record()
                        restarted = template.restored(world, saved)
                        self.assertEqual(template.actor.record(), before)
                        self.assertEqual(restarted.checkpoint(world), saved)
                    context = game.context(world, uninterrupted.actor.state)
                    prediction_before = restarted.actor.predictor.record()
                    a, b = uninterrupted.decide(world), restarted.decide(world)
                    self.assertEqual(a['decision'], b['decision'])
                    self.assertEqual(restarted.actor.predictor.record(), prediction_before)
                    key = a['decision']['action_id']
                    after, flow = game.transition(world, key, '0' if tick < 7 else '2')
                    feedback = game.feedback(world, after, key, context, flow)
                    uninterrupted.resolve(a['ticket'], feedback)
                    restarted.resolve(b['ticket'], feedback)
                    self.assertEqual(uninterrupted.actor.record(), restarted.actor.record())
                    self.assertEqual(restarted.actor.personality, initial)
                    self.assertEqual(restarted.actor.predictor.memory.observations, tick + 1)
                    world = after
                # Terminal state is a valid resolved save boundary too.
                saved = restarted.checkpoint(world)
                self.assertEqual(controller(game, game.start()).restored(world, saved).actor.record(),
                                 restarted.actor.record())
                with self.assertRaises(ValueError): restarted.decide(world)
                with self.assertRaises(ValueError): compile_batch([game.context(world)])

    def test_planning_settings_cannot_silently_change(self):
        game = ContinuingGame('auction', PROFILES[0], turns=5)
        world = game.start()
        saved = controller(game, world).checkpoint(world)
        changes = [dict(tail_steps=2), dict(seeds=(702,)), dict(world_model='frozen'),
            dict(max_regret=.2), dict(purpose_estimator='sampled'), dict(tail_recovery=0),
            dict(tail_recovery_limit=17), dict(tail_recovery_mode='always'),
            dict(model_id=game.model_id + '-changed'), dict(target=game.target + '-changed')]
        for change in changes:
            with self.subTest(change=change):
                template = controller(game, world, **change)
                before = template.actor.record()
                with self.assertRaises(ValueError): template.restored(world, saved)
                self.assertEqual(template.actor.record(), before)

    def test_explicit_optional_settings_survive_json_without_enabling_them_by_default(self):
        game = ContinuingGame('auction', PROFILES[0], turns=5)
        world = game.start()
        loop = controller(game, world, sampling_check=DecisionSampling(banks=((701,), (711,), (721,))),
            cumulative_preference=CumulativePreference(game.target, 'score', (('values.achievement', 1.),)))
        saved = json.loads(json.dumps(loop.checkpoint(world)))
        self.assertEqual(loop.restored(world, saved).checkpoint(world), saved)
        with self.assertRaises(ValueError): controller(game, world).restored(world, saved)

    def test_actor_configuration_and_public_boundary_must_match(self):
        game = ContinuingGame('auction', PROFILES[0], turns=5)
        world = game.start()
        original = controller(game, world)
        saved = original.checkpoint(world)
        mutations = [lambda x: setattr(x.actor.memory, 'capacity', 65),
            lambda x: setattr(x.actor.evidence, 'margin', .03),
            lambda x: setattr(x.actor.pressure, 'config', replace(x.actor.pressure.config, grace=4)),
            lambda x: setattr(x.actor.progress, 'config', replace(x.actor.progress.config, grace=4)),
            lambda x: setattr(x.actor, 'progress', None),
            lambda x: setattr(x.actor, 'predictor', None)]
        for mutate in mutations:
            template = controller(game, world)
            mutate(template)
            with self.assertRaises(ValueError): template.restored(world, saved)
        advanced = copy.deepcopy(world); advanced['tick'] += 1
        with self.assertRaises(ValueError): original.restored(advanced, saved)
        with self.assertRaises(ValueError):
            controller(ContinuingGame('auction', PROFILES[1], turns=5), world).restored(world, saved)
        template = controller(game, world)
        observe = template.observe
        template.observe = lambda w, state: dict(observe(w, state), objective='Another objective')
        with self.assertRaises(ValueError): template.restored(world, saved)

    def test_corruption_unknown_fields_and_inconsistent_owner_are_rejected_atomically(self):
        game = ContinuingGame('auction', PROFILES[0], turns=5)
        world = game.start(); template = controller(game, world)
        saved = template.checkpoint(world); before = template.actor.record()
        bad = []
        x = copy.deepcopy(saved); x['sha256'] = '0' * 64; bad.append(x)
        x = copy.deepcopy(saved); x['payload']['version'] = 'future-v9'; bad.append(resign(x))
        x = copy.deepcopy(saved); x['payload']['unknown'] = 1; bad.append(resign(x))
        x = copy.deepcopy(saved); x['payload']['actor']['unknown'] = 1; bad.append(resign(x))
        x = copy.deepcopy(saved); x['payload']['actor']['memory']['capacity'] = 65; bad.append(resign(x))
        x = copy.deepcopy(saved); x['payload']['actor']['predictor']['memory']['identity'] = 'wrong'; bad.append(resign(x))
        x = copy.deepcopy(saved); x['payload']['actor']['last_tick'] = 2; bad.append(resign(x))
        bad.extend([{}, None, dict(payload=[], sha256=digest([]))])
        for x in bad:
            with self.assertRaises(ValueError): template.restored(world, x)
            self.assertEqual(template.actor.record(), before)

    def test_pending_failure_retry_and_missing_effect_do_not_invent_learning(self):
        game = ContinuingGame('auction', PROFILES[0], turns=5)
        world = game.start(); loop = controller(game, world)
        saved = loop.checkpoint(world); context = game.context(world)
        result = loop.decide(world); key = result['decision']['action_id']
        after, flow = game.transition(world, key, '2')
        feedback = game.feedback(world, after, key, context, flow)
        memory, predictor = loop.actor.memory, loop.actor.predictor.record()
        with self.assertRaises(ValueError): loop.checkpoint(world)
        with self.assertRaises(ValueError): loop.restored(world, saved)
        with self.assertRaises(ValueError):
            loop.resolve(result['ticket'], replace(feedback, observed=[float('nan')] * len(feedback.observed)))
        self.assertIs(loop.actor.memory, memory)
        self.assertEqual(loop.actor.predictor.record(), predictor)
        self.assertIsNotNone(loop.actor.pending)
        # Revealed response is real, effects are missing: update only observed channels.
        loop.resolve(result['ticket'], replace(feedback, observed=None))
        self.assertEqual(loop.actor.predictor.memory.observations, 1)
        self.assertEqual(loop.actor.memory.record()['entries'], saved['payload']['actor']['memory']['entries'])
        checkpoint = loop.checkpoint(after)
        self.assertEqual(controller(game, after).restored(after, checkpoint).actor.record(), loop.actor.record())


if __name__ == '__main__': unittest.main()
