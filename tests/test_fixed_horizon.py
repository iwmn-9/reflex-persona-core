import copy
import json
import unittest
from unittest.mock import patch
from reflex.core import digest
from reflex.decision_loop import DecisionLoop
from reflex.fixed_horizon import FixedHorizonOwner
from reflex.model_observer import resolve_feedback
from reflex.receding_owner import RecedingOwner
from reflex.laboratory import PROFILES
from tools.continuing_games import ContinuingGame, GAMES
from tools.horizon_consensus import consensus_planner
from tools.run_owned_value import cumulative_basis
from tests.test_continuation_persona import IntrinsicTradeoff


def owner(game, **options):
    actor = game.actor(game.context(game.start()), learning='segmented')
    kwargs = dict(observe=game.context, events=game.events, terminal=game.terminal,
        assess=lambda w: game.goal(w).record(), request_for=game.request, feedback=game.feedback,
        model_id=game.model_id, target=game.target, cumulative_preference=cumulative_basis(game, True))
    return FixedHorizonOwner(actor, **dict(kwargs, **options))


class FixedHorizonTests(unittest.TestCase):
    def test_wrapper_matches_manual_composition_through_feedback_and_restoration(self):
        for family in GAMES:
            with self.subTest(family=family):
                game = ContinuingGame(family, PROFILES[2], turns=4)
                actual = owner(game, horizon=4, seeds=(701,))
                reference = copy.deepcopy(actual.actor); world = game.start()
                for tick in range(4):
                    context = game.context(world, reference.state); request = game.request(context); cap = {}
                    planner = consensus_planner(reference, request, world, observe=game.context,
                        events=game.events, terminal=game.terminal, assess=lambda w: game.goal(w).record(),
                        request_for=game.request, feedback=game.feedback, model_id=game.model_id,
                        target=game.target, horizon=4, minimum_horizon=4, seeds=(701,),
                        cumulative_preference=cumulative_basis(game, True), include_retry=True,
                        purpose_estimator='conditional', tail_recovery=2, recovery_cache=64,
                        coherent_cumulative_mean=True, captured=cap)
                    expected = DecisionLoop.decide_batch([(reference, request)], False, planner)[0]
                    factual_predictor = actual.actor.predictor.record()
                    result = actual.decide(world)
                    self.assertEqual(result, expected)
                    self.assertEqual(actual.last, cap)
                    self.assertEqual(actual.actor.predictor.record(), factual_predictor)
                    self.assertEqual(len(actual.last['horizon_consensus']['ballots']), 1)
                    with self.assertRaises(ValueError): actual.checkpoint(world)
                    key = result['decision']['action_id']; after, flow = game.transition(world, key, '0' if tick < 2 else '2')
                    feedback = game.feedback(world, after, key, context, flow)
                    actual.resolve(result['ticket'], feedback); resolve_feedback(reference, expected['ticket'], feedback)
                    self.assertEqual(actual.actor.record(), reference.record())
                    world = after
                    saved = json.loads(json.dumps(actual.checkpoint(world)))
                    actual = owner(game, horizon=4, seeds=(701,)).restored(world, saved)
                    self.assertIs(type(actual), FixedHorizonOwner)
                    self.assertEqual(actual.actor.record(), reference.record())
                with self.assertRaises(ValueError): actual.decide(world)

    def test_different_controllers_and_budgets_cannot_share_a_checkpoint(self):
        game = ContinuingGame('auction', PROFILES[0], turns=4); world = game.start()
        fixed = owner(game, horizon=4, seeds=(701,)); saved = fixed.checkpoint(world)
        ordinary = RecedingOwner(copy.deepcopy(fixed.actor), **fixed.options)
        with self.assertRaises(ValueError): ordinary.restored(world, saved)
        with self.assertRaises(ValueError): fixed.restored(world, ordinary.checkpoint(world))
        for change in (dict(horizon=5), dict(seeds=(702,)), dict(recovery_cache=0)):
            altered = owner(game, **dict(horizon=4, seeds=(701,)) | change)
            with self.assertRaises(ValueError): altered.restored(world, saved)

    def test_invalid_configuration_and_incomplete_forecast_preserve_owner(self):
        game = ContinuingGame('resources', PROFILES[0], turns=4)
        for options in (dict(horizon=True), dict(horizon=1), dict(horizon=129),
                        dict(seeds=(701, 701)), dict(seeds=(True,)), dict(recovery_cache=True),
                        dict(cumulative_preference=None)):
            with self.assertRaises(ValueError): owner(game, **options)
        fixed = owner(game, horizon=4, seeds=(701,)); before = fixed.actor.record()
        with patch('reflex.fixed_horizon.consensus_planner', return_value=lambda cs, ds: None):
            with self.assertRaises(ValueError): fixed.decide(game.start())
        self.assertEqual(fixed.actor.record(), before)
        self.assertIsNone(fixed.actor.pending)
        self.assertIsNone(fixed.last)

    def test_registered_h32_keeps_intrinsic_persona_tradeoff(self):
        decisions = []
        for profile in (PROFILES[0], PROFILES[2]):
            game = IntrinsicTradeoff(profile)
            fixed = owner(game, seeds=(701,)); before = digest(fixed.actor.record())
            result = fixed.decide(game.start())
            self.assertEqual(fixed.last['horizon_consensus']['source_owner'], before)
            self.assertEqual(len(fixed.last['horizon_consensus']['ballots']), 1)
            decisions.append(result['decision']['action_id'])
        self.assertEqual(decisions, ['x', 'y'])


if __name__ == '__main__': unittest.main()
