import copy
import json
import unittest

from reflex.experimental_gain import GainHorizonOwner, RelativeGainHorizonOwner
from reflex.laboratory import PROFILES
from reflex.model_observer import resolve_feedback
from tools.continuing_games import ContinuingGame, GAMES
from tools.run_adverse_recovery import actor_for, decide
from tools.run_owned_value import cumulative_basis


def controller(game, cls=RelativeGainHorizonOwner):
    return cls(actor_for(game, game.context(game.start()), 'transfer_relative_gain'),
        observe=game.context, events=game.events, terminal=game.terminal,
        assess=lambda w: game.goal(w).record(), request_for=game.request,
        feedback=game.feedback, model_id=game.model_id, target=game.target,
        cumulative_preference=cumulative_basis(game, True))


class RelativeIntegrationTests(unittest.TestCase):
    def test_registered_composition_and_controller_match_actual_learning_all_families(self):
        # Registered H32 / three seeds, but short terminal worlds isolate the
        # integration boundary without claiming long-world quality evidence.
        for family in GAMES:
            with self.subTest(family=family):
                game = ContinuingGame(family, PROFILES[2], turns=3)
                world = game.start()
                actual = controller(game)
                reference = copy.deepcopy(actual.actor)
                for tick in range(3):
                    before = actual.actor.record()
                    result = actual.decide(world)
                    expected, captured = decide(reference, game, world, 'transfer_relative_gain')
                    self.assertEqual(result, expected)
                    self.assertEqual(actual.last, captured)
                    self.assertEqual(actual.actor.predictor.record(), before['predictor'])
                    action = result['decision']['action_id']
                    after, flow = game.transition(world, action, str(tick))
                    feedback = game.feedback(world, after, action, result['context'], flow)
                    actual.resolve(result['ticket'], feedback)
                    resolve_feedback(reference, expected['ticket'], feedback)
                    self.assertEqual(actual.actor.record(), reference.record())
                    world = after
                    saved = json.loads(json.dumps(actual.checkpoint(world)))
                    actual = controller(game).restored(world, saved)
                    self.assertIs(type(actual), RelativeGainHorizonOwner)
                    self.assertEqual(actual.actor.record(), reference.record())

    def test_old_gain_history_cannot_be_silently_reinterpreted(self):
        game = ContinuingGame('auction', PROFILES[0], turns=3)
        world = game.start()
        origin = controller(game, GainHorizonOwner)
        relative = controller(game)
        self.assertEqual(origin.actor.record(), relative.actor.record())
        for target, source in ((origin, relative), (relative, origin)):
            before = target.actor.record()
            with self.assertRaises(ValueError):
                target.restored(world, source.checkpoint(world))
            self.assertEqual(target.actor.record(), before)


if __name__ == '__main__':
    unittest.main()
