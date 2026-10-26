import copy
import unittest
from dataclasses import replace
from reflex.core import digest
from reflex.native_owned import NativeOwnedBoundary, exported_state
from reflex.model_observer import resolve_feedback
from tests.test_fixed_horizon import owner
from tools.continuing_games import ContinuingGame
from reflex.laboratory import PROFILES


class NativeBoundaryTests(unittest.TestCase):
    def test_native_state_round_trip_and_simulated_learning_never_update_template(self):
        game=ContinuingGame('auction',PROFILES[1],turns=3)
        template=owner(game); boundary=NativeOwnedBoundary(template,game.start())
        initial=digest(template.actor.record()); actor=copy.deepcopy(template.actor); world=game.start()
        for tick in range(2):
            state=exported_state(actor)
            restored=boundary.actor(world,state)
            self.assertEqual(restored.record(),actor.record())
            native=boundary.input(world,state)
            self.assertEqual(native['observation']['tick'],tick)
            saved=copy.deepcopy(state); forecast=boundary.forecast(world,state)
            self.assertEqual(state,saved);self.assertEqual(digest(template.actor.record()),initial)
            self.assertEqual(forecast['owner'],boundary.configuration['actor'])
            self.assertLessEqual(forecast['horizon'],32)
            request=game.request(game.context(world,actor.state)); result=actor.decide(request,False)
            after,flow=game.transition(world,result['decision']['action_id'],'2')
            resolve_feedback(actor,result['ticket'],game.feedback(world,after,result['decision']['action_id'],request.context,flow))
            world=after

    def test_unimplemented_paths_are_explicit_errors(self):
        game=ContinuingGame('gambling',PROFILES[0],turns=2)
        with self.assertRaises(ValueError):NativeOwnedBoundary(owner(game,horizon=16),game.start())
        template=owner(game);original=template.request_for
        template.request_for=lambda c:replace(original(c),invalidate=(('x','y'),))
        boundary=NativeOwnedBoundary(template,game.start())
        with self.assertRaises(ValueError):boundary.input(game.start(),exported_state(template.actor))
        with self.assertRaises(ValueError):boundary.actor(game.start(),{})
        state=exported_state(template.actor);state['prediction']['observations']=100
        with self.assertRaises(ValueError):boundary.actor(game.start(),state)


if __name__=='__main__':unittest.main()
