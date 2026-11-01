import copy
import unittest
from reflex.decision_loop import DecisionLoop
from reflex.laboratory import PROFILES
from reflex.model_observer import resolve_feedback
from tools.continuing_games import ContinuingGame
from tools.frozen_experience import FrozenExperience


class FrozenExperienceTests(unittest.TestCase):
    def test_experience_stops_but_needs_progress_and_clocks_continue(self):
        for family in ('auction', 'resources', 'combat', 'gambling'):
            game=ContinuingGame(family,PROFILES[1],turns=4)
            w=game.start();actor=game.actor(game.context(w),learning='segmented')
            frozen=FrozenExperience(actor);baseline=actor.record()
            for t in range(3):
                context=game.context(w,actor.state)
                result=DecisionLoop.decide_batch([(actor,game.request(context))],False)[0]
                key=result['decision']['action_id'];after,flow=game.transition(w,key,'2')
                with self.assertRaises(ValueError):frozen.after_feedback(actor,context)
                resolve_feedback(actor,result['ticket'],game.feedback(w,after,key,result['context'],flow))
                current=actor.record()
                actor=frozen.after_feedback(actor,context);saved=actor.record()
                self.assertEqual(saved['predictor'],baseline['predictor'])
                self.assertEqual(saved['memory']['entries'],baseline['memory']['entries'])
                self.assertEqual(saved['evidence'],baseline['evidence'])
                for key in ('state','strategy','pressure','progress','last_tick'):
                    self.assertEqual(saved[key],current[key])
                self.assertEqual(saved['memory']['last_tick'],t)
                self.assertNotEqual(current['predictor'],baseline['predictor'])
                self.assertEqual(DecisionLoop.from_record(context,copy.deepcopy(saved),predictor=actor.predictor).record(),saved)
                w=after

    def test_wrong_owner_and_unadvanced_snapshot_are_rejected(self):
        game=ContinuingGame('resources',PROFILES[0],turns=4)
        actor=game.actor(game.context(game.start()),learning='segmented')
        frozen=FrozenExperience(actor)
        with self.assertRaises(ValueError):frozen.after_feedback(actor,game.context(game.start()))
        alien=ContinuingGame('resources',PROFILES[1],turns=4)
        other=alien.actor(alien.context(alien.start()),learning='segmented')
        with self.assertRaises(ValueError):frozen.after_feedback(other,alien.context(alien.start()))


if __name__=='__main__':unittest.main()
