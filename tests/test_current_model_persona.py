"""The experimental H32 reader must retain a functional personality tradeoff."""
import json
import unittest

from reflex.core import character_key,digest
from reflex.decision_loop import DecisionLoop
from reflex.laboratory import PROFILES
from reflex.model_observer import resolve_feedback
from tests.test_continuation_persona import IntrinsicTradeoff
from tools.run_adverse_recovery import actor_for,decide


class CurrentModelPersonaTests(unittest.TestCase):
    def test_h32_keeps_values_need_override_and_opaque_action_names(self):
        for profile,mode,names,expected in (
            (PROFILES[0],'principle',('x','y'),'x'),
            (PROFILES[2],'principle',('x','y'),'y'),
            (PROFILES[2],'need',('x','y'),'x'),
            (PROFILES[2],'principle',('zeta','alpha'),'alpha'),
        ):
            with self.subTest(profile=profile['id'],mode=mode,names=names):
                game=IntrinsicTradeoff(profile,names,mode=mode)
                world=game.start();context=game.context(world)
                actor=actor_for(game,context,'consistent')
                before=digest(actor.record())
                result,captured=decide(actor,game,world,'consistent')
                self.assertEqual(result['decision']['action_id'],expected)
                self.assertEqual(captured['horizon_consensus']['source_owner'],before)
                self.assertEqual(character_key(result['context']),character_key(context))
                self.assertEqual(result['reading']['reason'],'single_forecast_persona_choice')
                after,flow=game.transition(world,expected,'0')
                resolve_feedback(actor,result['ticket'],game.feedback(world,after,expected,result['context'],flow))
                saved=json.loads(json.dumps(actor.record()))
                restored=DecisionLoop.from_record(context,saved,predictor=actor.predictor)
                self.assertEqual(saved,restored.record())
                self.assertEqual(actor.predictor.observations,1)


if __name__=='__main__':unittest.main()
