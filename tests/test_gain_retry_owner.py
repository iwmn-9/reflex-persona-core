"""Actual existing H32 connection, with learning and every persona boundary."""
import copy
import json
import unittest

from reflex.core import character_key,digest
from reflex.decision_loop import DecisionLoop
from reflex.laboratory import PROFILES
from reflex.model_observer import resolve_feedback
from tests.test_continuation_persona import IntrinsicTradeoff
from tools.run_adverse_recovery import actor_for,decide
from tools.run_continuation_portfolio import game_for


class GainRetryOwnerTests(unittest.TestCase):
    def test_new_arm_has_identical_factual_learner_in_every_world_family(self):
        for family in ('auction','combat','resources','gambling'):
            with self.subTest(family=family):
                game=game_for(dict(mechanism='world',game=family,profile=2,turns=128,seed=10061202))
                world=game.start();context=game.context(world)
                control=actor_for(game,context,'transfer_purpose')
                candidate=actor_for(game,context,'transfer_purpose_gain')
                self.assertEqual(candidate.record(),control.record())
                self.assertIs(type(candidate.predictor.learner),type(control.predictor.learner))

    def test_h32_owned_integration_retains_intrinsic_values_and_actual_learning(self):
        for profile,mode,names,expected in [(0,'principle',('x','y'),'x'),(2,'principle',('x','y'),'y'),
                (2,'need',('x','y'),'x'),(2,'principle',('zeta','alpha'),'alpha')]:
            with self.subTest(profile=profile,mode=mode,names=names):
                game=IntrinsicTradeoff(PROFILES[profile],names,mode=mode);world=game.start()
                context=game.context(world);actor=actor_for(game,context,'transfer_purpose_gain')
                control=copy.deepcopy(actor);before=digest(actor.record())
                result,captured=decide(actor,game,world,'transfer_purpose_gain')
                self.assertEqual(result['decision']['action_id'],expected)
                self.assertEqual(captured['horizon_consensus']['source_owner'],before)
                self.assertTrue(any(n.endswith('-until-gain') for n in captured['modeled']))
                self.assertEqual(character_key(result['context']),character_key(context))
                after,flow=game.transition(world,expected,'0')
                resolve_feedback(actor,result['ticket'],game.feedback(world,after,expected,result['context'],flow))
                replay=DecisionLoop.decide_batch([(control,game.request(context))],False,
                    _counterfactual_roots=(expected,))[0]
                resolve_feedback(control,replay['ticket'],game.feedback(world,after,expected,replay['context'],flow))
                self.assertEqual(actor.record(),control.record())
                saved=json.loads(json.dumps(actor.record()))
                restored=DecisionLoop.from_record(context,saved,predictor=actor.predictor)
                self.assertEqual(restored.record(),saved)


if __name__=='__main__':unittest.main()
