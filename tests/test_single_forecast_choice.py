import copy
import json
import unittest
from dataclasses import replace

from reflex.core import compile_batch, character_key
from reflex.decision_loop import DecisionLoop
from reflex.model_observer import ModelFeedback, resolve_feedback
from tests.test_event_reader import fixture


class SingleForecastChoiceTests(unittest.TestCase):
    def test_small_known_model_gain_reaches_same_persons_choice(self):
        c, actor, request, _ = fixture()
        actor.predictor.predict = lambda context: dict(a=.48, b=.52, c=0.)
        reading = actor.predictor(c, 64)
        direct = actor.policy.decide(compile_batch([reading.context]), False).records(compile_batch([reading.context]))[0]
        result = actor.decide(request(c), False)
        self.assertEqual(direct['action_id'], 'b')
        self.assertEqual(result['decision']['action_id'], direct['action_id'])
        self.assertGreater(result['reading']['gain'], 0.)
        self.assertLess(result['reading']['gain'], .05)
        self.assertEqual(character_key(result['context']), character_key(c))

    def test_current_forecast_context_is_used_even_when_choice_unchanged(self):
        c, actor, request, _ = fixture()
        actor.predictor.predict = lambda context: dict(a=.52, b=.48, c=0.)
        result = actor.decide(request(c), False)
        self.assertEqual(result['decision']['action_id'], 'a')
        self.assertEqual([v['p'] for v in result['context']['actions'][0]['outcomes']], [.52,.48,0.])
        resolve_feedback(actor, result['ticket'], ModelFeedback(None,event={'revealed_action':'a'}))
        saved=json.loads(json.dumps(actor.record()))
        restored=DecisionLoop.from_record(c,saved,predictor=actor.predictor)
        self.assertEqual(saved,restored.record())

    def test_small_competing_advice_gain_keeps_existing_adoption_boundary(self):
        c, actor, request, _ = fixture()
        actor.predictor.predict = lambda context: dict(a=.48, b=.52, c=0.)
        def competing(context, cap):
            reading=actor.predictor(context,cap)
            return replace(reading,response_candidate=dict(a=.47,b=.53,c=0.))
        result=actor.decide(replace(request(c),reader=competing),False)
        self.assertEqual(result['decision']['action_id'],'a')
        self.assertEqual(result['reading']['reason'],'no_material_persona_gain')


if __name__=='__main__': unittest.main()
