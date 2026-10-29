import copy
import json
import unittest
from dataclasses import replace

from reflex.core import compile_batch, character_key
from reflex.decision_loop import DecisionLoop
from reflex.model_observer import ModelFeedback, resolve_feedback
from tests.test_event_reader import fixture
from tools.current_model_reader import CurrentModelReader


def current_fixture():
    c,actor,request,extra=fixture()
    actor.predictor=CurrentModelReader.from_reader(actor.predictor,c['scope'])
    return c,actor,request,extra


class SingleForecastChoiceTests(unittest.TestCase):
    def test_small_known_model_gain_reaches_same_persons_choice(self):
        c, actor, request, _ = current_fixture()
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
        c, actor, request, _ = current_fixture()
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

    def test_default_reader_retains_stable_small_gain_behavior(self):
        c,actor,request,_=fixture()
        actor.predictor.predict=lambda context: dict(a=.48,b=.52,c=0.)
        result=actor.decide(request(c),False)
        self.assertEqual(result['decision']['action_id'],'a')
        self.assertEqual(result['reading']['reason'],'no_material_persona_gain')

    def test_restore_rejects_wrong_selection_contract(self):
        c,actor,request,_=current_fixture()
        result=actor.decide(request(c),False)
        resolve_feedback(actor,result['ticket'],ModelFeedback(None,event={'revealed_action':'b'}))
        saved=json.loads(json.dumps(actor.record()))
        _,ordinary,_,_=fixture()
        with self.assertRaises(ValueError):
            DecisionLoop.from_record(c,saved,predictor=ordinary.predictor)
        restored=DecisionLoop.from_record(c,saved,predictor=actor.predictor)
        next_context=copy.deepcopy(c);next_context['tick']=1
        left=actor.decide(request(next_context),False)
        right=restored.decide(request(next_context),False)
        self.assertEqual(left,right)

    def test_invalid_current_model_declarations_do_not_mutate_owner(self):
        for changes in (dict(selection='unknown'),dict(selection='current_model',known_conditionals=False),
                        dict(selection='current_model',response_candidate=dict(a=.4,b=.3,c=.3))):
            c,actor,request,_=fixture();saved=copy.deepcopy(actor.record())
            def reader(context,cap):return replace(actor.predictor(context,cap),**changes)
            with self.assertRaises(ValueError):actor.decide(replace(request(c),reader=reader),False)
            self.assertEqual(saved,actor.record())


if __name__=='__main__': unittest.main()
