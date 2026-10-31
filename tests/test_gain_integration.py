import copy
import json
import unittest
from unittest.mock import patch

from reflex.core import digest
from reflex.decision_loop import DecisionLoop
from reflex.experimental_gain import GainHorizonOwner,transfer_event_reader
from reflex.fixed_horizon import FixedHorizonOwner
from reflex.laboratory import PROFILES
from reflex.model_observer import resolve_feedback
from reflex._horizon_consensus import consensus_planner
from tools.continuing_games import ContinuingGame,GAMES
from tools.run_adverse_recovery import actor_for
from tools.run_owned_value import cumulative_basis


def actor(game):
    person=game.actor(game.context(game.start()),learning='segmented');reader=person.predictor
    person.predictor=transfer_event_reader(person.scope,reader.learner.names,
        reader.models_for,reader.outcome_for,legal_for=reader.legal_for,
        model_id=game.model_id,known_conditionals=True,factors=reader.learner.factors,
        features_for=reader.features_for,key=reader.key)
    return person


def owner(game,**changes):
    options=dict(observe=game.context,events=game.events,terminal=game.terminal,
        assess=lambda w:game.goal(w).record(),request_for=game.request,feedback=game.feedback,
        model_id=game.model_id,target=game.target,cumulative_preference=cumulative_basis(game,True),
        horizon=3,seeds=(701,),recovery_cache=64)
    return GainHorizonOwner(actor(game),**dict(options,**changes))


class GainIntegrationTests(unittest.TestCase):
    def test_same_complete_judgment_learning_and_restore_in_all_rule_families(self):
        for family in GAMES:
            with self.subTest(family=family):
                game=ContinuingGame(family,PROFILES[2],turns=3);world=game.start()
                actual=owner(game);reference=actor_for(game,game.context(world),'transfer_purpose_gain')
                self.assertEqual(actual.actor.record(),reference.record())
                for tick in range(3):
                    request=game.request(game.context(world,reference.state));cap={}
                    planner=consensus_planner(reference,request,world,captured=cap,
                        observe=game.context,events=game.events,terminal=game.terminal,
                        assess=lambda w:game.goal(w).record(),request_for=game.request,feedback=game.feedback,
                        model_id=game.model_id,target=game.target,cumulative_preference=cumulative_basis(game,True),
                        minimum_horizon=3,horizon=3,seeds=(701,),recovery_cache=64,world_model='posterior',
                        purpose_estimator='conditional',tail_recovery=2,include_retry=True,
                        coherent_cumulative_mean=True,tail_purpose_reflex=True,retry_until_gain=True)
                    expected=DecisionLoop.decide_batch([(reference,request)],False,planner)[0]
                    old_predictor=actual.actor.predictor.record()
                    result=actual.decide(world)
                    self.assertEqual(result,expected);self.assertEqual(actual.last,cap)
                    self.assertEqual(actual.actor.predictor.record(),old_predictor)
                    with self.assertRaises(ValueError):actual.checkpoint(world)
                    with self.assertRaises(ValueError):actual.decide(world)
                    key=result['decision']['action_id'];after,flow=game.transition(world,key,str(tick))
                    feedback=game.feedback(world,after,key,result['context'],flow)
                    actual.resolve(result['ticket'],feedback);resolve_feedback(reference,expected['ticket'],feedback)
                    self.assertEqual(actual.actor.record(),reference.record());world=after
                    saved=json.loads(json.dumps(actual.checkpoint(world)))
                    actual=owner(game).restored(world,saved)
                    self.assertIs(type(actual),GainHorizonOwner)
                    self.assertEqual(actual.actor.record(),reference.record())
                with self.assertRaises(ValueError):actual.decide(world)

    def test_stable_and_other_budgets_cannot_silently_accept_candidate_save(self):
        game=ContinuingGame('auction',PROFILES[0],turns=3);world=game.start()
        experimental=owner(game);saved=experimental.checkpoint(world)
        stable=FixedHorizonOwner(copy.deepcopy(experimental.actor),**{k:experimental.options[k] for k in (
            'observe','events','terminal','assess','request_for','feedback','model_id','target','cumulative_preference','seeds')},horizon=3)
        with self.assertRaises(ValueError):stable.restored(world,saved)
        with self.assertRaises(ValueError):experimental.restored(world,stable.checkpoint(world))
        for change in (dict(horizon=4),dict(seeds=(702,)),dict(recovery_cache=0)):
            with self.assertRaises(ValueError):owner(game,**change).restored(world,saved)
        wrong=copy.deepcopy(saved);wrong['payload']['actor']['seed']+=1;wrong['sha256']=digest(wrong['payload'])
        before=experimental.actor.record()
        with self.assertRaises(ValueError):experimental.restored(world,wrong)
        self.assertEqual(experimental.actor.record(),before)

    def test_no_automatic_conversion_or_incomplete_prediction_commit(self):
        game=ContinuingGame('resources',PROFILES[0],turns=3)
        baseline=game.actor(game.context(game.start()),learning='segmented');before=baseline.record()
        with self.assertRaises(ValueError):GainHorizonOwner(baseline)
        self.assertEqual(baseline.record(),before)
        candidate=owner(game);before=candidate.actor.record()
        with patch('reflex.fixed_horizon.consensus_planner',return_value=lambda cs,ds:None):
            with self.assertRaises(ValueError):candidate.decide(game.start())
        self.assertEqual(candidate.actor.record(),before);self.assertIsNone(candidate.actor.pending)
        self.assertIsNone(candidate.last)

    def test_configuration_bounds_remain_strict(self):
        game=ContinuingGame('combat',PROFILES[0],turns=3)
        for change in (dict(horizon=True),dict(horizon=1),dict(seeds=(True,)),dict(seeds=(701,701)),
                       dict(recovery_cache=-1),dict(cumulative_preference=None)):
            with self.assertRaises(ValueError):owner(game,**change)


if __name__=='__main__':unittest.main()
