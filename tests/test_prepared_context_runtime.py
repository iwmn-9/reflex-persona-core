"""Reuse only an exact, already resolved deterministic single-owner preview."""
import copy
from dataclasses import asdict,replace
import unittest
from unittest.mock import patch

from reflex.closed_loop import owned_rollout
from reflex.core import digest
from reflex.decision_loop import DecisionLoop
from reflex.laboratory import PROFILES
from reflex.model_observer import resolve_feedback
from tools.continuing_games import ContinuingGame
from tools.run_adverse_recovery import actor_for
from tools.run_owned_value import cumulative_basis


def model_for(game,owner,request,world):
    model=owned_rollout(owner,observe=game.context,events=game.events,terminal=game.terminal,
        assess=lambda w:game.goal(w).record(),record=lambda w:w,request_for=game.request,
        feedback=game.feedback,model_id=game.model_id,target=game.target,tail_steps=0,
        world_model='posterior',cumulative_preference=cumulative_basis(game,True))
    return model.planner(owner,request,world,_forecast_only=True)


def fixture():
    game=ContinuingGame('combat',PROFILES[0],16);world=game.start()
    owner=actor_for(game,game.context(world),'transfer_purpose')
    return game,world,owner,game.request(game.context(world))


class PreparedContextRuntimeTests(unittest.TestCase):
    def test_experienced_forecasts_choices_learning_and_restore_are_exact(self):
        for family in ('auction','combat','gambling','resources'):
            for profile in PROFILES:
                with self.subTest(family=family,profile=profile['id']):
                    game=ContinuingGame(family,profile,16);world=game.start()
                    owner=actor_for(game,game.context(world),'transfer_purpose')
                    for _ in range(4):
                        request=game.request(game.context(world,owner.state))
                        chosen=owner.decide(request,False);key=chosen['decision']['action_id']
                        after,flow=game.transition(world,key,game.events(world,key)[0].symbol)
                        resolve_feedback(owner,chosen['ticket'],game.feedback(world,after,key,chosen['context'],flow))
                        world=after
                    request=game.request(game.context(world,owner.state));before=owner.record();outputs=[]
                    original=DecisionLoop.preview
                    for reuse in (False,True):
                        actor=copy.deepcopy(owner);bound=copy.deepcopy(owner);forecasts=[]
                        generate=model_for(game,bound,request,world)
                        def planner(cs,ds):
                            f=generate(cs,ds);forecasts.append(asdict(f));return f
                        def prepared(cs,ds,source,resolved,effective):
                            f=generate._from_prepared(cs,ds,source,resolved,effective)
                            forecasts.append(asdict(f));return f
                        if reuse:planner._from_prepared=prepared
                        with patch.object(DecisionLoop,'preview',autospec=True,side_effect=original) as preview:
                            result=DecisionLoop.decide_batch([(actor,request)],False,planner)[0]
                            self.assertEqual(preview.call_count,0 if reuse else 1)
                        key=result['decision']['action_id']
                        after,flow=game.transition(world,key,game.events(world,key)[-1].symbol)
                        resolve_feedback(actor,result['ticket'],game.feedback(world,after,key,result['context'],flow))
                        saved=actor.record()
                        self.assertEqual(DecisionLoop.from_record(game.context(after),saved,predictor=actor.predictor).record(),saved)
                        self.assertEqual(bound.record(),before)
                        outputs.append((forecasts,result,saved,after))
                    self.assertEqual(outputs[0],outputs[1]);self.assertEqual(owner.record(),before)

    def test_foreign_owner_or_request_falls_back_and_ignores_prepared_data(self):
        game,world,owner,request=fixture();generate=model_for(game,owner,request,world)
        effective,decision=owner.preview(request);expected=generate([effective],[decision])
        foreign=copy.deepcopy(owner);foreign.last_tick=123
        changed=copy.deepcopy(request.context);changed['tick']+=1
        for source,resolved in ((foreign,request),(owner,replace(request,context=changed))):
            with patch.object(DecisionLoop,'preview',autospec=True,side_effect=DecisionLoop.preview) as preview:
                actual=generate._from_prepared([effective],[decision],source,resolved,{'poison':True})
                self.assertEqual(preview.call_count,1)
            self.assertEqual(asdict(actual),asdict(expected))

    def test_mutating_planner_input_still_fails_without_committing_owner(self):
        game,world,owner,request=fixture();before=owner.record()
        generate=model_for(game,copy.deepcopy(owner),request,world)
        def planner(cs,ds):raise AssertionError('prepared path expected')
        def prepared(cs,ds,source,resolved,effective):
            cs[0]['seed']+=1
            return generate._from_prepared(cs,ds,source,resolved,effective)
        planner._from_prepared=prepared
        with self.assertRaisesRegex(ValueError,'different effective owner'):
            DecisionLoop.decide_batch([(owner,request)],False,planner)
        self.assertEqual(owner.record(),before)
        # The failed call retained no prepared state and did not poison retry.
        plain=lambda cs,ds:generate(cs,ds)
        result=DecisionLoop.decide_batch([(owner,request)],False,plain)[0]
        self.assertIn(result['decision']['action_id'],game.keys(world))

    def test_stale_bound_owner_is_still_rejected(self):
        game,world,owner,request=fixture();generate=model_for(game,owner,request,world)
        effective,decision=owner.preview(request);owner.last_tick=123
        with self.assertRaisesRegex(ValueError,'stale owned rollout'):
            generate._from_prepared([effective],[decision],owner,request,effective)

    def test_stochastic_and_multiple_owner_calls_do_not_use_private_protocol(self):
        game,world,owner,request=fixture()
        def planner(cs,ds):return None
        def prepared(*args):raise AssertionError('single deterministic resolved owner only')
        planner._from_prepared=prepared
        DecisionLoop.decide_batch([(owner,request)],True,planner)
        from tests.test_decision_loop import fixture as context,request as req
        contexts=[context('one'),context('two')]
        DecisionLoop.decide_batch([(DecisionLoop(c),req(c)) for c in contexts],False,planner)


if __name__=='__main__':unittest.main()
