"""Dead-end owner work is removable only where no continuation consumes it."""
import copy
from dataclasses import asdict
import unittest
from unittest.mock import patch

from reflex.closed_loop import ClosedLoop, owned_rollout
from reflex.core import digest
from reflex.decision_loop import DecisionLoop
from reflex.laboratory import PROFILES
from reflex.model_observer import resolve_feedback
from tools.continuing_games import ContinuingGame
from tools.run_adverse_recovery import actor_for
from tools.run_owned_value import cumulative_basis
from tests.test_tail_purpose_reflex import evaluate


class LeafForecastRuntimeTests(unittest.TestCase):
    def test_forecast_choice_actual_learning_and_restore_match_for_every_family_and_profile(self):
        for family in ('auction','combat','gambling','resources'):
            for profile in PROFILES:
                with self.subTest(family=family,profile=profile['id']):
                    game=ContinuingGame(family,profile,16);world=game.start()
                    owner=actor_for(game,game.context(world),'transfer_purpose')
                    # Give the comparison an experienced, changing owner.
                    for _ in range(4):
                        request=game.request(game.context(world,owner.state))
                        chosen=DecisionLoop.decide_batch([(owner,request)],False)[0]
                        key=chosen['decision']['action_id'];symbol=game.events(world,key)[0].symbol
                        after,flow=game.transition(world,key,symbol)
                        resolve_feedback(owner,chosen['ticket'],game.feedback(world,after,key,chosen['context'],flow))
                        world=after
                    request=game.request(game.context(world,owner.state));source=digest(owner.record())
                    outputs=[]
                    for fast in (False,True):
                        actor=copy.deepcopy(owner);captured=[]
                        model=owned_rollout(actor,observe=game.context,events=game.events,terminal=game.terminal,
                            assess=lambda w:game.goal(w).record(),record=lambda w:w,request_for=game.request,
                            feedback=game.feedback,model_id=game.model_id,target=game.target,tail_steps=0,
                            world_model='posterior',cumulative_preference=cumulative_basis(game,True))
                        generate=model.planner(actor,request,world,_forecast_only=fast)
                        def planner(cs,ds):
                            forecast=generate(cs,ds);captured.append(digest(asdict(forecast)));return forecast
                        result=DecisionLoop.decide_batch([(actor,request)],False,planner)[0]
                        key=result['decision']['action_id'];symbol=game.events(world,key)[-1].symbol
                        after,flow=game.transition(world,key,symbol)
                        resolve_feedback(actor,result['ticket'],game.feedback(world,after,key,result['context'],flow))
                        saved=actor.record()
                        restored=DecisionLoop.from_record(game.context(after),saved,predictor=actor.predictor)
                        self.assertEqual(saved,restored.record())
                        outputs.append((captured,result,saved,after))
                    self.assertEqual(outputs[0],outputs[1])
                    self.assertEqual(digest(owner.record()),source)

    def test_full_continuation_keeps_all_simulated_owner_updates(self):
        original=ClosedLoop.planner
        def full(self,*args,**kwargs):
            kwargs.pop('_forecast_only',None)
            return original(self,*args,**kwargs)
        for family in ('auction','combat','gambling','resources'):
            game=ContinuingGame(family,PROFILES[1],16)
            with self.subTest(family=family):
                with patch.object(ClosedLoop,'planner',full):old,old_capture=evaluate(game,game.start(),True,horizon=4)
                new,new_capture=evaluate(game,game.start(),True,horizon=4)
                self.assertEqual(old,new)
                self.assertEqual(old_capture['modeled'],new_capture['modeled'])
                self.assertEqual(old_capture['paths'],new_capture['paths'])
                self.assertEqual(digest(asdict(old_capture['forecast'])),digest(asdict(new_capture['forecast'])))

    def test_optimization_cannot_remove_a_continuation_or_claim_successor_audit(self):
        game=ContinuingGame('combat',PROFILES[0],16);world=game.start()
        owner=game.actor(game.context(world));request=game.request(game.context(world))
        for tail in (0,2):
            model=owned_rollout(owner,observe=game.context,events=game.events,terminal=game.terminal,
                assess=lambda w:game.goal(w).record(),record=lambda w:w,request_for=game.request,
                feedback=game.feedback,model_id=game.model_id,target=game.target,tail_steps=tail)
            with self.assertRaises(ValueError):model.planner(owner,request,world,_forecast_only=True,captured={})
            if tail:
                with self.assertRaises(ValueError):model.planner(owner,request,world,_forecast_only=True)
            with self.assertRaises(ValueError):model.planner(owner,request,world,_forecast_only=1)


if __name__=='__main__':unittest.main()
