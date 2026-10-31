"""Reuse one owned response distribution without replacing conditional rules."""
import copy
import unittest

from reflex.closed_loop import owned_rollout
from reflex.core import digest
from reflex.decision_loop import DecisionLoop
from reflex.laboratory import PROFILES
from tools.continuing_games import ContinuingGame
from tools.run_owned_value import cumulative_basis


def evaluate(family, mode, prepared):
    game=ContinuingGame(family,PROFILES[2],turns=8)
    world=game.start();actor=game.actor(game.context(world),learning='segmented')
    source=digest(actor.record());request=game.request(game.context(world,actor.state))
    model=owned_rollout(actor,observe=game.context,events=game.events,terminal=game.terminal,
        assess=lambda w:game.goal(w).record(),record=lambda w:w,request_for=game.request,
        feedback=game.feedback,model_id=game.model_id,target=game.target,tail_steps=0,
        tree_steps=2,future_choices='reflex',world_model=mode,
        cumulative_preference=cumulative_basis(game,True))
    counts=dict(distributions=0,conditional_actions=0)
    if prepared:
        original=model._prepare_transitions
        def prepare(owner,context):
            counts['distributions']+=1;conditional=original(owner,context)
            def action(w,key):
                counts['conditional_actions']+=1
                return conditional(w,key)
            return action
        model._prepare_transitions=prepare
    else:
        del model._prepare_transitions
    captured={};planner=model.planner(actor,request,world,captured=captured)
    result=DecisionLoop.decide_batch([(copy.deepcopy(actor),request)],False,planner)[0]
    assert digest(actor.record())==source and world==game.start()
    return result,captured,counts


class PreparedResponseTests(unittest.TestCase):
    def test_all_rules_and_world_models_keep_complete_tree_and_owner_evidence(self):
        for family in ('auction','combat','gambling','resources'):
            for mode in ('frozen','posterior','persistent'):
                with self.subTest(family=family,mode=mode):
                    old,old_tree,_=evaluate(family,mode,False)
                    new,new_tree,cost=evaluate(family,mode,True)
                    self.assertEqual(new,old)
                    self.assertEqual(new_tree,old_tree)
                    self.assertGreater(cost['conditional_actions'],cost['distributions'])


if __name__=='__main__':unittest.main()
