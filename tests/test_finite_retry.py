"""Diagnostic correction of the already documented unfinished-cycle failure."""
import copy
import unittest

from reflex.core import digest
from reflex.decision_loop import DecisionLoop
from reflex._continuation_portfolio import portfolio_planner
from tools.preparation_world import PreparationChain
from tools.run_owned_value import cumulative_basis
from tests.test_continuation_portfolio import RenamedChain


def evaluate(game, prefixes=(), horizon=8, world=None, actor=None):
    world=game.start() if world is None else world
    actor=game.actor(game.context(world),learning='segmented') if actor is None else actor
    original=digest(actor.record()); physical=copy.deepcopy(world)
    request=game.request(game.context(world,actor.state)); captured={}
    planner=portfolio_planner(actor,request,world,observe=game.context,events=game.events,
        terminal=game.terminal,assess=lambda w:game.goal(w).record(),request_for=game.request,
        feedback=game.feedback,model_id=game.model_id,target=game.target,horizon=horizon,
        seeds=(701,),world_model='posterior',cumulative_preference=cumulative_basis(game,True),
        include_retry=True,captured=captured,purpose_estimator='conditional',tail_recovery=2,
        coherent_cumulative_mean=True,tail_purpose_reflex=True,retry_prefixes=prefixes)
    result=DecisionLoop.decide_batch([(copy.deepcopy(actor),request)],False,planner)[0]
    assert digest(actor.record())==original and world==physical
    return result,captured


class FiniteRetryTests(unittest.TestCase):
    def test_stop_retry_before_starting_an_unfinished_second_cycle(self):
        game=PreparationChain(5,payout=14.,turns=32)
        before,old=evaluate(game)
        after,new=evaluate(game,(2,4))
        self.assertEqual(before['decision']['action_id'],'x')
        self.assertEqual(after['decision']['action_id'],'y')
        # Original modeled trajectories remain exactly available. Additional
        # proposals do not alter observations, learning, randomness or axes.
        for name in old['modeled']:
            self.assertEqual(old['modeled'][name],new['modeled'][name])
            self.assertEqual(old['paths'][name],new['paths'][name])

    def test_bad_preparation_still_loses_and_opaque_renaming_survives(self):
        bad=PreparationChain(5,payout=1.,turns=32)
        renamed=RenamedChain(5,payout=14.,turns=32)
        self.assertEqual(evaluate(bad,(2,4))[0]['decision']['action_id'],'x')
        self.assertEqual(evaluate(renamed,(2,4))[0]['decision']['action_id'],'first')

    def test_reject_invalid_or_unbounded_prefix_declarations(self):
        for prefix in ((True,), (1,), (8,), (4,2), (2,2), [2]):
            with self.subTest(prefix=prefix),self.assertRaises(ValueError):
                evaluate(PreparationChain(5),prefix)


if __name__=='__main__':unittest.main()
