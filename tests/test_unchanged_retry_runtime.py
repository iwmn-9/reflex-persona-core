"""Complete-path reuse must preserve every prefix and hypothetical learner."""
import copy
import unittest

from reflex._continuation_portfolio import portfolio_planner
from reflex.core import digest
from reflex.decision_loop import DecisionLoop
from reflex.laboratory import PROFILES
from tools.continuing_games import ContinuingGame
from tools.preparation_world import PreparationChain
from tools.run_owned_value import cumulative_basis
from tests.test_gain_retry import evaluate


def shared_forecast(scope):
    game=ContinuingGame('combat',PROFILES[2],turns=8)
    world=game.start();actor=game.actor(game.context(world),learning='segmented')
    before=digest(actor.record());request=game.request(game.context(world,actor.state));cap={}
    planner=portfolio_planner(actor,request,world,observe=game.context,events=game.events,
        terminal=game.terminal,assess=lambda w:game.goal(w).record(),request_for=game.request,
        feedback=game.feedback,model_id=game.model_id,target=game.target,horizon=4,
        seeds=(701,702),world_model='posterior',cumulative_preference=cumulative_basis(game,True),
        include_retry=True,captured=cap,purpose_estimator='conditional',tail_recovery=2,
        recovery_cache=64,coherent_cumulative_mean=True,tail_purpose_reflex=True,
        retry_until_gain=True,continuation_scope=scope,capture_prefixes=True)
    result=DecisionLoop.decide_batch([(copy.deepcopy(actor),request)],False,planner)[0]
    assert before==digest(actor.record()) and world==game.start()
    return result,cap


class UnchangedRetryRuntimeTests(unittest.TestCase):
    def test_no_gain_reuses_each_exact_branch_including_prefixes_and_owner(self):
        for scope in ('root','shared'):
            with self.subTest(scope=scope):
                _,cap=shared_forecast(scope);matched=0
                for name,rows in cap['modeled'].items():
                    if not name.endswith('-until-gain'):continue
                    repeated=name.removesuffix('-until-gain')
                    for index,row in enumerate(rows):
                        if row['retry_termination_step'] is not None:continue
                        original=cap['modeled'][repeated][index]
                        self.assertEqual({k:v for k,v in row.items() if k!='retry_termination_step'},original)
                        self.assertIsNot(row,original)
                        self.assertIsNot(row['prefixes'],original['prefixes'])
                        self.assertEqual(cap['paths'][name][index],cap['paths'][repeated][index])
                        self.assertIsNot(cap['paths'][name][index].effects[0],cap['paths'][repeated][index].effects[0])
                        matched+=len(row['actions'])
                self.assertGreater(matched,0)
                self.assertGreaterEqual(cap['forecast'].audit['continuation_portfolio']['reused_steps'],matched)
                self.assertEqual(cap['simulated_steps'],sum(len(r['actions']) for rows in cap['modeled'].values() for r in rows))

    def test_earlier_gain_is_not_lost_when_the_final_goal_returns_to_origin(self):
        game=PreparationChain(5,payout=8.,turns=128)
        _,cap=evaluate(game,True,horizon=8)
        full=cap['modeled']['candidate-001-retry'][0]
        limited=cap['modeled']['candidate-001-retry-until-gain'][0]
        self.assertLessEqual(full['assessment']['value'],game.goal(game.start()).value)
        self.assertEqual(limited['retry_termination_step'],5)
        self.assertNotEqual(limited['actions'],full['actions'])
        self.assertNotEqual(limited['final_owner'],full['final_owner'])


if __name__=='__main__':unittest.main()
