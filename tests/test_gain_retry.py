"""End retry on public progress, without hard-coding preparation duration."""
import copy
import unittest

from reflex.core import character_key,digest
from reflex.decision_loop import DecisionLoop
from reflex._continuation_portfolio import portfolio_planner
from reflex.laboratory import PROFILES
from tools.preparation_world import PreparationChain
from tools.continuing_games import ContinuingGame
from tools.run_owned_value import cumulative_basis
from tests.test_continuation_persona import IntrinsicTradeoff
from tests.test_continuation_portfolio import RenamedChain


def evaluate(game,enabled=False,horizon=8):
    world=game.start();actor=game.actor(game.context(world),learning='segmented')
    original=digest(actor.record());request=game.request(game.context(world,actor.state));captured={}
    planner=portfolio_planner(actor,request,world,observe=game.context,events=game.events,
        terminal=game.terminal,assess=lambda w:game.goal(w).record(),request_for=game.request,
        feedback=game.feedback,model_id=game.model_id,target=game.target,horizon=horizon,
        seeds=(701,),world_model='posterior',cumulative_preference=cumulative_basis(game,True),
        include_retry=True,captured=captured,purpose_estimator='conditional',tail_recovery=2,
        coherent_cumulative_mean=True,tail_purpose_reflex=True,retry_until_gain=enabled)
    result=DecisionLoop.decide_batch([(copy.deepcopy(actor),request)],False,planner)[0]
    assert digest(actor.record())==original and world==game.start()
    return result,captured


class GainRetryTests(unittest.TestCase):
    def test_every_duration_and_both_payoff_signs(self):
        for delay in range(2,8):
            for good in (True,False):
                with self.subTest(delay=delay,profitable=good):
                    game=PreparationChain(delay,payout=2*delay+2 if good else delay,turns=128)
                    result,cap=evaluate(game,True)
                    self.assertEqual(result['decision']['action_id'],'y' if good else 'x')
                    if good:
                        name=next(n for n in cap['modeled'] if n.startswith('candidate-001-') and n.endswith('-until-gain'))
                        for row in cap['modeled'][name]:
                            self.assertEqual(row['retry_termination_step'],delay)

    def test_original_proposals_learning_and_rename_stay_intact(self):
        game=PreparationChain(5,payout=14.,turns=32)
        _,old=evaluate(game);_,new=evaluate(game,True)
        for name in old['modeled']:
            self.assertEqual(old['modeled'][name],new['modeled'][name])
            self.assertEqual(old['paths'][name],new['paths'][name])
        self.assertEqual(evaluate(RenamedChain(5,payout=14.,turns=32),True)[0]['decision']['action_id'],'first')

    def test_stochastic_original_paths_stay_exact_in_every_rule_family(self):
        for family in ('auction','combat','gambling','resources'):
            with self.subTest(family=family):
                game=ContinuingGame(family,PROFILES[0],turns=128)
                _,old=evaluate(game,horizon=4);result,new=evaluate(game,True,horizon=4)
                self.assertIn(result['decision']['action_id'],game.keys(game.start()))
                for name in old['modeled']:
                    self.assertEqual(old['modeled'][name],new['modeled'][name])
                    self.assertEqual(old['paths'][name],new['paths'][name])

    def test_intrinsic_value_tradeoff_and_need_override_survive(self):
        for profile,mode,names,expected in [(0,'principle',('x','y'),'x'),(2,'principle',('x','y'),'y'),
                (2,'need',('x','y'),'x'),(2,'principle',('zeta','alpha'),'alpha')]:
            with self.subTest(profile=profile,mode=mode,names=names):
                game=IntrinsicTradeoff(PROFILES[profile],names,mode=mode)
                result,_=evaluate(game,True)
                self.assertEqual(result['decision']['action_id'],expected)
                self.assertEqual(character_key(result['context']),character_key(game.context(game.start())))

    def test_strict_boolean(self):
        for value in (1,None,'yes'):
            with self.subTest(value=value),self.assertRaises(ValueError):evaluate(PreparationChain(5),value)

    def test_h32_long_preparation_with_initial_resource_shortage(self):
        for payout,expected in ((24.,'y'),(11.,'x')):
            with self.subTest(payout=payout):
                game=PreparationChain(11,payout=payout,turns=128)
                result,_=evaluate(game,True,horizon=32)
                self.assertEqual(result['decision']['action_id'],expected)


if __name__=='__main__':unittest.main()
