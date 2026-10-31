"""Do not mistake common positive drift for a return on preparing."""
import copy
import unittest
from reflex._continuation_portfolio import portfolio_planner
from reflex.core import character_key,digest
from reflex.decision_loop import DecisionLoop
from reflex.laboratory import PROFILES
from tools.background_preparation import BackgroundChain
from tools.continuing_games import ContinuingGame
from tools.run_owned_value import cumulative_basis
from tests.test_continuation_persona import IntrinsicTradeoff


def evaluate(game,reference='ordinary',horizon=8,*,seeds=(701,),enabled=True):
    world=game.start();actor=game.actor(game.context(world),learning='segmented')
    original=digest(actor.record());request=game.request(game.context(world,actor.state));cap={}
    planner=portfolio_planner(actor,request,world,observe=game.context,events=game.events,
        terminal=game.terminal,assess=lambda w:game.goal(w).record(),request_for=game.request,
        feedback=game.feedback,model_id=game.model_id,target=game.target,horizon=horizon,
        seeds=seeds,world_model='posterior',cumulative_preference=cumulative_basis(game,True),
        include_retry=True,captured=cap,purpose_estimator='conditional',tail_recovery=2,
        coherent_cumulative_mean=True,tail_purpose_reflex=True,retry_until_gain=enabled,
        retry_gain_reference=reference,capture_prefixes=True)
    result=DecisionLoop.decide_batch([(copy.deepcopy(actor),request)],False,planner)[0]
    assert digest(actor.record())==original and world==game.start()
    return result,cap


class RelativeGainTests(unittest.TestCase):
    def test_positive_common_income_does_not_end_preparation_after_first_step(self):
        game=BackgroundChain(1.125)
        _,old=evaluate(game,'origin');result,new=evaluate(game)
        name='candidate-001-retry-until-gain'
        self.assertEqual(old['modeled'][name][0]['retry_termination_step'],1)
        self.assertEqual(new['modeled'][name][0]['retry_termination_step'],5)
        self.assertEqual(new['modeled'][name][0]['actions'][:5],['y']*5)
        self.assertEqual(character_key(result['context']),character_key(game.context(game.start())))

    def test_original_stochastic_proposals_unchanged_and_reference_is_paired(self):
        for family in ('auction','combat','gambling','resources'):
            with self.subTest(family=family):
                game=ContinuingGame(family,PROFILES[0],turns=128)
                _,old=evaluate(game,'origin',4,seeds=(701,702))
                _,new=evaluate(game,'ordinary',4,seeds=(701,702))
                for name,rows in new['modeled'].items():
                    if not name.endswith('-until-gain'):
                        self.assertEqual(rows,old['modeled'][name])
                        self.assertEqual(new['paths'][name],old['paths'][name])
                        continue
                    root=name.split('-retry')[0]+'-reflex'
                    for row,ordinary in zip(rows,new['modeled'][root]):
                        crossed=[]
                        for i,prefix in enumerate(row['prefixes']):
                            reference=ordinary['prefixes'][min(i,len(ordinary['prefixes'])-1)]['assessment']['value']
                            if prefix['assessment']['value']>reference+1e-12:crossed.append(i+1)
                        self.assertEqual(row['retry_termination_step'],crossed[0] if crossed else None)

    def test_personal_tradeoff_and_needs_remain(self):
        for profile,mode,names,expected in [(0,'principle',('x','y'),'x'),(2,'principle',('x','y'),'y'),
                (2,'need',('x','y'),'x'),(2,'principle',('zeta','alpha'),'alpha')]:
            with self.subTest(profile=profile,mode=mode,names=names):
                game=IntrinsicTradeoff(PROFILES[profile],names,mode=mode)
                result,_=evaluate(game)
                self.assertEqual(result['decision']['action_id'],expected)
                self.assertEqual(character_key(result['context']),character_key(game.context(game.start())))

    def test_explicit_reference_contract(self):
        for value in (None,True,'unknown'):
            with self.subTest(reference=value),self.assertRaises(ValueError):evaluate(BackgroundChain(0.),value)
        with self.assertRaises(ValueError):evaluate(BackgroundChain(0.),enabled=False)


if __name__=='__main__':unittest.main()
