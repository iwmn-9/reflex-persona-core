import copy
import unittest

from reflex._continuation_portfolio import portfolio_planner
from reflex.core import digest
from reflex.decision_loop import DecisionLoop
from reflex.laboratory import PROFILES
from tools.continuing_games import ContinuingGame
from tools.cross_evaluated_continuation import cross_evaluated_planner
from tools.run_owned_value import cumulative_basis


def fixture(family):
    game=ContinuingGame(family,PROFILES[1],turns=3)
    world=game.start();actor=game.actor(game.context(world),learning='segmented')
    request=game.request(game.context(world,actor.state))
    options=dict(observe=game.context,events=game.events,terminal=game.terminal,
        assess=lambda w:game.goal(w).record(),request_for=game.request,feedback=game.feedback,
        model_id=game.model_id,target=game.target,horizon=3,seeds=(701,),
        purpose_estimator='conditional',cumulative_preference=cumulative_basis(game,True),
        tail_recovery=2,recovery_cache=64,coherent_cumulative_mean=True,
        continuation_scope='shared',include_retry=True,retry_until_gain=True,retry_gain_reference='expected')
    return game,world,actor,request,options


class CrossEvaluationTests(unittest.TestCase):
    def test_filtered_fixed_rules_preserve_every_original_trajectory(self):
        for family in ('auction','resources','combat','gambling'):
            with self.subTest(family=family):
                _,world,actor,request,options=fixture(family);before=digest(actor.record())
                full={}
                p=portfolio_planner(actor,request,world,captured=full,**options)
                original=DecisionLoop.decide_batch([(copy.deepcopy(actor),request)],False,p)[0]
                policies=full['forecast'].audit['continuation_portfolio']['policies']
                references={n.split('-retry')[0]+'-reflex':p['reference_goal_curve']
                    for n,p in policies.items() if n.endswith('-until-gain')}
                subset={}
                p=portfolio_planner(actor,request,world,captured=subset,policy_names=set(policies),
                    fixed_references=references,**options)
                filtered=DecisionLoop.decide_batch([(copy.deepcopy(actor),request)],False,p)[0]
                self.assertEqual(full['paths'],subset['paths'])
                self.assertEqual(full['modeled'],subset['modeled'])
                self.assertEqual(original['decision'],filtered['decision'])
                self.assertEqual(digest(actor.record()),before)

    def test_cross_evaluation_covers_all_roots_and_freezes_discovered_rules(self):
        _,world,actor,request,options=fixture('resources');before=digest(actor.record());cap={}
        planner=cross_evaluated_planner(actor,request,world,evaluation_seeds=(1701,),captured=cap,**options)
        result=DecisionLoop.decide_batch([(copy.deepcopy(actor),request)],False,planner)[0]
        self.assertTrue(result['deliberation']['adopted'])
        audit=cap['forecast'].audit['cross_evaluation']
        self.assertFalse(set(audit['discovery_seeds'])&set(audit['evaluation_seeds']))
        self.assertTrue(set(audit['ordinary'])<=set(audit['retained']))
        policies=cap['evaluated']['forecast'].audit['continuation_portfolio']['policies']
        for name,p in policies.items():
            if name.endswith('-until-gain'):
                self.assertEqual(p['reference_goal_curve'],tuple(audit['fixed_references'][name.split('-retry')[0]+'-reflex']))
        self.assertLessEqual(cap['forecast'].max_regret,cap['ordinary_forecast'].max_regret)
        self.assertEqual(digest(actor.record()),before)

    def test_invalid_subset_and_overlapping_banks_do_not_commit(self):
        _,world,actor,request,options=fixture('combat');before=digest(actor.record())
        with self.assertRaises(ValueError):
            cross_evaluated_planner(actor,request,world,evaluation_seeds=(701,),**options)
        for names in ({'not-a-continuation'}, {'candidate-000-reflex'}):
            with self.assertRaises(ValueError):
                planner=portfolio_planner(actor,request,world,policy_names=names,**options)
                DecisionLoop.decide_batch([(copy.deepcopy(actor),request)],False,planner)
        with self.assertRaises(ValueError):
            portfolio_planner(actor,request,world,policy_names={'candidate-000-reflex'},
                fixed_references={'candidate-000-reflex':[float('nan')]*3},**options)
        self.assertEqual(digest(actor.record()),before)


if __name__=='__main__':unittest.main()
