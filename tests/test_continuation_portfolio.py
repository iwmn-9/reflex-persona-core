import copy
import tempfile
import unittest
from pathlib import Path

from reflex.core import digest
from reflex.decision_loop import DecisionLoop
from reflex.laboratory import PROFILES
from tools.continuation_portfolio import portfolio_planner
from tools.continuing_games import ContinuingGame
from tools.preparation_world import PreparationChain
from tools.run_continuation_portfolio import execution
from tools.run_owned_value import cumulative_basis


def choose(game, retry=True, horizon=8, seeds=(701,)):
    world=game.start(); actor=game.actor(game.context(world),learning='segmented')
    req=game.request(game.context(world)); saved=digest(actor.record()); cap={}
    planner=portfolio_planner(actor,req,world,observe=game.context,events=game.events,
        terminal=game.terminal,assess=lambda w:game.goal(w).record(),request_for=game.request,
        feedback=game.feedback,model_id=game.model_id,target=game.target,horizon=horizon,
        seeds=seeds,cumulative_preference=cumulative_basis(game,True),include_retry=retry,captured=cap)
    result=DecisionLoop.decide_batch([(copy.deepcopy(actor),req)],False,planner)[0]
    assert digest(actor.record())==saved
    return result,cap


class RenamedChain(PreparationChain):
    names={'x':'second','y':'first'}

    def keys(self,w):return tuple(self.names[k] for k in super().keys(w))

    def transition(self,w,key,symbol):
        # Call the original rule on a plain adapter so its legality assertion
        # checks the original vocabulary; the controller sees only renamed IDs.
        original=PreparationChain(self.delay,payout=self.payout,turns=self.turns,profile=self.profile)
        actual=next(k for k,v in self.names.items() if v==key)
        after,flow=original.transition(w,actual,symbol);after['last']=key
        return after,flow


class ContinuationPortfolioTests(unittest.TestCase):
    def test_complete_preparation_is_a_candidate_without_mutating_owner(self):
        g=PreparationChain(3,payout=10.,turns=32)
        base,_=choose(g,False);candidate,cap=choose(g)
        self.assertEqual(base['decision']['action_id'],'x')
        self.assertEqual(candidate['decision']['action_id'],'y')
        self.assertGreater(len(cap['modeled']),2)
        renamed,_=choose(RenamedChain(3,payout=10.,turns=32))
        self.assertEqual(renamed['decision']['action_id'],'first')

    def test_unprofitable_preparation_is_rejected_and_terminal_is_absorbing(self):
        result,cap=choose(PreparationChain(3,payout=1.,turns=4),horizon=8)
        self.assertEqual(result['decision']['action_id'],'x')
        for rows in cap['modeled'].values():
            self.assertTrue(all(row['terminal'] and len(row['actions'])==4 for row in rows))

    def test_stochastic_world_keeps_all_roots_and_responses(self):
        g=ContinuingGame('combat',PROFILES[2],turns=8)
        result,cap=choose(g,horizon=2,seeds=(701,702))
        self.assertIn(result['decision']['action_id'],g.keys(g.start()))
        self.assertEqual(len(cap['modeled']),2*len(g.keys(g.start())))
        self.assertTrue(all(len(rows)==6 for rows in cap['modeled'].values()))

    def test_adding_bad_continuations_does_not_relax_the_competence_floor(self):
        # Regression: on this unchanged public state the raw proposal spread
        # doubled the allowed loss even though no better outcome was found.
        g=ContinuingGame('resources',PROFILES[2],turns=128)
        _,base=choose(g,False,horizon=8)
        _,expanded=choose(g,True,horizon=8)
        old,new=base['forecast'],expanded['forecast']
        self.assertAlmostEqual(max(old.purpose.values()),max(new.purpose.values()))
        self.assertGreater(new.audit['purpose_band']['raw_proposal_regret'],old.max_regret)
        self.assertGreaterEqual(max(new.purpose.values())-new.max_regret,
            max(old.purpose.values())-old.max_regret-1e-12)

    def test_resume_keeps_real_owner_and_unannounced_changes_out_of_forecast(self):
        spec=dict(id='resume',mechanism='chain',length=2,payout=8.,changes=[[3,2.]],profile=0,turns=7)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            whole=execution(root/'whole',spec,True)
            self.assertIsNone(execution(root/'resumed',spec,True,max_decisions=2))
            resumed=execution(root/'resumed',spec,True)
            self.assertEqual(resumed['trace'],whole['trace'])
            stable=execution(root/'stable',dict(spec,changes=[]),True)
            for a,b in zip(whole['trace'][:3],stable['trace'][:3]):
                for field in ('before','action','before_owner','predicted_actions','selected_purpose'):
                    self.assertEqual(a[field],b[field])
            self.assertEqual(whole['trace'][3]['before']['payout'],2.)


if __name__=='__main__':unittest.main()
