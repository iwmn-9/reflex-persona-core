import copy
import tempfile
import unittest
from pathlib import Path
from reflex.core import digest
from reflex.decision_loop import DecisionLoop
from tools.continuation_portfolio import portfolio_planner
from tools.horizon_consensus import consensus_planner,select_support
from tools.preparation_world import PreparationChain
from tools.run_owned_value import cumulative_basis
from tools.run_continuation_portfolio import execution
from tests.test_continuation_portfolio import RenamedChain


def decision(game,minimum=2,maximum=4,consensus=True):
    w=game.start();actor=game.actor(game.context(w),learning='segmented');before=digest(actor.record())
    request=game.request(game.context(w,actor.state));cap={}
    options=dict(observe=game.context,events=game.events,terminal=game.terminal,
        assess=lambda w:game.goal(w).record(),request_for=game.request,feedback=game.feedback,
        model_id=game.model_id,target=game.target,cumulative_preference=cumulative_basis(game,True),
        purpose_estimator='conditional',coherent_cumulative_mean=True,seeds=(701,),captured=cap,horizon=maximum)
    if consensus:options['minimum_horizon']=minimum
    planner=(consensus_planner if consensus else portfolio_planner)(actor,request,w,**options)
    result=DecisionLoop.decide_batch([(copy.deepcopy(actor),request)],False,planner)[0]
    assert digest(actor.record())==before
    return result,cap


class HorizonConsensusTests(unittest.TestCase):
    def test_one_cutoff_is_original_chooser_with_identical_risk_and_learning_paths(self):
        game=PreparationChain(3,payout=10.,turns=8)
        base,a=decision(game,consensus=False)
        new,b=decision(game,minimum=4)
        self.assertEqual(base['decision'],new['decision'])
        self.assertEqual(a['paths'],b['paths'])
        self.assertEqual(a['forecast'].purpose,b['forecast'].purpose)

    def test_settled_return_is_not_counted_repeatedly_and_action_names_do_not_matter(self):
        base,a=decision(PreparationChain(2,payout=10.,turns=3),minimum=3,maximum=6)
        renamed,b=decision(RenamedChain(2,payout=10.,turns=3),minimum=3,maximum=6)
        self.assertEqual(len(a['horizon_consensus']['ballots']),1)
        self.assertEqual(RenamedChain.names[base['decision']['action_id']],renamed['decision']['action_id'])
        self.assertEqual(sum(a['horizon_consensus']['votes'].values()),1)

    def test_tied_support_uses_longest_owner_choice_not_action_name(self):
        selected,votes=select_support([dict(horizon=2,action='z'),dict(horizon=3,action='a')])
        self.assertEqual(selected['action'],'a')
        with self.assertRaises(ValueError):select_support([dict(horizon=2,action=None)])

    def test_checkpoint_resume_preserves_horizon_votes_and_actual_owner(self):
        spec=dict(id='consensus-resume',mechanism='chain',length=2,payout=8.,changes=[[3,2.]],profile=0,turns=6,
            controller=dict(horizon_consensus=True,purpose_estimator='conditional',coherent_cumulative_mean=True))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            whole=execution(root/'whole',spec,True)
            self.assertIsNone(execution(root/'resume',spec,True,max_decisions=2))
            resumed=execution(root/'resume',spec,True)
            self.assertEqual(resumed['trace'],whole['trace'])
            self.assertEqual(resumed['final'],whole['final'])

if __name__=='__main__':unittest.main()
