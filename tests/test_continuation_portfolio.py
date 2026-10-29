import copy
import tempfile
import unittest
from pathlib import Path

from reflex.core import digest
from reflex.closed_loop import owned_rollout
from reflex.decision_loop import DecisionLoop
from reflex.laboratory import PROFILES
from tools.continuation_portfolio import portfolio_planner
from tools.continuing_games import ContinuingGame
from tools.preparation_world import PreparationChain
from tools.run_continuation_portfolio import execution
from tools.run_owned_value import cumulative_basis


def choose(game, retry=True, horizon=8, seeds=(701,), purpose_estimator='sampled',tail_recovery=0,recovery_cache=0):
    world=game.start(); actor=game.actor(game.context(world),learning='segmented')
    req=game.request(game.context(world)); saved=digest(actor.record()); cap={}
    planner=portfolio_planner(actor,req,world,observe=game.context,events=game.events,
        terminal=game.terminal,assess=lambda w:game.goal(w).record(),request_for=game.request,
        feedback=game.feedback,model_id=game.model_id,target=game.target,horizon=horizon,
        seeds=seeds,cumulative_preference=cumulative_basis(game,True),include_retry=retry,captured=cap,
        purpose_estimator=purpose_estimator,tail_recovery=tail_recovery,recovery_cache=recovery_cache)
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
        # The added retry of an unprofitable construction is deterministically
        # worse. Do not assume a stochastic resource continuation remains bad
        # after the simulated owner's selection behavior improves.
        g=PreparationChain(3,payout=1.,turns=128,profile=PROFILES[2])
        _,base=choose(g,False,horizon=8)
        _,expanded=choose(g,True,horizon=8)
        old,new=base['forecast'],expanded['forecast']
        self.assertAlmostEqual(max(old.purpose.values()),max(new.purpose.values()))
        self.assertGreater(new.audit['purpose_band']['raw_proposal_regret'],old.max_regret)
        self.assertGreaterEqual(max(new.purpose.values())-new.max_regret,
            max(old.purpose.values())-old.max_regret-1e-12)

    def test_improved_continuations_preserve_existing_paths_and_competence_floor(self):
        # Retain the original regression world. Extra continuations may now
        # improve its best value; they still may not erase the old alternatives
        # or lower the competence floor.
        g=ContinuingGame('resources',PROFILES[2],turns=128)
        _,base=choose(g,False,horizon=8)
        _,expanded=choose(g,True,horizon=8)
        old,new=base['forecast'],expanded['forecast']
        # Display-plan indexes shift when extra proposals are inserted. Match
        # the actual continuation identity, not its position in the bank.
        new_by_proposal={new.audit['plans'][name]['proposal']:value
                         for name,value in new.purpose.items()}
        for name,value in old.purpose.items():
            self.assertEqual(value,new_by_proposal[old.audit['plans'][name]['proposal']])
        self.assertGreaterEqual(max(new.purpose.values()),max(old.purpose.values()))
        self.assertGreaterEqual(max(new.purpose.values())-new.max_regret,
            max(old.purpose.values())-old.max_regret-1e-12)

    def test_conditional_mean_keeps_sampled_learning_and_risk_paths(self):
        g=ContinuingGame('combat',PROFILES[0],turns=8)
        _,sampled=choose(g,horizon=2,seeds=(701,702,703))
        _,conditional=choose(g,horizon=2,seeds=(701,702,703),purpose_estimator='conditional')
        self.assertEqual(sampled['paths'],conditional['paths'])
        for key,rows in conditional['modeled'].items():
            for i,row in enumerate(rows):
                actual=dict(row);actual.pop('purpose_mean_estimate')
                self.assertEqual(actual,sampled['modeled'][key][i])
            # Exactly one future decision: conditioned on the enumerated root
            # response, averaging every public next response removes all tail
            # outcome sampling noise. The person's simulated state still varies.
            for offset in range(0,len(rows),3):
                values=[r['purpose_mean_estimate'] for r in rows[offset:offset+3]]
                self.assertAlmostEqual(max(values),min(values))

    def test_original_owned_recovery_is_preserved_without_extra_continuations(self):
        class EmptyInventory(ContinuingGame):
            def start(self):
                w=super().start();w['ammo']=0;return w
        g=EmptyInventory('combat',PROFILES[0],turns=8)
        new,cap=choose(g,False,horizon=4,seeds=(701,),purpose_estimator='conditional',tail_recovery=2)
        w=g.start();actor=g.actor(g.context(w),learning='segmented');request=g.request(g.context(w));old={}
        model=owned_rollout(actor,observe=g.context,events=g.events,terminal=g.terminal,
            assess=lambda w:g.goal(w).record(),record=lambda w:w,request_for=g.request,
            feedback=g.feedback,model_id=g.model_id,target=g.target,tail_steps=3,seeds=(701,),
            purpose_estimator='conditional',tail_recovery=2,world_model='posterior',
            cumulative_preference=cumulative_basis(g,True))
        result=DecisionLoop.decide_batch([(copy.deepcopy(actor),request)],False,
            model.planner(actor,request,w,captured=old))[0]
        before=old['node']['forecast'];after=cap['forecast']
        def means(f):return {f.roots[k][0]:v for k,v in f.purpose.items()}
        self.assertEqual(set(means(before)),set(means(after)))
        for root,value in means(before).items():self.assertAlmostEqual(value,means(after)[root])
        self.assertEqual(result['decision']['action_id'],new['decision']['action_id'])
        self.assertTrue(any(row['recovery_calls'] for rows in cap['modeled'].values() for row in rows))

    def test_recovery_cache_reuses_only_identical_complete_owners_and_worlds(self):
        class EmptyInventory(ContinuingGame):
            def start(self):
                w=super().start();w['ammo']=0;return w
        g=EmptyInventory('combat',PROFILES[2],turns=8)
        args=dict(horizon=4,seeds=(701,702,703),purpose_estimator='conditional',tail_recovery=2)
        old,a=choose(g,**args);new,b=choose(g,**args,recovery_cache=64)
        for result in (old,new):
            result['deliberation']['metadata']['continuation_portfolio'].pop('recovery_cache')
        self.assertEqual(old,new)
        self.assertEqual(a['modeled'],b['modeled'])
        self.assertEqual(a['paths'],b['paths'])
        self.assertEqual(a['forecast'].purpose,b['forecast'].purpose)
        self.assertGreater(b['forecast'].audit['continuation_portfolio']['recovery_cache']['hits'],0)

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
