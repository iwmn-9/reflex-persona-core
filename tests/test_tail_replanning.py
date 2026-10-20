"""Future competence must not disappear merely because immediate gain is positive."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from reflex.closed_loop import owned_rollout
from reflex.core import character_key
from reflex.decision_loop import DecisionLoop
from reflex.laboratory import PROFILES
from reflex.receding_owner import RecedingOwner
from reflex.sampling_check import DecisionSampling, SampleBanks
from tools.continuing_games import ContinuingGame, GAMES
from tools.run_owned_value import cumulative_basis, run


def options(g, **extra):
    return dict(observe=g.context, events=g.events, terminal=g.terminal,
        assess=lambda w:g.goal(w).record(), record=lambda w:w,
        request_for=g.request, feedback=g.feedback, model_id=g.model_id,
        target=g.target, tail_steps=2, seeds=(701,), world_model='posterior',
        purpose_estimator='conditional', tail_recovery=2,
        cumulative_preference=cumulative_basis(g, True), **extra)


class TailReplanningTests(unittest.TestCase):
    def test_small_positive_gain_cannot_disable_preparation(self):
        g=ContinuingGame('combat',PROFILES[0],8)
        w=g.start();w.update(ammo=0,hp=11.75)
        c=g.context(w);own=g.actor(c,learning='segmented');req=g.request(c)
        saved=copy.deepcopy(own.record())
        old=owned_rollout(own,**options(g))
        new=owned_rollout(own,**options(g,tail_recovery_mode='always'))
        self.assertGreater(g.goal(g.transition(w,'guard','0')[0]).value,g.goal(w).value)
        self.assertIsNone(old.frontier.future_planner(own,req,w))
        planner=new.frontier.future_planner(own,req,w)
        self.assertIsNotNone(planner)
        detached=copy.deepcopy(own)
        chosen=DecisionLoop.decide_batch([(detached,req)],False,planner)[0]
        self.assertEqual(chosen['decision']['action_id'],'reload')
        self.assertEqual(own.record(),saved)
        self.assertNotEqual(old.frontier.identity,new.frontier.identity)
        # Preparation is a choice, not a bonus or a mandatory action. With
        # that capacity already available, spending it can be preferable.
        ready=g.start();ready['ammo']=3;cc=g.context(ready)
        person=g.actor(cc,learning='segmented');request=g.request(cc)
        controller=owned_rollout(person,**options(g,tail_recovery_mode='always'))
        choice=DecisionLoop.decide_batch([(person,request)],False,
            controller.frontier.future_planner(person,request,ready))[0]
        self.assertTrue(choice['decision']['action_id'].startswith('strike:'))

    def test_every_future_step_learns_and_uses_same_fixed_person_in_all_families(self):
        for game in GAMES:
            g=ContinuingGame(game,PROFILES[2],3);w=g.start();own=g.actor(g.context(w),learning='segmented')
            saved=copy.deepcopy(own.record());fixed=character_key(g.context(w))
            ctrl=owned_rollout(own,**options(g,tail_recovery_mode='always'))
            original=ctrl.frontier.future_planner;seen=[]
            def observed(person,request,state):
                self.assertEqual(character_key(request.context),fixed)
                seen.append(person.predictor.observations)
                return original(person,request,state)
            ctrl.frontier.future_planner=observed
            sample=ctrl.frontier.distribution(w,own)
            self.assertEqual(seen,[0,1])
            self.assertEqual(ctrl.frontier.future_calls,2)
            self.assertEqual(len(sample.trajectories[0][2]),2)
            self.assertEqual(own.record(),saved)

    def test_bank_pooling_matches_fresh_replanning_without_changing_person(self):
        g=ContinuingGame('gambling',PROFILES[1],3);w=g.start();own=g.actor(g.context(w),learning='segmented')
        args=options(g,tail_recovery_mode='always')
        banks=SampleBanks(own,args,DecisionSampling(((701,),(711,),(721,))))
        banks.evaluate(w,(0,));banks.evaluate(w,(1,));pooled=banks.evaluate(w,(0,1))
        other=copy.deepcopy(own);ctrl=owned_rollout(other,**dict(args,seeds=(701,711)))
        req=g.request(g.context(w));cap={}
        fresh=DecisionLoop.decide_batch([(other,req)],False,ctrl.planner(other,req,w,captured=cap))[0]
        self.assertEqual(pooled['captured']['node']['forecast'],cap['node']['forecast'])
        self.assertEqual(pooled['result']['decision'],fresh['decision'])

    def test_entire_tail_budget_is_required_and_ordinary_default_is_unchanged(self):
        g=ContinuingGame('gambling',PROFILES[0],3);w=g.start();own=g.actor(g.context(w),learning='segmented')
        args=options(g)
        for extra in ({'tail_recovery_mode':'unknown'},
                      {'tail_recovery_mode':'always','tail_recovery_limit':1},
                      {'tail_recovery_mode':'always','tail_recovery':0}):
            with self.assertRaises(ValueError):owned_rollout(own,**dict(args,**extra))
        left=RecedingOwner(copy.deepcopy(own),**args).decide(w)
        right=RecedingOwner(copy.deepcopy(own),**args,tail_recovery_mode='stalled').decide(w)
        self.assertEqual(left,right)

    def test_checkpoint_records_future_policy_and_disallows_changing_it_mid_world(self):
        job=['gambling',0,57991,'shift-return',3,1,2,0]
        cfg=dict(tail_replanning=True,cumulative_achievement=True,tail_goal_refinement=True,model_seeds=(701,))
        expected=run(job,'receding_recovery',**cfg)
        with tempfile.TemporaryDirectory() as directory:
            cp=Path(directory)/'checkpoint.json'
            self.assertIsNone(run(job,'receding_recovery',checkpoint=cp,max_decisions=1,**cfg))
            wrong=dict(cfg,tail_replanning=False)
            with self.assertRaisesRegex(ValueError,'contract changed'):
                run(job,'receding_recovery',checkpoint=cp,**wrong)
            actual=run(job,'receding_recovery',checkpoint=cp,**cfg)
            expected.pop('seconds');actual.pop('seconds')
            self.assertEqual(actual,expected)
            self.assertTrue(actual['tail_replanning'])
            self.assertEqual(json.loads(cp.read_text())['payload']['contract']['configuration']['tail_replanning'],True)

    def test_incomplete_future_planner_never_silently_becomes_reflex(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        g=ContinuingGame('combat',PROFILES[0],3);w=g.start();c=g.context(w)
        own=g.actor(c,learning='segmented')
        ctrl=owned_rollout(own,**options(g,tail_recovery_mode='always'))
        with patch('reflex.closed_loop.owned_rollout',
                   return_value=SimpleNamespace(planner=lambda *args:lambda *inputs:None)):
            planner=ctrl.frontier.future_planner(own,g.request(c),w)
        with self.assertRaisesRegex(ValueError,'no silent reflex'):
            planner([c],[])

    def test_frozen_cli_resume_and_source_audit_keep_new_configuration(self):
        import subprocess,sys,hashlib
        from tools.freeze_experiment import freeze
        from tools.resume_horizon_study import resume
        job=['gambling',0,57992,'outside',2,1,1,0]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'run';freeze(root)
            code="""import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from tools.run_owned_value import run
from tools.run_horizon_study import __doc__ as semantics
root=Path(sys.argv[2]);job=json.loads(sys.argv[3])
cfg=dict(tail_replanning=True,cumulative_achievement=True)
(root/'registration.json').write_text(json.dumps(dict(jobs=[job],arms=['receding_recovery'],world_seeds=[job[2]],stage='diagnostic',semantics=semantics,configuration=cfg)))
assert run(job,'receding_recovery',checkpoint=root/'checkpoint.json',max_decisions=1,**cfg) is None
"""
            subprocess.run([sys.executable,'-c',code,str(root/'_source'),str(root),json.dumps(job)],
                           check=True,capture_output=True)
            proof=resume(root,hashlib.sha256((root/'source_snapshot.json').read_bytes()).hexdigest(),job)
            self.assertTrue(proof['original_prefix_unchanged'])
            result=json.loads((root/'trajectories.jsonl').read_text())
            self.assertTrue(result['tail_replanning'])
            self.assertEqual(json.loads((root/'audit.json').read_text())['checks']['restored_action_feedback'],2)


if __name__=='__main__':unittest.main()
