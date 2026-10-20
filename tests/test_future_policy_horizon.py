"""Delayed preparation is a public consequence, never an action-name bonus."""
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from reflex.closed_loop import owned_rollout
from reflex.core import character_key,digest
from reflex.decision_loop import DecisionLoop
from reflex.receding_owner import RecedingOwner
from tools.run_owned_value import cumulative_basis
from tools.preparation_world import DelayedYield


def options(g,horizon):
    return dict(observe=g.context,events=g.events,terminal=g.terminal,
        assess=lambda w:g.goal(w).record(),record=lambda w:w,
        request_for=g.request,feedback=g.feedback,model_id=g.model_id,target=g.target,
        tail_steps=15,world_model='posterior',purpose_estimator='conditional',
        tail_recovery=2,tail_recovery_mode='always',tail_recovery_limit=15,
        tail_recovery_horizon=horizon,cumulative_preference=cumulative_basis(g,True))


def future_choice(g,w,horizon):
    c=g.context(w);own=g.actor(c,learning='segmented');before=digest(own.record())
    controller=owned_rollout(own,**options(g,horizon));req=g.request(c)
    planner=controller.frontier.future_planner(own,req,w)
    detached=copy.deepcopy(own)
    result=DecisionLoop.decide_batch([(detached,req)],False,planner)[0]
    assert digest(own.record())==before
    assert character_key(result['context'])==character_key(c)
    return result['decision']['action_id']


class FuturePolicyHorizonTests(unittest.TestCase):
    def test_long_delays_and_small_immediate_gains_use_the_same_persona(self):
        for delay in (3,7,11):
            with self.subTest(delay=delay):
                g=DelayedYield(delay);w=g.start()
                self.assertEqual(future_choice(g,w,2),'x')
                self.assertEqual(future_choice(g,w,16),'y')

    def test_reduced_payback_and_deadline_make_preparation_unhelpful(self):
        g=DelayedYield(7,payout=2.)
        self.assertEqual(future_choice(g,g.start(),16),'x')
        g=DelayedYield(7);late=g.start();late['tick']=22
        self.assertEqual(future_choice(g,late,16),'x')

    def test_invalid_horizon_and_shared_search_seeds_are_rejected(self):
        g=DelayedYield(3);own=g.actor(g.context(g.start()),learning='segmented')
        for horizon in (1,129,True,2.0):
            with self.subTest(horizon=horizon),self.assertRaises(ValueError):
                RecedingOwner(own,**options(g,horizon))
        with self.assertRaisesRegex(ValueError,'separate'):
            owned_rollout(own,**options(g,16),seeds=(1701,))
        with self.assertRaisesRegex(ValueError,'complete-tail'):
            owned_rollout(own,**dict(options(g,16),tail_recovery_mode='stalled'))

    def test_future_horizon_identifies_the_policy_but_does_not_change_default(self):
        g=DelayedYield(3);own=g.actor(g.context(g.start()),learning='segmented')
        a=owned_rollout(own,**options(g,2))
        b=owned_rollout(own,**options(g,16))
        self.assertNotEqual(a.frontier.identity,b.frontier.identity)
        cfg=options(g,2);cfg.pop('tail_recovery_horizon')
        self.assertEqual(a.frontier.identity,owned_rollout(own,**cfg).frontier.identity)

    def test_registered_numeric_option_survives_frozen_cli_resume(self):
        from tools.freeze_experiment import freeze
        from tools.resume_horizon_study import resume
        from tools.value_contract_config import configurations,arguments
        cfg=dict(tail_replanning=True,tail_replanning_horizon=8,cumulative_achievement=True)
        plan={'configurations':{'baseline':dict(tail_replanning=True),'candidate':cfg}}
        self.assertEqual(configurations(plan)['candidate'],cfg)
        self.assertIn('8',arguments(cfg))
        for bad in (True,2,129,8.0):
            with self.subTest(horizon=bad),self.assertRaises(ValueError):
                configurations({'configurations':{'baseline':{},'candidate':dict(cfg,tail_replanning_horizon=bad)}})
        job=['gambling',0,57993,'outside',2,1,1,0]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'run';freeze(root)
            code="""import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from tools.run_owned_value import run
from tools.run_horizon_study import __doc__ as semantics
root=Path(sys.argv[2]);job=json.loads(sys.argv[3]);cfg=json.loads(sys.argv[4])
(root/'registration.json').write_text(json.dumps(dict(jobs=[job],arms=['receding_recovery'],world_seeds=[job[2]],stage='diagnostic',semantics=semantics,configuration=cfg)))
assert run(job,'receding_recovery',checkpoint=root/'checkpoint.json',max_decisions=1,**cfg) is None
try:
    run(job,'receding_recovery',checkpoint=root/'checkpoint.json',**dict(cfg,tail_replanning_horizon=16))
except ValueError as error:
    assert 'contract changed' in str(error)
else: raise AssertionError('resumed future policy changed')
"""
            subprocess.run([sys.executable,'-c',code,str(root/'_source'),str(root),json.dumps(job),json.dumps(cfg)],
                check=True,capture_output=True)
            proof=resume(root,hashlib.sha256((root/'source_snapshot.json').read_bytes()).hexdigest(),job)
            self.assertTrue(proof['original_prefix_unchanged'])
            row=json.loads((root/'trajectories.jsonl').read_bytes())
            self.assertEqual(row['tail_replanning_horizon'],8)
            self.assertEqual(json.loads((root/'audit.json').read_bytes())['checks']['restored_action_feedback'],2)

    def test_long_play_driver_keeps_resolved_history_and_observes_public_shock(self):
        from tools.run_preparation_horizon import execution
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(execution(folder,delay=3,shock=True,future_horizon=8,
                turns=3,outer_horizon=2,max_decisions=1))
            prior=json.loads((Path(folder)/'checkpoint.json').read_bytes())['payload']
            result=execution(folder,delay=3,shock=True,future_horizon=8,turns=3,outer_horizon=2)
            self.assertEqual(prior['trace'],result['trace'][:1])
            self.assertEqual(result['trace'][0]['before']['payout'],6.)
            self.assertEqual(result['trace'][0]['after']['payout'],2.)
            self.assertEqual(result['checks'],dict(fixed_axes=3,owner_parity=3))
            self.assertEqual(result['physical_replay'],3)


if __name__=='__main__':unittest.main()
