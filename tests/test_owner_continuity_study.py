import copy
import unittest
from tools.diagnose_auction_continuity import reference
from tools.continuing_games import ContinuingGame
from reflex.laboratory import PROFILES
from tools.run_owned_value import run
from tools.audit_continuation_returns import audit_run
from reflex.closed_loop import owned_rollout
from reflex.decision_loop import DecisionLoop
from reflex.model_observer import resolve_feedback
from reflex.core import digest


class ContinuityDiagnosticTests(unittest.TestCase):
    def test_recovery_reference_replay_uses_the_declared_future_controller(self):
        g=ContinuingGame('combat',PROFILES[1],turns=8,variant=1);w=g.start();w['ammo']=0
        own=g.actor(g.context(w),learning='segmented')
        controller=owned_rollout(own,observe=g.context,events=g.events,terminal=g.terminal,
            assess=lambda w:g.goal(w).record(),record=lambda w:w,request_for=g.request,
            feedback=g.feedback,model_id=g.model_id,target=g.target,tail_steps=7,
            world_model='posterior',purpose_estimator='conditional',tail_recovery=2)
        trace=[];calls=0
        for tick in range(8):
            c=g.context(w,own.state);req=g.request(c);checkpoint=own.record();plan=None
            if tick:
                plan=controller.frontier.future_planner(copy.deepcopy(own),req,w)
                calls+=plan is not None
            result=DecisionLoop.decide_batch([(own,req)],False,plan,
                _counterfactual_roots=('guard',) if tick==0 else None)[0]
            key=result['decision']['action_id'];after,flow=g.transition(w,key,'2')
            resolve_feedback(own,result['ticket'],g.feedback(w,after,key,result['context'],flow))
            trace.append(dict(tick=tick,before=w,after=after,action=key,response='2',goal=g.goal(after).value,
                checkpoint=checkpoint,before_owner=digest(checkpoint),owner=digest(own.record()),
                planned=True,forecast_end=8,predicted=0.))
            w=after
        row=dict(game='combat',profile='steady',seed=11,regime='outside',arm='receding_recovery',
            turns=8,tail_steps=7,interval=1,trace=trace)
        checked=audit_run(row)['cases'][0]
        self.assertGreater(calls,0)
        self.assertEqual(checked['future_calls'],calls)
        self.assertEqual(checked['actual_actions'],checked['reference_actions'])
        self.assertEqual(checked['control_contribution'],0.)
        self.assertEqual(checked['realized_persona_gain'],0.)

    def test_actual_root_checkpoint_matches_exact_replay(self):
        r=run(('auction',0,11,'outside',4,1,3),'receding_posterior')
        stored=audit_run(r)
        replay=copy.deepcopy(r);replay['trace'][1].pop('checkpoint')
        reconstructed=audit_run(replay)
        self.assertEqual(stored['summary'],reconstructed['summary'])
        self.assertEqual(stored['cases'][0]['root_owner'],'actual-checkpoint')
        self.assertEqual(reconstructed['cases'][0]['root_owner'],'exact-feedback-replay')
        wrong=copy.deepcopy(r);wrong['trace'][1]['checkpoint']['last_tick']=99
        with self.assertRaises(AssertionError):audit_run(wrong)

    def test_reference_only_execution_has_zero_control_contribution(self):
        r=run(('auction',0,11,'outside',4,7,3),'endpoint')
        checked=audit_run(r)
        self.assertEqual(len(checked['cases']),1)
        case=checked['cases'][0]
        self.assertEqual(case['actual_actions'],case['reference_actions'])
        self.assertEqual(case['control_contribution'],0.)
        self.assertEqual(case['realized_persona_gain'],0.)
        self.assertEqual(case['total_gap'],case['world_and_sampling_residual'])

    def test_evaluator_only_reference_matches_full_public_tree(self):
        for variant in (0,1):
            g=ContinuingGame('auction',PROFILES[0],turns=3,variant=variant)
            laws=((.8,.1,.1),(.1,.1,.8),(.1,.8,.1))
            fast=reference(3,variant,lambda t:laws[t])
            def q(w,k):
                return sum(p*v(g.transition(w,k,str(z))[0]) for z,p in enumerate(laws[w['tick']]))
            def v(w):
                return g.goal(w).value if g.terminal(w) else max(q(w,k) for k in g.keys(w))
            w=g.start()
            for key,value in fast(0,8).items():self.assertAlmostEqual(value,q(w,key),places=13)


if __name__=='__main__':unittest.main()
