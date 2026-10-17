import copy
import unittest
from tools.diagnose_auction_continuity import reference
from tools.continuing_games import ContinuingGame
from reflex.laboratory import PROFILES
from tools.run_owned_value import run
from tools.audit_continuation_returns import audit_run


class ContinuityDiagnosticTests(unittest.TestCase):
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
