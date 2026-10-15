import unittest
from reflex.examples import context,action,effect
from reflex.progress import Activity,PurposeRequest,PurposeFeedback,ProgressWatch


class ProgressRecoveryTests(unittest.TestCase):
    def fixture(self):
        c=context('recovery',[action(k,effect()) for k in ('restore','wait','work')])
        acts={'restore':Activity('attempt','capacity-restoration'),
            'wait':Activity('wait','temporary-opening',patience=3,release='opening'),
            'work':Activity('attempt','purpose-work')}
        w=ProgressWatch(c['scope']);w.observe(0,'work',PurposeRequest(0.,.8,acts),PurposeFeedback(0.,.8))
        return c,w,acts

    def test_local_recovery_is_not_repeated_failure_but_does_not_reset_global_stall(self):
        c,w,acts=self.fixture();prior_failures=dict(w.failures)
        for tick in range(1,7):
            p=PurposeRequest(0.,(tick-1)*.1,acts)
            self.assertIn('restore',w.mask(c,p)[0])
            audit=w.observe(tick,'restore',p,PurposeFeedback(0.,tick*.1))
            self.assertTrue(audit['method_progress']);self.assertFalse(audit['improved'])
        self.assertEqual(w.failures,prior_failures);self.assertEqual(w.stalls,7);self.assertEqual(w.ready_peak,.8)

    def test_recovery_does_not_renew_expired_wait_or_clear_other_method_failures(self):
        c,w,acts=self.fixture();p=PurposeRequest(0.,.2,acts)
        for tick in range(1,4):w.observe(tick,'wait',p,PurposeFeedback(0.,.2))
        for tick in range(4,7):w.observe(tick,'work',p,PurposeFeedback(0.,.2))
        old=dict(w.failures);w.observe(7,'restore',p,PurposeFeedback(0.,.3))
        self.assertEqual(w.failures,old);allowed,_=w.mask(c,PurposeRequest(0.,.3,acts))
        self.assertNotIn('wait',allowed);self.assertNotIn('work',allowed);self.assertIn('restore',allowed)

    def test_recovery_that_stops_working_is_blocked_again(self):
        c,w,acts=self.fixture();p=PurposeRequest(0.,.2,acts)
        w.observe(1,'restore',p,PurposeFeedback(0.,.3))
        for tick in range(2,5):w.observe(tick,'restore',p,PurposeFeedback(0.,.2))
        self.assertNotIn('restore',w.mask(c,p)[0])

    def test_missing_or_tradeoff_feedback_is_not_recovery_proof(self):
        for fb in (None,PurposeFeedback(-.2,.3)):
            c,w,acts=self.fixture();p=PurposeRequest(0.,.2,acts)
            r=w.observe(1,'restore',p,fb);self.assertFalse(r['method_progress'])


if __name__=='__main__':unittest.main()
