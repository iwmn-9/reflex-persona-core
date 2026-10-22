import unittest
from tools.confirm_continuation import family_gate,require_registered_rows


def row(game,delta,conditional):return dict(spec=dict(game=game),delta=delta,conditional_delta=conditional)


class ConfirmationGateTests(unittest.TestCase):
    def test_another_family_cannot_compensate_for_a_loss(self):
        self.assertFalse(family_gate([row('a',1.,1.),row('b',-.01,0.)])[0])
        self.assertFalse(family_gate([row('a',1.,1.),row('b',0.,-.01)])[0])

    def test_individual_losses_remain_counted_under_the_unchanged_family_rule(self):
        ok,groups=family_gate([row('a',.1,.1),row('a',-.05,-.05),row('b',0.,0.)])
        self.assertTrue(ok);self.assertEqual(groups[0]['worse'],1)
        self.assertEqual(groups[0]['conditions'],2)

    def test_missing_results_cannot_pass(self):
        self.assertFalse(family_gate([])[0])

    def test_exact_registration_is_required(self):
        specs=[dict(id='a',seed=1),dict(id='b',seed=2)]
        rows=[dict(id=s['id'],spec=s.copy()) for s in specs]
        require_registered_rows(rows,specs)
        with self.assertRaises(AssertionError):require_registered_rows(rows[:1],specs)
        with self.assertRaises(AssertionError):require_registered_rows([rows[0],rows[0]],specs)
        rows[1]['spec']['seed']=3
        with self.assertRaises(AssertionError):require_registered_rows(rows,specs)


if __name__=='__main__':unittest.main()
