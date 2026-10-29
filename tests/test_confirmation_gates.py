import copy
import unittest
from tools.review_adverse_confirmation import numeric_gates, reserved_specs


class ConfirmationGatesTests(unittest.TestCase):
    def fixture(self):
        plan = dict(specs=reserved_specs())
        rows = [dict(id=s['id'], game=s['game'], profile=s['profile'], score_delta=.01, conditional_delta=.01)
                for s in plan['specs']]
        summary = dict(complete=True, pairs_complete=40, pairs=rows,
                       families={g:dict(score_delta=.01, conditional_delta=.01) for g in ('auction','combat','gambling','resources')})
        return plan, dict(original16=copy.deepcopy(summary), stable=copy.deepcopy(summary))

    def test_complete_numeric_pass_still_does_not_adopt(self):
        p,c=self.fixture(); r=numeric_gates(p,c)
        self.assertTrue(r['numeric_gate_passed']); self.assertFalse(r['adoption'])

    def test_one_family_loss_cannot_be_offset(self):
        p,c=self.fixture(); c['stable']['families']['combat']['conditional_delta']=-.001
        self.assertFalse(numeric_gates(p,c)['numeric_gate_passed'])

    def test_missing_or_duplicate_condition_cannot_pass(self):
        for duplicate in (False,True):
            p,c=self.fixture(); rows=c['original16']['pairs'];rows.pop()
            if duplicate:rows.append(copy.deepcopy(rows[0]))
            self.assertFalse(numeric_gates(p,c)['numeric_gate_passed'])

    def test_wrong_seed_or_shorter_world_rejected(self):
        for key,value in (('seed',100),('turns',127)):
            p,c=self.fixture(); p['specs'][0][key]=value
            with self.assertRaises(AssertionError):numeric_gates(p,c)

    def test_two_gambling_world_gains_do_not_pass(self):
        p,c=self.fixture()
        for row in c['original16']['pairs']:
            if row['game']=='gambling' and row['id'].endswith(('irregular','steady')):
                row['score_delta']=-.001
        self.assertFalse(numeric_gates(p,c)['numeric_gate_passed'])


if __name__=='__main__':unittest.main()
