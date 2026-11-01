import copy
import unittest
from tools.review_recovery_benefit import aggregate


class RecoveryBenefitSummaryTests(unittest.TestCase):
    def setUp(self):
        self.specs=[dict(id=f,game=f,profile=1) for f in ('auction','combat','resources','gambling')]
        self.rows=[dict(id=s['id'],world_mode=m,experience=e,window_complete=True,actual_steps=32,
            score_increment=1.,conditional_increment=1.1 if e=='learning' and m=='changed' else 1.,
            origin_sha256=s['id'],uniforms=[.5]*32,recovered_within_16=True)
            for s in self.specs for m in ('changed','steady') for e in ('learning','frozen')]

    def test_only_full_paired_family_gates_can_pass(self):
        self.assertTrue(aggregate(self.specs,self.rows)['C3_development_passed'])
        self.assertFalse(aggregate(self.specs,self.rows[:-1])['C3_development_passed'])
        rows=copy.deepcopy(self.rows);rows[0]['conditional_increment']=0.
        self.assertFalse(aggregate(self.specs,rows)['C3_development_passed'])
        rows=copy.deepcopy(self.rows);rows[0]['uniforms'][0]=.1
        with self.assertRaises(ValueError):aggregate(self.specs,rows)
        with self.assertRaises(ValueError):aggregate(self.specs,self.rows+self.rows[:1])

    def test_censored_or_absent_origin_stays_in_denominator(self):
        rows=copy.deepcopy(self.rows)
        for r in rows[:4]:r.update(unavailable=True,window_complete=False,actual_steps=0)
        result=aggregate(self.specs,rows)
        self.assertEqual(result['prediction_denominator'],4)
        self.assertEqual(result['prediction_recovered'],3)
        self.assertFalse(result['C3_development_passed'])
        self.assertIsNone(result['families']['changed']['auction']['mean_delta'])


if __name__=='__main__':unittest.main()
