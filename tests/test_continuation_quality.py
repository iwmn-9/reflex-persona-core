import unittest
from tools.review_continuation import quality_requirements


def cohort():
    return [dict(spec=dict(game=g,profile=p,seed=s),delta=.1,conditional_delta=.1)
        for g in ('auction','combat','resources','gambling')
        for s in range(4 if g=='gambling' else 2) for p in range(4)]


class ContinuationQualityTests(unittest.TestCase):
    def test_equal_results_cannot_count_as_requested_improvement(self):
        rows=cohort()
        for row in rows:row.update(delta=0.,conditional_delta=0.)
        self.assertFalse(quality_requirements(rows)['passed'])

    def test_large_gain_in_one_world_cannot_hide_three_losing_worlds(self):
        rows=cohort()
        for row in rows:
            if row['spec']['game']=='gambling':row['delta']=1. if row['spec']['seed']==0 else -.01
        result=quality_requirements(rows)
        self.assertTrue(result['checks']['positive_gambling'])
        self.assertFalse(result['passed'])

    def test_personality_loss_is_retained_without_demanding_equal_people(self):
        rows=cohort();rows[-1]['delta']=-.01
        result=quality_requirements(rows)
        self.assertTrue(result['passed'])
        self.assertEqual(next(r for r in result['families'] if r['game']=='gambling')['worse'],1)

    def test_missing_persona_is_not_a_complete_independent_world(self):
        rows=cohort();rows.pop()
        self.assertFalse(quality_requirements(rows)['passed'])


if __name__=='__main__':unittest.main()
