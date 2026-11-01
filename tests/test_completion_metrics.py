import copy
import unittest

from tools.completion_metrics import summarize_trace


def rows(predictions, schedule):
    return [dict(tick=t,before={'tick':t},after={'tick':t+1},
                 response_prediction=p,truth=schedule[t],response='a',conditional_increment=0.01)
            for t,p in enumerate(predictions)]


class CompletionMetricsTests(unittest.TestCase):
    def test_recovery_counts_observations_not_including_before_change(self):
        a={'a':.8,'b':.2};b={'a':.2,'b':.8}
        schedule=[a]*32+[b]*32
        trace=rows(schedule,schedule)
        result=summarize_trace(trace,scheduled_turns=64,terminal=True,truth_schedule=schedule)
        self.assertEqual(result['regimes'][1]['recovered_in'],9)
        self.assertEqual(result['prediction_recovery_rate'],1.)
        self.assertTrue(result['behavior_recovery'].startswith('unmeasured'))

    def test_unrecovered_and_early_death_stay_in_denominator(self):
        a={'a':.8,'b':.2};b={'a':.2,'b':.8}
        schedule=[a]*32+[b]*32+[a]*32
        trace=rows([a]*35,schedule)
        result=summarize_trace(trace,scheduled_turns=96,terminal=True,truth_schedule=schedule)
        self.assertEqual(result['qualifying_change_segments'],2)
        self.assertEqual(result['prediction_recovery_rate'],0.)
        self.assertEqual(result['censored_changes'],2)
        self.assertEqual(result['unobserved_changes'],1)

    def test_steady_world_has_no_fake_recovery_success(self):
        law={'a':.8,'b':.2};schedule=[law]*32
        result=summarize_trace(rows(schedule,schedule),scheduled_turns=32,terminal=True,truth_schedule=schedule)
        self.assertIsNone(result['prediction_recovery_rate'])
        self.assertEqual(result['qualifying_change_segments'],0)

    def test_recovered_then_early_termination_is_still_censored(self):
        a={'a':.8,'b':.2};b={'a':.2,'b':.8};schedule=[a]*32+[b]*32
        result=summarize_trace(rows(schedule[:42],schedule),scheduled_turns=64,
            terminal=True,truth_schedule=schedule)
        self.assertEqual(result['regimes'][1]['recovered_in'],9)
        self.assertTrue(result['regimes'][1]['censored'])
        self.assertEqual(result['prediction_recovery_rate'],0.)

    def test_incomplete_observation_and_zero_probability_are_explicit(self):
        q={'a':.8,'b':.2};p={'a':0.,'b':1.};schedule=[q]*32
        result=summarize_trace(rows([p]*4,schedule),scheduled_turns=32,terminal=False,truth_schedule=schedule)
        self.assertFalse(result['complete'])
        self.assertTrue(result['regimes'][0]['censored'])
        self.assertEqual(result['windows'][0]['impossible_observations'],4)

    def test_corrupt_history_and_probabilities_fail(self):
        q={'a':.8,'b':.2};schedule=[q]*32
        original=rows(schedule,schedule)
        for field,value in [('tick',9),('response_prediction',{'a':1.1,'b':-.1}),
                            ('truth',{'a':.7,'b':.3}),('conditional_increment',float('nan'))]:
            trace=copy.deepcopy(original);trace[4][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                summarize_trace(trace,scheduled_turns=32,terminal=True,truth_schedule=schedule)


if __name__=='__main__':unittest.main()
