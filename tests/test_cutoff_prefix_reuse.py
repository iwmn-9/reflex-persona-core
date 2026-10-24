import copy
import unittest
from tools.reuse_cutoff_prefix import project_entry


class CutoffPrefixTests(unittest.TestCase):
    def entry(self):
        last=dict(horizon=32,action='a',purpose=.5)
        return dict(action='a',before_owner='owner',selected_purpose=.5,
            horizon_consensus=dict(maximum_horizon=32,minimum_horizon=16,
                source_owner='owner',ballots=[dict(horizon=16,action='b',purpose=.4),last],
                selected=last,votes={'a':1,'b':1}))

    def test_reused_forecast_is_exactly_the_single_long_cutoff(self):
        entry=self.entry();before=copy.deepcopy(entry);actual=project_entry(entry)
        self.assertEqual(entry,before)
        self.assertEqual(actual['horizon_consensus']['ballots'],[entry['horizon_consensus']['ballots'][-1]])
        self.assertEqual(actual['horizon_consensus']['minimum_horizon'],32)

    def test_disagreement_cannot_be_replayed_as_fixed_horizon(self):
        entry=self.entry();entry['action']='b'
        self.assertIsNone(project_entry(entry))

    def test_equal_action_without_same_forecast_or_owner_is_rejected(self):
        for field in ('selected','source_owner','purpose'):
            with self.subTest(field=field):
                entry=self.entry()
                if field=='selected':entry['horizon_consensus']['selected']=entry['horizon_consensus']['ballots'][0]
                elif field=='source_owner':entry['horizon_consensus'][field]='other'
                else:entry['selected_purpose']=.4
                with self.assertRaises(AssertionError):project_entry(entry)


if __name__=='__main__':unittest.main()
