import copy
import unittest
from tools.review_settled_owner_reference import aggregate


class SettledReferenceSummaryTests(unittest.TestCase):
    def setUp(self):
        self.plan = dict(arms=['stable'], specs=[dict(id='a', kind='chain', profile=0),
                                               dict(id='b', kind='delay', profile=1)])
        self.rows = [dict(spec=s, arm='stable', result=dict(complete=True,
                     reference_admissible=True, normalized_regret=0.)) for s in self.plan['specs']]

    def test_incomparable_root_prevents_a_full_group_pass(self):
        self.rows[1]['result'].update(reference_admissible=False, normalized_regret=None)
        group = aggregate(self.rows, self.plan)['groups']['stable']
        self.assertEqual(group['all']['registered_roots'], 2)
        self.assertEqual(group['all']['comparable'], 1)
        self.assertIsNone(group['all']['numeric_gate'])
        self.assertIsNone(group['all']['full_group'])
        self.assertEqual(group['by_kind']['delay']['status'], 'unmeasured')
        self.assertTrue(group['by_persona']['0']['numeric_gate'])

    def test_large_personal_loss_is_not_hidden_by_grouping(self):
        self.rows[1]['result']['normalized_regret'] = .6
        result = aggregate(self.rows, self.plan)
        self.assertFalse(result['groups']['stable']['all']['numeric_gate'])
        self.assertFalse(result['groups']['stable']['by_persona']['1']['numeric_gate'])
        self.assertFalse(result['adoption'])

    def test_missing_duplicate_and_nonfinite_roots_are_rejected(self):
        for rows in (self.rows[:1], self.rows + self.rows[:1]):
            with self.assertRaises(ValueError): aggregate(rows, self.plan)
        rows = copy.deepcopy(self.rows)
        rows[0]['result']['normalized_regret'] = float('nan')
        with self.assertRaises(ValueError): aggregate(rows, self.plan)


if __name__ == '__main__': unittest.main()
