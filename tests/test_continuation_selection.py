import copy
import unittest
from tools.select_continuation import choose_candidate


class SimpleFirstSelectionTests(unittest.TestCase):
    def setUp(self):
        self.choices = {name: dict(complete=True, preparation_no_loss=True,
            no_family_loss=True, families=[]) for name in ('fixed32', 'consensus')}

    def test_sufficient_fixed_horizon_wins_without_score_maximization(self):
        self.choices['consensus']['families'] = [dict(mean_delta=100)]
        before = copy.deepcopy(self.choices)
        self.assertEqual(choose_candidate(self.choices), 'fixed32')
        self.assertEqual(self.choices, before)

    def test_complex_candidate_requires_a_failed_simple_gate(self):
        for gate in ('preparation_no_loss', 'no_family_loss'):
            with self.subTest(gate=gate):
                choices = copy.deepcopy(self.choices)
                choices['fixed32'][gate] = False
                self.assertEqual(choose_candidate(choices), 'consensus')
                choices['consensus'][gate] = False
                self.assertIsNone(choose_candidate(choices))

    def test_partial_evidence_cannot_select_either_candidate(self):
        for name in self.choices:
            choices = copy.deepcopy(self.choices)
            choices[name]['complete'] = False
            with self.assertRaises(ValueError):
                choose_candidate(choices)


if __name__ == '__main__':
    unittest.main()
