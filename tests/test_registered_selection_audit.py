"""Preserved forecast units/masks must reproduce the actual personal choice."""
import copy
import json
import unittest

from reflex.core import Policy
from reflex.deliberation import select
from reflex.purpose_plan import goal_forecast
from tests import test_return_means
from tools.probe_registered_owner import personal_ranking


class RegisteredSelectionAuditTests(unittest.TestCase):
    def packet(self, blocked=False):
        context, paths, rows, options = test_return_means.ReturnMeanTests().fixture()
        policy = Policy(principle_priority='finite')
        forecast = goal_forecast(context, paths, rows, **options,
            policy=policy, coherent_cumulative_mean=True)
        progress = dict(applied=True, blocked={'a': 'observed no progress'}) if blocked else None
        allowed = {'b'} if blocked else {'a', 'b'}
        decisions, audit = select([context], forecast, policy, [allowed])
        return json.loads(json.dumps(dict(decision=decisions[0], deliberation=audit,
            roots=forecast.roots, purpose_by_plan=forecast.purpose,
            selection_inputs=dict(base_context=context, forecast_contexts=forecast.contexts,
                target=forecast.target, extra_selectors={}, outcome_limit=forecast.outcome_limit,
                progress=progress, policy=dict(principle_priority=policy.principle_priority,
                    commitments=policy.commitments, residual=policy.residual.tolist())))))

    def test_saved_units_and_corrected_means_preserve_the_native_choice(self):
        packet = self.packet()
        before = copy.deepcopy(packet)
        review = personal_ranking(packet)
        self.assertTrue(review['exact_selection_reproduced'])
        self.assertEqual(packet['decision']['action_id'], 'a')
        selected = packet['deliberation']['selected_plan']
        self.assertTrue(review['plans'][selected]['eligible'])
        self.assertEqual(packet, before)

    def test_actual_progress_exclusion_survives_packet_restore(self):
        packet = self.packet(blocked=True)
        self.assertEqual(packet['decision']['action_id'], 'b')
        review = personal_ranking(packet)
        self.assertTrue(review['exact_selection_reproduced'])
        for key, roots in packet['roots'].items():
            if roots == ['a']:
                self.assertFalse(review['plans'][key]['eligible'])

    def test_contradictory_saved_choice_is_not_explained_away(self):
        packet = self.packet()
        packet['decision']['action_id'] = 'b'
        with self.assertRaises(AssertionError):
            personal_ranking(packet)


if __name__ == '__main__':
    unittest.main()
