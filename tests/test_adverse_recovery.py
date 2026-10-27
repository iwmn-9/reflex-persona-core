import copy
import json
from pathlib import Path
import tempfile
import unittest

from reflex.core import character_key
from reflex.decision_loop import DecisionLoop
from reflex.laboratory import PROFILES
from reflex._continuation_portfolio import portfolio_planner
from tools.continuing_games import ContinuingGame
from tools.run_adverse_recovery import run
from tools.adverse_segment import audit
from tools.run_owned_value import cumulative_basis


def forecast(sampling=None):
    g = ContinuingGame('combat', PROFILES[1], turns=3)
    w = g.start()
    actor = g.actor(g.context(w), learning='segmented')
    request = g.request(g.context(w))
    before = actor.record()
    captured = {}
    options = {} if sampling is None else dict(sampling=sampling)
    planner = portfolio_planner(actor, request, w, observe=g.context, events=g.events,
        terminal=g.terminal, assess=lambda w:g.goal(w).record(), request_for=g.request,
        feedback=g.feedback, model_id=g.model_id, target=g.target, horizon=3,
        seeds=(701,702,703), captured=captured, cumulative_preference=cumulative_basis(g,True),
        purpose_estimator='conditional', coherent_cumulative_mean=True, **options)
    chosen = DecisionLoop.decide_batch([(copy.deepcopy(actor), request)], False, planner)[0]
    assert actor.record() == before
    return chosen, captured


class AdverseRecoveryTests(unittest.TestCase):
    def test_default_and_explicit_independent_are_identical(self):
        a, left = forecast()
        b, right = forecast('independent')
        self.assertEqual(a, b)
        self.assertEqual(left, right)

    def test_stratification_keeps_probability_mass_candidates_and_persona(self):
        independent, left = forecast('independent')
        stratified, right = forecast('stratified')
        self.assertEqual(left['forecast'].roots, right['forecast'].roots)
        self.assertEqual(character_key(independent['context']), character_key(stratified['context']))
        self.assertEqual(left['simulated_steps'], right['simulated_steps'])
        self.assertEqual(right['forecast'].audit['continuation_portfolio']['sampling'], 'stratified')
        self.assertNotEqual(left['paths'], right['paths'])
        for paths in right['paths'].values():
            self.assertAlmostEqual(sum(b.probability for b in paths), 1.)
            self.assertEqual(len(paths), 9)

    def test_full_and_resumed_actual_learning_match_and_wrong_arm_rejected(self):
        spec = dict(id='resume-example', mechanism='world', game='combat', profile=1,
                    seed=601, regime='outside', turns=3)
        registration = dict(plan_sha256='test-plan', source_sha256='test-source')
        with tempfile.TemporaryDirectory() as folder:
            full = run(Path(folder)/'full', spec, 'stratified', registration=registration)
            self.assertIsNone(run(Path(folder)/'split', spec, 'stratified', max_decisions=1, registration=registration))
            self.assertFalse(audit(Path(folder)/'split', spec, 'stratified', registration)['complete'])
            self.assertFalse((Path(folder)/'split/result.json').exists())
            with self.assertRaises(ValueError):
                run(Path(folder)/'split', spec, 'stable', registration=registration)
            with self.assertRaises(ValueError):
                run(Path(folder)/'split', spec, 'stratified', registration=dict(registration,plan_sha256='changed'))
            resumed = run(Path(folder)/'split', spec, 'stratified', registration=registration)
            for key in ('trace', 'final', 'final_owner', 'final_rng', 'checks', 'configuration'):
                self.assertEqual(full[key], resumed[key], key)
            self.assertEqual(resumed['replayed_actual_steps'], 3)
            finished = run(Path(folder)/'split', spec, 'stratified', registration=registration)
            self.assertEqual(finished['trace'], resumed['trace'])
            self.assertTrue(audit(Path(folder)/'split', spec, 'stratified', registration)['complete'])
            with self.assertRaises(AssertionError):
                audit(Path(folder)/'split', spec, 'stable', registration)


if __name__ == '__main__':
    unittest.main()
