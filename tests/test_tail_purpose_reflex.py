import copy
import unittest

from reflex.core import digest
from reflex.decision_loop import DecisionLoop
from reflex.laboratory import PROFILES
from reflex._continuation_portfolio import portfolio_planner
from tools.continuing_games import ContinuingGame
from tools.preparation_world import PreparationChain
from tools.run_owned_value import cumulative_basis


def evaluate(game, world, enabled, horizon=3):
    owner = game.actor(game.context(world), learning='segmented')
    before = digest(owner.record()); captured = {}
    request = game.request(game.context(world, owner.state))
    planner = portfolio_planner(owner, request, world, observe=game.context,
        events=game.events, terminal=game.terminal, assess=lambda w: game.goal(w).record(),
        request_for=game.request, feedback=game.feedback, model_id=game.model_id,
        target=game.target, horizon=horizon, seeds=(701,), world_model='posterior',
        cumulative_preference=cumulative_basis(game, True), include_retry=True,
        captured=captured, purpose_estimator='conditional', tail_recovery=2,
        coherent_cumulative_mean=True, tail_purpose_reflex=enabled)
    result = DecisionLoop.decide_batch([(copy.deepcopy(owner), request)], False, planner)[0]
    assert digest(owner.record()) == before
    assert world == game.world(request.context)
    return result, captured


class RenamedGame(ContinuingGame):
    names = {'bet:0':'zeta', 'bet:1':'alpha', 'bet:2':'kappa', 'pass':'theta', 'bond':'delta'}

    def keys(self, world):
        return tuple(self.names[k] for k in super().keys(world))

    def transition(self, world, key, symbol):
        base = ContinuingGame(self.game, self.profile, self.turns, self.variant)
        original = next(k for k, v in self.names.items() if v == key)
        after, flow = base.transition(world, original, symbol)
        after['last'] = key
        return after, flow


class TailPurposeReflexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = ContinuingGame('gambling', PROFILES[1], 32)
        cls.world = cls.game.start(); cls.world.update(tick=3, cash=20., investment=2)
        cls.old = evaluate(cls.game, cls.world, False)
        cls.new = evaluate(cls.game, cls.world, True)

    def test_passive_income_does_not_hide_future_action_benefit(self):
        old, new = self.old[1], self.new[1]
        for proposal in old['modeled']:
            if proposal.endswith('-reflex'):
                self.assertTrue(all(r['actions'][1] == 'pass' for r in old['modeled'][proposal]))
                self.assertTrue(all(r['actions'][1] != 'pass' for r in new['modeled'][proposal]))
        for key in ('personality', 'values', 'needs'):
            self.assertEqual(self.old[0]['context'][key], self.new[0]['context'][key])
        self.assertNotIn('tail_purpose_reflex', old['forecast'].audit['continuation_portfolio'])

    def test_forced_retry_trajectories_keep_their_existing_meaning(self):
        old, new = self.old[1], self.new[1]
        for proposal in old['modeled']:
            if proposal.endswith('-retry'):
                self.assertEqual(old['modeled'][proposal], new['modeled'][proposal])
                self.assertEqual(old['paths'][proposal], new['paths'][proposal])

    def test_renamed_actions_keep_the_future_goal_resolution(self):
        g = RenamedGame('gambling', PROFILES[1], 32)
        _, renamed = evaluate(g, copy.deepcopy(self.world), True)
        for proposal, rows in renamed['modeled'].items():
            if proposal.endswith('-reflex'):
                self.assertTrue(all(r['actions'][1] != g.names['pass'] for r in rows))

    def test_long_preparation_remains_available_and_bad_preparation_rejected(self):
        profitable = PreparationChain(5, payout=14., turns=32, profile=PROFILES[0])
        bad = PreparationChain(5, payout=1., turns=32, profile=PROFILES[0])
        # At H8 the existing retry proposal starts an unfinished second cycle;
        # both old and new controllers choose x. Do not mislabel the inherited
        # candidate-coverage limit as a new regression or silently drop it.
        for enabled in (False, True):
            self.assertEqual(evaluate(profitable, profitable.start(), enabled, 8)[0]['decision']['action_id'], 'x')
            self.assertEqual(evaluate(profitable, profitable.start(), enabled, 10)[0]['decision']['action_id'], 'y')
            self.assertEqual(evaluate(bad, bad.start(), enabled, 10)[0]['decision']['action_id'], 'x')


if __name__ == '__main__':
    unittest.main()
