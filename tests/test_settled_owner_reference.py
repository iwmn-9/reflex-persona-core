import copy
import unittest
from reflex.core import digest
from reflex.laboratory import PROFILES
from tests.test_continuation_persona import IntrinsicTradeoff
from tools.continuing_games import ContinuingGame
from tools.preparation_world import PreparationChain
from tools.settled_owner_reference import evaluate, one_public_event


class SettledOwnerReferenceTests(unittest.TestCase):
    def test_preserves_intrinsic_sacrifice_instead_of_calling_it_regret(self):
        for profile,expected in ((0,'x'),(2,'y')):
            game=IntrinsicTradeoff(PROFILES[profile],turns=2)
            world=game.start();actor=game.actor(game.context(world),learning='segmented')
            before=digest(actor.record())
            result=evaluate(game,actor,world,'expected',maximum_steps=2)
            self.assertEqual(result['selected_action'],expected)
            self.assertEqual(result['reference_action'],expected)
            self.assertEqual(result['normalized_regret'],0.)
            self.assertEqual(digest(actor.record()),before)

    def test_no_hidden_or_unsettled_reference(self):
        stochastic=ContinuingGame('auction',PROFILES[0],turns=2)
        with self.assertRaises(ValueError):one_public_event(stochastic,stochastic.start(),'bid:0')
        game=PreparationChain(2,payout=6.,turns=3)
        actor=game.actor(game.context(game.start()),learning='segmented')
        with self.assertRaises(ValueError):evaluate(game,actor,game.start(),'stable',maximum_steps=1)

    def test_shared_bank_does_not_change_the_stable_controller(self):
        game=PreparationChain(2,payout=6.,turns=3)
        actor=game.actor(game.context(game.start()),learning='segmented')
        with self.assertRaises(ValueError):
            evaluate(game,actor,game.start(),'stable',maximum_steps=3,continuation_scope='shared')


if __name__=='__main__':unittest.main()
