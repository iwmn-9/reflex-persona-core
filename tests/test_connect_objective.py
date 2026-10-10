import unittest
import numpy as np
from reflex.board_models import ConnectPosition
from reflex.connect_objective import choose


class ObjectiveOpponentTests(unittest.TestCase):
    def test_takes_an_immediate_win(self):
        s=ConnectPosition()
        for a in (0,6,1,6,2,5):s=s.play(f'DROP:{a}')
        self.assertEqual(choose(s,np.random.default_rng(3)),'DROP:3')

    def test_blocks_an_immediate_public_loss(self):
        s=ConnectPosition()
        for a in (0,6,2,6,4,6):s=s.play(f'DROP:{a}')
        self.assertEqual(choose(s,np.random.default_rng(3)),'DROP:6')


if __name__=='__main__':unittest.main()
