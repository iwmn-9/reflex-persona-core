import copy
import unittest
import numpy as np
from reflex.board_models import ConnectPosition
from reflex.laboratory import PROFILES
from reflex.monte_carlo import RolloutBudget
from reflex.connect_policy_rollout import incumbent, decide, OwnerPolicyModel, Branch


class ConnectPolicyRolloutTests(unittest.TestCase):
    def test_same_future_base_controller_keeps_owner_identity_and_state(self):
        s=ConnectPosition();p=PROFILES[0];c,d,_=incumbent(s,p,91,0,'public-case',None)
        m=OwnerPolicyModel(s,p,91,0,'public-case',None,d);m.begin_trial()
        b=Branch(s.play('DROP:2'),'DROP:2');r=np.random.default_rng(18)
        a=m.choose(b,r,'persona');b=m.step(b,a)
        expected=incumbent(b.position,p,91,2,'public-case',copy.deepcopy(m.memory))[1]
        self.assertEqual(m.choose(b,r,'persona'),expected['action_id'])
        self.assertEqual(m.memory,expected['next_state'])

    def test_exhausted_policy_rollout_preserves_actual_record(self):
        s=ConnectPosition();p=PROFILES[1];c,d,_=incumbent(s,p,91,0,'public-case',None)
        rc,rd,st=decide(s,p,91,0,'public-case',rollout=RolloutBudget(max_nodes=0))
        self.assertEqual(c,rc);self.assertEqual(d,rd);self.assertFalse(st['policy_rollout']['used'])


if __name__=='__main__':unittest.main()
