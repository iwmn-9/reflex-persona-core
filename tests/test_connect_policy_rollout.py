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

    def test_exact_public_state_cache_retains_private_owner_state(self):
        s=ConnectPosition();p=PROFILES[0];_,d,_=incumbent(s,p,91,0,'public-case',None)
        m=OwnerPolicyModel(s,p,91,0,'public-case',None,d);records=[]
        for _ in range(2):
            m.begin_trial();b=Branch(s.play('DROP:2'),'DROP:2');r=np.random.default_rng(18)
            b=m.step(b,m.choose(b,r,'persona'))
            records.append((m.choose(b,r,'persona'),copy.deepcopy(m.memory)))
        self.assertEqual(records[0],records[1]);self.assertEqual(m.requests,2);self.assertEqual(m.calls,1)

    def test_rollout_record_matches_the_complete_effective_context(self):
        from reflex.tabletop_trials import score
        s=ConnectPosition()
        for a in ('DROP:0','DROP:6','DROP:1','DROP:6','DROP:2','DROP:5'):s=s.play(a)
        c,d,st=decide(s,PROFILES[0],91,6,'public-case',rollout=RolloutBudget(samples=2,min_samples=2,max_nodes=10000,max_steps=42))
        self.assertTrue(st['policy_rollout']['used']);self.assertEqual(score(c)[0],d)

    def test_latent_rival_strength_is_shared_by_all_roots_of_one_trial(self):
        s=ConnectPosition();p=PROFILES[0];_,d,_=incumbent(s,p,91,0,'public-case',None)
        m=OwnerPolicyModel(s,p,91,0,'public-case',None,d,rival_policy='mixture')
        kinds=[]
        for a in s.legal():
            m.begin_trial();b=Branch(s.play(a),a);m.choose(b,np.random.default_rng(18),'persona');kinds.append(m.trial_kinds[0])
        self.assertEqual(len(set(kinds)),1);self.assertEqual(m.trial,0)


if __name__=='__main__':unittest.main()
