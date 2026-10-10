import copy
import unittest
import numpy as np
from reflex.board_models import ThanksPosition
from reflex.laboratory import PROFILES
from reflex.monte_carlo import RolloutBudget, TerminalEvaluation
from reflex.policy_rollout import evaluate_policy
from reflex.strong_search import PublicMemory, PERSONA
from reflex.strong_table import decide as base_decide
from reflex.thanks_policy_rollout import OwnerPolicyModel, Branch, decide


class SmallWorld:
    def begin_trial(self):pass
    def terminal(self,s):return s<=0
    def chance(self,s):return False
    def legal(self,s):return ('step',)
    def choose(self,s,rng,policy):return 'step'
    def step(self,s,a):return s-1
    def evaluate(self,s):
        from reflex.examples import effect
        return TerminalEvaluation(effect(1.),1.,True,False,0.)


class PolicyRolloutTests(unittest.TestCase):
    def test_incomplete_round_removes_its_first_root_terminal(self):
        budget=RolloutBudget(samples=2,min_samples=1,max_nodes=3,max_steps=10)
        _,samples,stats=evaluate_policy({'a':1,'b':2},SmallWorld(),budget,'test')
        self.assertEqual(stats['completed_samples'],1)
        self.assertEqual([len(v) for v in samples.values()],[1,1])
        self.assertTrue(stats['budget_exhausted'])

    def test_no_partial_round_is_usable(self):
        budget=RolloutBudget(samples=2,min_samples=1,max_nodes=1,max_steps=10)
        _,samples,stats=evaluate_policy({'a':1,'b':2},SmallWorld(),budget,'test')
        self.assertFalse(stats['used']);self.assertTrue(all(not v for v in samples.values()))

    def setup_model(self):
        s=ThanksPosition(((),)*4,(11,)*4,0,20,seen=(20,),remaining=2,payments=(0,)*4)
        m=PublicMemory('no_thanks',0);p=PROFILES[0]
        c,d,_=base_decide('no_thanks',s,0,p,123,0,0,m,None,'adaptive',PERSONA,variant='certified_expiry')
        return s,m,p,c,d

    def test_common_latent_deck_even_when_first_channel_differs(self):
        s,m,p,c,d=self.setup_model();model=OwnerPolicyModel(s,0,p,m,None,d,123,0,0,'adaptive')
        model.begin_trial();model._initialize(np.random.default_rng(1));deck=tuple(model.deck);kinds=model.kinds[:];nonce=model.nonce
        model.begin_trial();model._initialize(np.random.default_rng(999))
        self.assertEqual(deck,tuple(model.deck));self.assertEqual(kinds,model.kinds);self.assertEqual(nonce,model.nonce)
        self.assertTrue(set(deck).isdisjoint(s.seen))

    def test_future_owner_gets_public_position_and_detached_own_state(self):
        s,m,p,c,d=self.setup_model();before=copy.deepcopy(m.record());calls=[]
        def owner(game,pos,viewer,profile,seed,enc,tick,memory,state,mode,budget,**kw):
            calls.append((pos,copy.deepcopy(state),budget,kw))
            self.assertIsNot(memory,m);self.assertEqual(profile,p)
            self.assertFalse(hasattr(pos,'deck'));self.assertEqual(viewer,0)
            nd=copy.deepcopy(d);nd['action_id']='TAKE';nd['next_state']['intent_action']='TAKE'
            return c,nd,{}
        model=OwnerPolicyModel(s,0,p,m,None,d,123,0,0,'adaptive',base=owner)
        roots={a:Branch(s.play(a),a) for a in s.legal()}
        _,samples,stats=evaluate_policy(roots,model,RolloutBudget(samples=4,min_samples=4,max_nodes=10000),'public')
        self.assertTrue(stats['used']);self.assertGreater(len(calls),0)
        self.assertTrue(all(call[2]==PERSONA and call[3]['variant']=='certified_expiry' for call in calls))
        self.assertEqual(m.record(),before);self.assertTrue(all(len(v)==4 for v in samples.values()))

    def test_budget_failure_keeps_incumbent_record_and_context(self):
        s,m,p,c,d=self.setup_model()
        rc,rd,stats=decide('no_thanks',s,0,p,123,0,0,m,None,'adaptive',PERSONA,
            rollout=RolloutBudget(samples=2,min_samples=2,max_nodes=0))
        self.assertEqual(rc,c);self.assertEqual(rd,d);self.assertFalse(stats['policy_rollout']['used'])

    def test_card_reveal_keeps_the_actual_controller_decision_clock(self):
        s,m,p,c,d=self.setup_model();ticks=[]
        def owner(game,pos,viewer,profile,seed,enc,tick,memory,state,mode,budget,**kw):
            ticks.append(tick);return c,d,{}
        model=OwnerPolicyModel(s,0,p,m,None,d,123,0,17,'adaptive',base=owner)
        model.begin_trial();branch=model.sample(Branch(s.play('TAKE'),'TAKE'),np.random.default_rng(1))
        self.assertEqual(branch.steps,0)
        model.choose(branch,np.random.default_rng(2),'persona')
        self.assertEqual(ticks,[18])

    def test_same_public_future_uses_actual_owner_identity_and_selector(self):
        s,m,p,c,d=self.setup_model()
        model=OwnerPolicyModel(s,0,p,m,None,d,123,2,17,'adaptive')
        model.begin_trial();branch=model.sample(Branch(s.play('TAKE'),'TAKE'),np.random.default_rng(1))
        expected=base_decide('no_thanks',branch.position,0,p,123,2,18,model.memory,
            copy.deepcopy(model.owner_state),'adaptive',PERSONA,variant='certified_expiry')[1]
        actual=model.choose(branch,np.random.default_rng(2),'persona')
        self.assertEqual(actual,expected['action_id']);self.assertEqual(model.owner_state,expected['next_state'])

    def test_owner_seed_ablation_is_explicit(self):
        s,m,p,c,d=self.setup_model();nonces=[]
        def owner(*args,**kw):nonces.append(args[4]);return c,d,{}
        for mode in ('same_owner','resampled'):
            model=OwnerPolicyModel(s,0,p,m,None,d,123,0,0,'adaptive',base=owner,future_seed=mode)
            model.begin_trial();branch=model.sample(Branch(s.play('TAKE'),'TAKE'),np.random.default_rng(1))
            model.choose(branch,np.random.default_rng(2),'persona')
        self.assertEqual(nonces[0],123);self.assertNotEqual(nonces[1],123)


if __name__=='__main__':unittest.main()
