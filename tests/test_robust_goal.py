import copy
from dataclasses import replace
import unittest
import numpy as np
from reflex.examples import context,action,effect
from reflex.robust_goal import select
from reflex.goal_progress import choose_with_progress
from reflex.finite_continuation import resolve_chain,resolve_chains
from reflex.laboratory import PROFILES
from reflex.board_models import ThanksPosition
from reflex.strong_search import PublicMemory,SearchBudget
from reflex.supported_memory import GuardedPublicMemory
from reflex.strong_table import decide
from tools.probe_settlement_consistency import evaluate


class RobustGoalTests(unittest.TestCase):
    def test_disagreeing_models_leave_compromise_without_rewriting_personality(self):
        c=context('models',[action(k,effect(1 if k=='A' else 0)) for k in ('A','B','C')]);before=copy.deepcopy(c)
        a=np.array([[.85],[.4],[.75]]);b=np.array([[.2],[.95],[.75]])
        adjusted,d,g=select(c,('A','B','C'),{'shared':(a,np.ones_like(a)),'challenger':(b,np.ones_like(b))},weights={'shared':np.ones_like(a),'challenger':np.ones_like(b)})
        self.assertEqual(d['action_id'],'C');self.assertEqual(g['allowed'],['C'])
        self.assertAlmostEqual(g['unavoidable_model_regret'],.2)
        for k in ('personality','values','needs'):self.assertEqual(adjusted[k],before[k])
        self.assertEqual(c,before)

    def test_single_sampled_model_matches_existing_purpose_corridor_and_progress(self):
        c=context('single',[action('A',effect(.3)),action('B',effect(.8))])
        for success in (np.array([[0,1]*16,[1,0]*16]),np.zeros((2,32)),np.ones((2,32))):
            progress=np.array([[.6]*32,[.2]*32])
            _,a,g=choose_with_progress(c,('A','B'),success,progress)
            _,b,h=select(c,('A','B'),{'shared':(success,progress)})
            self.assertEqual(a,b);self.assertEqual(g['allowed'],h['allowed']);self.assertEqual(g['signal'],h['signal'])

    def test_all_models_must_have_zero_success_and_bad_unused_progress_still_fails(self):
        c=context('not-flat',[action('A',effect()),action('B',effect())])
        a=np.zeros((2,2));b=np.array([[0,1],[0,0.]])
        _,d,g=select(c,('A','B'),{'shared':(a,a),'other':(b,np.ones_like(b))})
        self.assertEqual(g['signal'],'terminal_success')
        with self.assertRaises(ValueError):select(c,('A','B'),{'shared':(b,np.full_like(b,np.nan))})
        with self.assertRaises(ValueError):select(c,('A','B'),{'shared':(b,b)},weights={'shared':np.zeros_like(b)})

    def test_aligned_chain_uses_one_owner_action_in_every_model(self):
        nodes=[dict(actor=0,settle='now',terminal=(1.,0.),forecast={'now':.5,'later':.5},**{'continue':'later'}),
            dict(actor=1,settle='now',terminal=(0.,1.),forecast={'now':.25,'later':.75},**{'continue':'later'}),
            dict(actor=0,settle='now',terminal=(2.,0.),forecast={'now':1.},**{'continue':'later'})]
        other=copy.deepcopy(nodes);other[1]['forecast']={'now':.8,'later':.2}
        calls=[]
        def choose(i,models):calls.append((i,tuple(models)));return 'later' if i==0 else 'now'
        roots,choices=resolve_chains({'a':nodes,'b':other},0,choose)
        self.assertEqual(calls,[(2,('a','b')),(0,('a','b'))]);self.assertEqual(choices,{2:'now',0:'later'})
        self.assertEqual(roots['a'][0]['later'],((.25,(0.,1.)),(.75,(2.,0.))))
        self.assertEqual(roots['b'][0]['later'],((.8,(0.,1.)),(.2,(2.,0.))))
        bad=copy.deepcopy(other);bad[1]['terminal']=(3.,0.)
        with self.assertRaises(ValueError):resolve_chains({'a':nodes,'b':bad},0,choose)
        last=[dict(actor=0,settle='now',terminal=(1.,0.),forecast={'now':1.})]
        roots,choices=resolve_chains({'a':last,'b':last},0,lambda i,o:'now')
        self.assertEqual(choices,{0:'now'});self.assertEqual(roots['a'],roots['b'])

    def test_two_model_forecast_matches_actual_future_robust_owner_in_both_modes(self):
        held=((3,10,11,32,33),(7,8,16,17,34,35),(13,14,19,20,21,23),(5,24,25,28,29,31))
        s=ThanksPosition(held,(4,11,15,8),1,26,6,tuple(sorted([26]+[x for h in held for x in h])),0,(29,28,28,28))
        future=replace(s,cards=(held[0][:-1],)+held[1:],seen=tuple(x for x in s.seen if x!=33),remaining=1)
        m=GuardedPublicMemory('no_thanks',1)
        for actor in (0,2,3):
            for i in range(16):m.observe(replace(s,turn=actor),actor,'TAKE',f'local-{actor}-{i}')
            for i in range(32):m.observe(replace(future,turn=actor),actor,'PASS',f'shared-{actor}-{i}')
            key=m._key('current_card_only',actor);m.transfer.entries.pop(key,None)
            for _ in range(4):m.transfer.categorical(key,{'TAKE':.5,'PASS':.5},{'TAKE':.95,'PASS':.05},'TAKE')
        for mode in ('adaptive','planned'):
            for p in PROFILES:
                before=copy.deepcopy(m.record());_,d,stats=decide('no_thanks',s,1,p,7600,0,0,m,None,mode,variant='robust')
                self.assertEqual(len(stats['model_actions']),2 if mode=='adaptive' else 1)
                for key,actions in stats['model_actions'].items():
                    for root in s.legal():
                        actual=evaluate(s,1,p,7600,m,None,root,'robust',d,mode=mode,model_source='shared' if key=='shared' else 'selected')
                        for k,field in (('credit','win_share'),('score','mean_score'),('progress','goal_progress')):
                            self.assertAlmostEqual(actual[k],actions[root][field],places=10)
                self.assertEqual(m.record(),before)

    def test_unearned_specialization_does_not_change_nonterminal_search_or_action(self):
        s=ThanksPosition.start(4,26);budget=SearchBudget(8,16,16,1)
        for p in PROFILES:
            c,a,g=decide('no_thanks',s,0,p,7721,0,0,PublicMemory('no_thanks',0),None,'adaptive',budget,variant='certified_expiry')
            d,b,h=decide('no_thanks',s,0,p,7721,0,0,GuardedPublicMemory('no_thanks',0),None,'adaptive',budget,variant='robust')
            self.assertEqual(a,b);self.assertEqual(g['actions'],h['actions']);self.assertEqual(tuple(h['model_actions']),('shared',))


if __name__=='__main__':unittest.main()
