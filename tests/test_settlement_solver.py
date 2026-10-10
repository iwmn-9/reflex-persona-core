import copy
from dataclasses import replace
import itertools
import unittest
import numpy as np
from reflex.board_models import ThanksPosition, card_points
from reflex.laboratory import PROFILES
from reflex.finite_continuation import resolve_chain
from reflex.supported_memory import SupportedPublicMemory
from reflex.settlement_solver import solve
from reflex.goal_guard import choose_with_weighted_goal
from reflex.examples import context, action, effect


class SettlementTests(unittest.TestCase):
    def test_planned_future_owner_uses_fixed_prior_not_virtual_learning(self):
        from tools.probe_settlement_consistency import evaluate
        from reflex.strong_table import decide
        held=((3,10,11,32,33),(7,8,16,17,34,35),(13,14,19,20,21,23),(5,24,25,28,29,31))
        s=ThanksPosition(held,(4,11,15,8),1,26,6,tuple(sorted([26]+[x for h in held for x in h])),0,(29,28,28,28))
        for p in PROFILES:
            m=SupportedPublicMemory('no_thanks',1);before=copy.deepcopy(m.record())
            c,d,stats=decide('no_thanks',s,1,p,7600,0,0,m,None,'planned',variant='settlement')
            for root in s.legal():
                actual=evaluate(s,1,p,7600,m,None,root,'settlement',d,mode='planned')
                for k,field in (('credit','win_share'),('score','mean_score'),('progress','goal_progress')):
                    self.assertAlmostEqual(actual[k],stats['actions'][root][field],places=10)
            self.assertEqual(m.record(),before)

    def test_generic_chain_resolves_owner_policy_instead_of_an_optimistic_plan(self):
        nodes=[dict(actor=0,settle='now',terminal=(1.,0.),forecast={'now':.5,'later':.5},**{'continue':'later'}),
               dict(actor=1,settle='now',terminal=(0.,1.),forecast={'now':.25,'later':.75},**{'continue':'later'}),
               dict(actor=0,settle='now',terminal=(2.,0.),forecast={'now':1.},**{'continue':'later'})]
        roots,choices=resolve_chain(nodes,0,lambda i,outcomes:'later' if i==0 else 'now')
        self.assertEqual(roots[0]['later'],((.25,(0.,1.)),(.75,(2.,0.))))
        self.assertEqual(choices,{2:'now',0:'later'})
        with self.assertRaises(ValueError):resolve_chain(nodes,0,lambda i,o:'illegal')

    def test_complete_mass_has_no_sampling_uncertainty_or_illegal_promotion(self):
        c=context('weighted',[action('bad',effect(.9)),action('good',effect(.2)),action('illegal',effect(1),legal=False)])
        d,g=choose_with_weighted_goal(c,('bad','good','illegal'),np.array([[0.],[1.],[1.]]),np.ones((3,1)))
        self.assertEqual(d['action_id'],'good');self.assertEqual(g['allowed'],['good'])
        self.assertTrue(all(v['sampling_error']==0 for v in g['bounds'].values()))
        with self.assertRaises(ValueError):choose_with_weighted_goal(c,('bad','good','illegal'),np.ones((3,1)),np.zeros((3,1)))

    def test_final_card_rule_values_equal_independent_arithmetic(self):
        for chips in ((1,0,21,22),(3,2,19,20),(11,11,11,11)):
            s=replace(ThanksPosition.start(4,18),remaining=0,chips=chips,cards=((12,14),(17,),(22,),()))
            for p in PROFILES:
                m=SupportedPublicMemory('no_thanks',0);before=copy.deepcopy(m.record())
                c,d,stats=solve(s,0,p,500,0,0,m,None)
                self.assertEqual(m.record(),before);self.assertEqual(c['personality'],dict(zip(('openness','conscientiousness','extraversion','agreeableness','neuroticism'),p['traits'])))
                expected=card_points(s.cards[0]+(18,))-chips[0]-s.pot
                self.assertEqual(stats['actions']['TAKE']['mean_score'],expected)
                self.assertLessEqual(stats['finite_chain_nodes'],45)
                if chips[1]==0:
                    self.assertEqual(stats['actions']['PASS']['mean_score'],card_points(s.cards[0])-chips[0]+1)

    def test_future_owner_choice_matches_fresh_actual_controller_after_public_reveals(self):
        for p in PROFILES:
            s=replace(ThanksPosition.start(4,31),remaining=0,chips=(4,4,4,4))
            m=SupportedPublicMemory('no_thanks',0);state=None;clock=0
            # Follow actual PASS decisions; all other public PASS observations
            # are permitted hypotheses, not hidden opponent state.
            while len(s.legal())>1 and clock<16:
                if s.turn==0:
                    c,d,stats=solve(s,0,p,777,0,clock,m,state)
                    if d['action_id']=='TAKE':break
                    self.assertGreaterEqual(stats['future_owner_opportunities'],0);state=d['next_state']
                else:m.observe(s,s.turn,'PASS',f'encounter-0-tick-{clock}-actor-{s.turn}')
                s=s.play('PASS');clock+=1

    def test_solver_refuses_hidden_future_and_wrong_active_owner(self):
        m=SupportedPublicMemory('no_thanks',0);s=ThanksPosition.start(4,20)
        with self.assertRaises(ValueError):solve(s,0,PROFILES[0],0,0,0,m,None)
        with self.assertRaises(ValueError):solve(replace(s,remaining=0,turn=1),0,PROFILES[0],0,0,0,m,None)


if __name__=='__main__':unittest.main()
