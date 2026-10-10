import copy
from dataclasses import replace
import unittest
import numpy as np
from reflex.goal_progress import relative_progress, choose_with_progress, omit_expired_proxies
from reflex.examples import context, action, effect
from reflex.board_models import ThanksPosition
from reflex.laboratory import PROFILES
from reflex.strong_search import PublicMemory, _thanks_simulate
from reflex.strong_table import decide
from reflex.strong_search import SearchBudget


class GoalProgressTests(unittest.TestCase):
    def test_progress_is_invariant_to_score_units_translation_and_participant_order(self):
        scores=np.array([[[20,30,40,25],[2,4,3,5]],[[25,30,40,25],[3,4,3,5]]])
        a=relative_progress(scores,0,direction=-1,scale=35)
        b=relative_progress(scores*7+200,0,direction=-1,scale=35*7)
        c=relative_progress(-scores[:,:,::-1],3,direction=1,scale=35)
        np.testing.assert_allclose(a,b);np.testing.assert_allclose(a,c)
        self.assertTrue(np.all((a>0)&(a<1)))

    def test_losing_plateau_excludes_avoidable_large_progress_regret_without_changing_traits(self):
        c=context('plateau',[action('HARM',effect(-1)),action('RECOVER',effect(-1))],{'esteem':.5},{'power':.8})
        original=copy.deepcopy(c);success=np.zeros((2,32));progress=np.array([[.01]*32,[.5]*32])
        adjusted,d,g=choose_with_progress(c,('HARM','RECOVER'),success,progress)
        self.assertEqual(d['action_id'],'RECOVER');self.assertEqual(g['allowed'],['RECOVER'])
        self.assertEqual(g['signal'],'goal_progress_on_constant_success');self.assertEqual(c,original)
        for k in ('personality','values','needs'):self.assertEqual(adjusted[k],c[k])

    def test_equal_mean_different_events_and_secured_success_do_not_activate_progress(self):
        c=context('tied',[action('A',effect(0)),action('B',effect(0))],{}, {})
        for success in (np.array([[0,1]*16,[1,0]*16]),np.ones((2,32))):
            adjusted,d,g=choose_with_progress(c,('A','B'),success,np.array([[0]*32,[1]*32]))
            self.assertEqual(g['signal'],'terminal_success');self.assertIs(adjusted,c)

    def test_bad_secondary_samples_are_rejected_before_selection(self):
        c=context('bad',[action('A',effect(0))],{}, {})
        for bad in (np.array([[np.nan]*16]),np.ones((1,17)),np.array([[2.]*16])):
            with self.assertRaises(ValueError):choose_with_progress(c,('A',),np.ones((1,16)),bad)

    def test_other_axis_outcome_distribution_is_not_replaced_by_its_first_branch(self):
        c=context('risk',[action('A',effect(-1,values={'security':-1},p=.25),
                                          effect(-1,values={'security':1},p=.75))],{}, {'security':.7})
        adjusted,_,_=choose_with_progress(c,('A',),np.zeros((1,16)),np.array([[.2,.8]*8]))
        rows=adjusted['actions'][0]['outcomes']
        self.assertAlmostEqual(sum(r['p']*r['values']['security'] for r in rows),.5)
        self.assertAlmostEqual(sum(r['p'] for r in rows),1.)
        self.assertEqual(len(rows),4)
        self.assertIn('progress_alignment',adjusted['facts'])

    def test_custom_owner_cannot_silently_get_a_canonical_personality(self):
        p=copy.deepcopy(PROFILES[0]);p['traits']=(.1,)*5
        s=replace(ThanksPosition.start(4,35),remaining=0)
        with self.assertRaises(ValueError):
            _thanks_simulate(s,0,PublicMemory('no_thanks',0),True,('TAKE',),(0,),16,np.random.default_rng(1),own_profile=p)

    def test_real_expiry_keeps_fixed_axes_costs_and_other_active_needs(self):
        c=context('expiry',[action('A',effect(.3,needs={'safety':.8,'esteem':.2},values={'security':.7},cost=.4))],
                  {'safety':.9,'esteem':.5},{'security':.8})
        c['state'].update(primary_need='safety',mode='need',mode_urgency=.9)
        saved=copy.deepcopy(c)
        expired=omit_expired_proxies(c,needs=('safety',),values=('security',),style=('neuroticism',))
        self.assertEqual(c,saved);self.assertEqual(expired['values'],c['values']);self.assertEqual(expired['personality'],c['personality'])
        self.assertIsNone(expired['state']['primary_need']);self.assertFalse(expired['needs']['safety']['enabled'])
        self.assertEqual(expired['actions'][0]['outcomes'][0]['cost'],.4)
        self.assertEqual(expired['needs']['esteem'],c['needs']['esteem'])

    def test_last_card_cannot_buy_nonexistent_future_flexibility_at_seventeen_point_cost(self):
        cards=((10,32),(3,4,5,6,7,8),(9,11,12,13,14,15,16,17,19,20),(21,22,23,24,25))
        seen=tuple(sorted([18]+[c for h in cards for c in h]))
        s=ThanksPosition(cards,(1,0,11,32),0,18,0,seen,0,(10,11,0,0));budget=SearchBudget(8,16,16,1)
        for p in PROFILES:
            m=PublicMemory('no_thanks',0)
            _,old,_=decide('no_thanks',s,0,p,8880,0,100,m,None,'adaptive',budget,variant='progress')
            c,new,st=decide('no_thanks',s,0,p,8880,0,100,m,None,'adaptive',budget,variant='horizon_progress')
            self.assertEqual(old['action_id'],'TAKE');self.assertEqual(new['action_id'],'PASS')
            self.assertEqual(st['actions']['TAKE']['mean_score'],59)
            self.assertEqual(st['actions']['PASS']['mean_score'],42)
            self.assertFalse(c['needs']['safety']['enabled'])
            _,certified,details=decide('no_thanks',s,0,p,8880,0,100,m,None,'adaptive',budget,variant='certified_expiry')
            self.assertEqual(certified['action_id'],'PASS')

    def test_last_card_alone_does_not_certify_expiry_of_current_negotiation(self):
        s=replace(ThanksPosition.start(4,30),remaining=0);p=PROFILES[1];m=PublicMemory('no_thanks',0)
        budget=SearchBudget(8,16,16,1)
        a=decide('no_thanks',s,0,p,8001,0,0,m,None,'adaptive',budget,variant='progress')
        b=decide('no_thanks',s,0,p,8001,0,0,m,None,'adaptive',budget,variant='certified_expiry')
        self.assertEqual(a[:2],b[:2]);self.assertNotIn('expired_proxies',b[0]['facts'])

    def test_owner_reflex_continuation_matches_forced_single_card_rules(self):
        s=replace(ThanksPosition.start(4,35),remaining=0,chips=(11,0,11,11))
        memory=PublicMemory('no_thanks',0);before=copy.deepcopy(memory.record())
        for profile in PROFILES:
            scores,shares=_thanks_simulate(s,0,memory,True,('TAKE','PASS'),(0,0),16,np.random.default_rng(8),own_profile=profile)
            np.testing.assert_array_equal(scores[0],np.tile([24,0,-11,-11],(16,1)))
            np.testing.assert_array_equal(scores[1],np.tile([-10,34,-11,-11],(16,1)))
            self.assertEqual(memory.record(),before)

    def test_both_components_are_deterministic_and_preserve_owner_state(self):
        s=replace(ThanksPosition.start(4,30),remaining=2);p=PROFILES[3];m=PublicMemory('no_thanks',0)
        state=dict(primary_need='esteem',mode='principle',intent_action='PASS',age=5,mode_urgency=.5)
        saved=copy.deepcopy(state);budget=SearchBudget(8,16,16,1)
        a=decide('no_thanks',s,0,p,2,0,0,m,state,'adaptive',budget,variant='combined')
        b=decide('no_thanks',s,0,p,2,0,0,m,state,'adaptive',budget,variant='combined')
        self.assertEqual(a,b);self.assertEqual(state,saved)
        self.assertIn('NOT future root-search',a[2]['continuation'])


if __name__=='__main__':unittest.main()
