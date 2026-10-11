import copy
from dataclasses import replace
import unittest
from reflex.core import Policy,digest
from reflex.owner_continuation import OwnerBranch
from reflex.conditional_forecast import forecast
from reflex import team_projects as tp,team_combat as tc
from reflex.laboratory import PROFILES
from tools.run_conditional_continuation import run_game,solo_opening


def fixture():
    w=tp.Workshop.start();actors=(0,1)
    cs=[tp.make_context(w,i,PROFILES[i],9800) for i in actors]
    ds=[Policy(principle_priority='finite').choose(c,False) for c in cs]
    return w,actors,cs,ds


class ConditionalContinuationTests(unittest.TestCase):
    def test_counterfactual_root_age_and_detached_siblings(self):
        w,a,cs,ds=fixture();cs[0]['state'].update(intent_action='rest',age=4)
        ds[0]=Policy(principle_priority='finite').choose(cs[0],False);saved=copy.deepcopy((cs,ds))
        left=OwnerBranch(a,cs,ds,('rest','gather'));right=OwnerBranch(a,cs,ds,('gather','rest'))
        self.assertEqual(left.states[0]['age'],5);self.assertEqual(right.states[0]['age'],0)
        after,_=tp.resolve(w,{i:'rest' for i in range(6)})
        left.choose(a,after,tp.make_context)
        self.assertEqual(right.states[0]['intent_action'],'gather');self.assertEqual((cs,ds),saved)

    def test_refreshes_public_energy_but_retains_axes_and_owner_state(self):
        w,a,cs,ds=fixture();branch=OwnerBranch(a,cs,ds,('gather','rest'))
        after,_=tp.resolve(w,{i:'gather' for i in range(6)})
        c=branch.context(0,after,tp.make_context)
        self.assertNotEqual(c['needs'],cs[0]['needs']);self.assertEqual(c['values'],cs[0]['values'])
        self.assertEqual(c['personality'],cs[0]['personality']);self.assertEqual(c['state']['intent_action'],'gather')

    def test_rejects_wrong_owner_time_profile_and_private_state_factory(self):
        w,a,cs,ds=fixture();branch=OwnerBranch(a,cs,ds,('rest','rest'));after=replace(w,tick=1)
        for key in ('scope','personality','state','tick'):
            def wrong(world,actor,profile,seed,state):
                c=tp.make_context(world,actor,profile,seed,state)
                if key=='tick':c['tick']=0
                elif key=='scope':c[key]['npc']='foreign'
                elif key=='personality':c[key]['openness']=0.
                else:c[key]['age']=999
                return c
            with self.assertRaises(ValueError):branch.context(0,after,wrong)

    def test_non_owner_illegal_root_and_bad_receipt_rejected(self):
        w,a,cs,ds=fixture()
        with self.assertRaises(ValueError):OwnerBranch(a,cs,ds,('illegal','rest'))
        bad=copy.deepcopy(ds);bad[0]['context_hash']='foreign'
        with self.assertRaises(ValueError):OwnerBranch(a,cs,bad,('rest','rest'))
        branch=OwnerBranch(a,cs,ds,('rest','rest'))
        with self.assertRaises(ValueError):branch.choose((2,),replace(w,tick=1),tp.make_context)

    def test_rejects_missing_or_forged_exchange_decision(self):
        w,a,cs,ds=fixture();after=replace(w,tick=1)
        with self.assertRaises(ValueError):OwnerBranch(a,cs,ds,('rest','rest')).choose(a,after,tp.make_context,lambda *x:([],{}))
        def forged(world,actors,contexts,decisions):
            decisions[0]['next_state']['intent_action']='forged'
            return decisions,{}
        with self.assertRaises(ValueError):OwnerBranch(a,cs,ds,('rest','rest')).choose(a,after,tp.make_context,forged)

    def test_one_turn_equivalence_and_identical_root_bank_projects(self):
        w,a,cs,ds=fixture();saved=copy.deepcopy(cs)
        old=tp.forecast(w,a,cs,ds,horizon=1,partner_model='uncertain')
        new=forecast('projects',w,a,cs,ds,horizon=1)
        self.assertEqual(new.roots,old.roots);self.assertEqual(new.purpose,old.purpose)
        for c,other in zip(new.contexts,old.contexts):self.assertEqual(c['actions'],other['actions'])
        self.assertEqual(cs,saved);self.assertEqual(w.tick,0)

    def test_one_turn_equivalence_combat_with_unsigned_alive(self):
        w=tc.opening('rescue',0);a=(0,1);cs=[tc.make_person(w,i,PROFILES[i],9800) for i in a]
        ds=[Policy(principle_priority='finite').choose(c,False) for c in cs]
        old=tc.forecast(w,a,cs,ds,horizon=1,partner_model='uncertain')
        new=forecast('combat',w,a,cs,ds,horizon=1)
        self.assertEqual(new.roots,old.roots);self.assertEqual(new.purpose,old.purpose)
        for c,other in zip(new.contexts,old.contexts):self.assertEqual(c['actions'],other['actions'])

    def test_same_root_futures_reconsider_without_real_learning(self):
        w,a,cs,ds=fixture();saved=digest(cs)
        personal=forecast('projects',w,a,cs,ds,horizon=3)
        shared=forecast('projects',w,a,cs,ds,horizon=3,continuation='exchange')
        self.assertGreater(personal.audit['future_owner_steps'],0)
        self.assertGreater(shared.audit['inner_nodes'],0);self.assertEqual(personal.roots,shared.roots)
        self.assertEqual(digest(cs),saved)

    def test_solo_world_uses_same_rules_and_no_other_signed_person(self):
        for genre,scenario in (('projects','balanced'),('combat','open')):
            row=run_game((genre,scenario,0,9800,'solo','bargain_personal'))
            self.assertEqual(row['members'],1)
            self.assertTrue(all(len(t['signed'])<=1 for t in row['trace']))
            self.assertEqual(len(row['initial']['people' if genre=='projects' else 'units']),2)

    def test_bounded_configuration_rejected(self):
        w,a,cs,ds=fixture()
        for kwargs in ({'horizon':0},{'inner_plans':0},{'continuation':'oracle'}):
            with self.assertRaises(ValueError):forecast('projects',w,a,cs,ds,**kwargs)


if __name__=='__main__':unittest.main()
