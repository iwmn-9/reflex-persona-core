import copy
from dataclasses import replace
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import digest,Policy
from reflex.examples import context,action,effect
from reflex.deliberation import JointForecast
from reflex.team_choice import select,concession
from reflex import team_projects as tp,team_combat as tc
from reflex.laboratory import PROFILES


def fixture(pain=False):
    cs=[context('team-test',[action('wait',effect()),action('help',effect())],
        needs={'physiology':.9},traits={'agreeableness':.4},mode='need') for _ in range(2)]
    for i,c in enumerate(cs):c['scope']['npc']=f'actor-{i}'
    fs=copy.deepcopy(cs)
    for i,c in enumerate(fs):
        c['actions']=[action('idle',effect(.1)),action('shared',effect(.8,
            needs={'physiology':-1.} if pain and i==0 else {}))]
    f=JointForecast(tuple(fs),{'idle':('wait','wait'),'shared':('help','help')},
        {'idle':0.,'shared':1.},4,.15)
    return cs,f


class TeamCooperationTests(unittest.TestCase):
    def test_accepted_plan_keeps_owners_traits_and_root_memory(self):
        cs,f=fixture();before=copy.deepcopy(cs)
        ds,a=select(cs,f,group='red');self.assertTrue(a['adopted'])
        self.assertEqual(cs,before)
        for c,d in zip(cs,ds):
            self.assertEqual(d['action_id'],'help');self.assertEqual(d['context_hash'],digest(c))
            self.assertEqual(d['next_state']['intent_action'],'help')
            self.assertLessEqual(a['member_regrets'][c['scope']['npc']],a['concessions'][c['scope']['npc']]+1e-12)

    def test_consent_can_refuse_sacrifice_even_when_team_mean_accepts(self):
        cs,f=fixture(True);ds,a=select(cs,f,group='red',mode='sum')
        self.assertTrue(a['adopted']);self.assertEqual(ds[0]['action_id'],'help')
        ds,a=select(cs,f,group='red',mode='consent')
        self.assertIsNone(ds);self.assertFalse(a['adopted'])

    def test_wrong_tick_duplicate_owner_and_rewritten_profile_rejected(self):
        cs,f=fixture();wrong=copy.deepcopy(cs);wrong[1]['tick']=1
        with self.assertRaises(ValueError):select(wrong,f,group='red')
        wrong=copy.deepcopy(cs);wrong[1]['scope']['npc']=wrong[0]['scope']['npc']
        with self.assertRaises(ValueError):select(wrong,f,group='red')
        futures=copy.deepcopy(f.contexts);futures[0]['personality']['agreeableness']=0
        with self.assertRaises(ValueError):select(cs,replace(f,contexts=futures),group='red')

    def test_invalid_real_root_and_nonfinite_purpose_rejected(self):
        cs,f=fixture()
        with self.assertRaises(ValueError):select(cs,replace(f,roots={'idle':('wait','wait'),'shared':('illegal','help')}),group='red')
        with self.assertRaises(ValueError):select(cs,replace(f,purpose={'idle':0.,'shared':float('nan')}),group='red')

    def test_cooperation_projection_uses_existing_axes_without_mutation(self):
        cs,_=fixture();c=cs[0];first=concession(c);before=copy.deepcopy(c)
        c['personality']['agreeableness']=.95;self.assertGreater(concession(c),first)
        c['personality']['agreeableness']=before['personality']['agreeableness'];self.assertEqual(c,before)

    def test_material_requests_reserve_prestate_and_fail_together(self):
        w=tp.Workshop.start();choices={i:'rest' for i in range(6)}
        choices[0]=choices[1]='build';choices[2]='gather'
        after,a=tp.resolve(w,choices)
        self.assertEqual(a['material_conflicts'],[0,1]);self.assertEqual(after.points[0],0)
        self.assertEqual(after.stock[0],4);self.assertEqual(after.people[0].energy,6)

    def test_single_reservation_succeeds_and_aid_never_targets_enemy(self):
        w=tp.Workshop.start('exhausted');choices={i:'rest' for i in range(6)}
        self.assertIn('aid:1',tp.legal(w,0));self.assertNotIn('aid:4',tp.legal(w,0))
        choices.update({0:'build',2:'aid:1'});after,a=tp.resolve(w,choices)
        self.assertEqual(after.points[0],3);self.assertEqual(after.people[1].energy,6)
        self.assertEqual(after.people[2].energy,3);self.assertEqual(a['material_conflicts'],[])

    def test_two_or_four_member_world_is_not_fixed_to_three(self):
        for n in (2,4):
            w=tp.Workshop.start(members=n);self.assertEqual(len(w.people),2*n)
            after,_=tp.resolve(w,{i:'rest' for i in range(2*n)})
            self.assertEqual(after.tick,1)

    def test_resource_forecast_does_not_modify_world_or_private_context(self):
        w=tp.Workshop.start();actors=(0,1);cs=[tp.make_context(w,i,PROFILES[i],13) for i in actors]
        ds=[Policy(principle_priority='finite').choose(c,False) for c in cs];saved=copy.deepcopy(cs)
        f=tp.forecast(w,actors,cs,ds,horizon=2,max_plans=8)
        self.assertEqual(cs,saved);self.assertEqual(w.tick,0)
        self.assertTrue(all(len(root)==2 for root in f.roots.values()))
        out,a=select(cs,f,group='workshop');self.assertIn('members',a)

    def test_partial_combat_group_forecast_supplies_unsigned_living_actor(self):
        w=tc.opening('open',0);actors=(0,1)
        cs=[tc.make_person(w,i,PROFILES[i],12) for i in actors]
        ds=[Policy(principle_priority='finite').choose(c,False) for c in cs];saved=copy.deepcopy(cs)
        f=tc.forecast(w,actors,cs,ds,horizon=2,samples=2,max_plans=6)
        self.assertEqual(cs,saved);self.assertGreater(f.audit['nodes'],0)
        self.assertEqual(len(f.contexts),2)
        for roots in f.roots.values():self.assertEqual(len(roots),2)


if __name__=='__main__':unittest.main()
