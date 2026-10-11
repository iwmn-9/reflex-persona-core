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
from reflex.team_observation import PartnerMemory


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

    def test_unilateral_reference_does_not_claim_others_will_sacrifice(self):
        cs,f=fixture();fs=copy.deepcopy(f.contexts)
        roots={'idle':('wait','wait'),'dream_a':('wait','help'),
               'dream_b':('help','wait'),'shared':('help','help')}
        for i,c in enumerate(fs):
            c['actions']=[action('idle',effect()),
                action('dream_a',effect(needs={'physiology':1 if i==0 else -1})),
                action('dream_b',effect(needs={'physiology':-1 if i==0 else 1})),
                action('shared',effect(needs={'physiology':.2}))]
        forecast=JointForecast(tuple(fs),roots,{'idle':0.,'dream_a':1.,'dream_b':1.,'shared':1.},4,.15)
        ds,a=select(cs,forecast,group='red');self.assertIsNone(ds)
        ds,a=select(cs,forecast,group='red',outside=('wait','wait'))
        self.assertTrue(a['adopted']);self.assertEqual([d['action_id'] for d in ds],['help','help'])
        self.assertTrue(all(g>0 for g in a['member_gains'].values()))

    def test_unilateral_reference_must_include_real_independent_proposal(self):
        cs,f=fixture();broken=replace(f,roots={'idle':('wait','help'),'shared':('help','wait')})
        with self.assertRaises(ValueError):select(cs,broken,group='red',outside=('wait','wait'))

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

    def test_uncertain_unsigned_partner_branches_never_add_a_voting_owner(self):
        w=tp.Workshop.start();actors=(0,1);cs=[tp.make_context(w,i,PROFILES[i],13) for i in actors]
        ds=[Policy(principle_priority='finite').choose(c,False) for c in cs]
        f=tp.forecast(w,actors,cs,ds,horizon=2,max_plans=8,partner_model='uncertain')
        self.assertEqual(f.audit['scenarios'],2);self.assertEqual(f.audit['unsigned'],[2])
        self.assertEqual(len(f.contexts),2)
        self.assertTrue(all(len(a['outcomes'])==2 for a in f.contexts[0]['actions']))

    def test_weighted_partner_model_retains_probability_mass_and_no_votes(self):
        w=tp.Workshop.start();actors=(0,1);cs=[tp.make_context(w,i,PROFILES[i],13) for i in actors]
        ds=[Policy(principle_priority='finite').choose(c,False) for c in cs]
        f=tp.forecast(w,actors,cs,ds,horizon=2,max_plans=8,partner_model='uncertain',partner_weights=(.1,.9))
        for a in f.contexts[0]['actions']:self.assertEqual([o['p'] for o in a['outcomes']],[.1,.9])
        with self.assertRaises(ValueError):tp.forecast(w,actors,cs,ds,partner_weights=(1.,1.))

    def test_public_observation_does_not_update_on_an_imagined_forecast(self):
        book=PartnerMemory('g','e','red',(2,));before=book.receipt()
        models={2:{'hold':{'rest':1.,'build':0.},'independent':{'rest':0.,'build':1.}}}
        weights=book.begin(book.scope,0,models);self.assertEqual(weights[2],(.5,.5))
        self.assertEqual(book.receipt(),before)
        with self.assertRaises(ValueError):book.observe(book.scope,0,{2:'build'},witnessed=False)
        self.assertEqual(book.receipt(),before)
        book.observe(book.scope,0,{2:'build'},witnessed=True)
        self.assertGreater(book.weights(2)[1],.5)

    def test_missing_observation_is_abandoned_without_success_or_failure(self):
        book=PartnerMemory('g','e','red',(2,));models={2:{'hold':{'rest':1.},'independent':{'rest':1.}}}
        book.begin(book.scope,0,models);book.abandon(book.scope,0)
        self.assertEqual(book.receipt()['beliefs']['2']['observations'],0)
        with self.assertRaises(ValueError):book.begin(book.scope,0,models)

    def test_wrong_episode_or_duplicate_action_cannot_train_partner(self):
        book=PartnerMemory('g','e','red',(2,));models={2:{'hold':{'rest':1.},'independent':{'rest':1.}}}
        with self.assertRaises(ValueError):book.begin(('g','other','red'),0,models)
        book.begin(book.scope,0,models);book.observe(book.scope,0,{2:'rest'},witnessed=True)
        with self.assertRaises(ValueError):book.observe(book.scope,0,{2:'rest'},witnessed=True)
        self.assertEqual(book.receipt()['beliefs']['2']['observations'],0)

    def test_behavior_shift_can_revise_prior_without_changing_any_personality(self):
        book=PartnerMemory('g','e','red',(2,));models={2:{'hold':{'rest':1.,'build':0.},'independent':{'rest':0.,'build':1.}}}
        for tick in range(4):
            book.begin(book.scope,tick,models);book.observe(book.scope,tick,{2:'build'},witnessed=True)
        self.assertGreater(book.weights(2)[1],.9)
        for tick in range(4,8):
            book.begin(book.scope,tick,models);book.observe(book.scope,tick,{2:'rest'},witnessed=True)
        self.assertGreater(book.weights(2)[0],book.weights(2)[1])

    def test_unknown_but_legal_behavior_does_not_identify_one_hypothesis(self):
        book=PartnerMemory('g','e','red',(2,));models={2:{'hold':{'rest':1.,'build':0.,'gather':0.},
            'independent':{'rest':0.,'build':1.,'gather':0.}}}
        book.begin(book.scope,0,models);book.observe(book.scope,0,{2:'gather'},witnessed=True)
        self.assertEqual(book.weights(2),(.5,.5))

    def test_public_partner_learning_survives_full_combat_and_resource_episodes(self):
        from tools.run_team_cooperation import run_game
        for genre,scenario in (('combat','open'),('projects','balanced')):
            r=run_game((genre,scenario,0,9600,'partial','bargain_observed'))
            self.assertTrue(r['trace'])
            self.assertTrue(all(t['partner_learning'] is not None for t in r['trace']))
            last=r['trace'][-1]['partner_learning']['after']
            self.assertEqual(last['last_tick'],r['ticks']-1)


if __name__=='__main__':unittest.main()
