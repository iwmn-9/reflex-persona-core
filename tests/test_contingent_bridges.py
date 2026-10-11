import copy,unittest
from reflex.contingent_games import BRIDGES,make_bridge,Frame
from reflex.laboratory import profiles
from reflex.core import compile_batch,digest
from reflex.contingent_trials import play


class ContingentBridgeTests(unittest.TestCase):
    def test_replacing_public_projection_removes_stale_chunks(self):
        from reflex.public_perception import with_public
        b=make_bridge(profiles()[0],dict(game='navigation'));c=b.context(b.start(),None)
        large=with_public(c,{'public':'x'*500});saved=digest(large)
        small=with_public(large,{'public':'y'})
        self.assertEqual(small,with_public(c,{'public':'y'}));self.assertEqual(digest(large),saved)
        with self.assertRaises(ValueError):with_public(c,{'public':float('nan')})

    def test_all_local_bridge_observations_follow_common_contract(self):
        for name in BRIDGES:
            with self.subTest(game=name):
                bridge=make_bridge(profiles()[0],dict(game=name));w=bridge.project(bridge.start());c=bridge.observe(w)
                compile_batch([c]);self.assertTrue(any(a['legal'] for a in c['actions']))

    def test_private_no_thanks_future_order_cannot_change_forecast(self):
        bridge=make_bridge(profiles()[0],dict(game='no_thanks'));w=bridge.project(bridge.start())
        c=bridge.observe(w);after=bridge.advance(w,'PASS',401)
        bridge.actual_deck=(bridge.actual_deck[0],)+bridge.actual_deck[:0:-1]
        self.assertEqual(digest(c),digest(bridge.observe(w)))
        self.assertEqual(after,bridge.advance(w,'PASS',401))

    def test_unrevealed_gambling_change_is_not_a_forecast_feature(self):
        a=make_bridge(profiles()[0],dict(game='gambling',scenario='opportunity'))
        b=make_bridge(profiles()[0],dict(game='gambling',scenario='change'))
        x=a.project(a.start());y=b.project(b.start())
        self.assertEqual(digest(a.observe(x)),digest(b.observe(y)))
        self.assertEqual(a.advance(x,'swing',401),b.advance(y,'swing',401))

    def test_actual_intelligence_policy_is_not_used_as_duel_model(self):
        a=make_bridge(profiles()[0],dict(game='duel',variant=0));b=make_bridge(profiles()[0],dict(game='duel',variant=1))
        x=a.project(a.start());y=b.project(b.start())
        self.assertEqual(digest(a.observe(x)),digest(b.observe(y)))
        nx,fx=a.advance(x,'STRIKE',401);ny,fy=b.advance(y,'STRIKE',401)
        self.assertEqual(a.record(nx),b.record(ny));self.assertEqual(fx,fy)

    def test_unrevealed_empire_event_and_trap_phase_do_not_enter_perception(self):
        for name,key,changed in (('empire_race','wait',{'change':True}),
            ('cyclic_arena','GUARD',{'phase_offset':10}),('shared_commons','SECURE',{'phase_offset':10}),
            ('branching_delivery','CHECK',{'phase_offset':10})):
            a=make_bridge(profiles()[0],dict(game=name));b=make_bridge(profiles()[0],dict(game=name,**changed))
            x=a.project(a.start());y=b.project(b.start())
            self.assertEqual(digest(a.observe(x)),digest(b.observe(y)))
            nx,fx=a.advance(x,key,401);ny,fy=b.advance(y,key,401)
            self.assertEqual(a.record(nx),b.record(ny));self.assertEqual(fx,fy)

    def test_per_actor_intent_is_not_shared_through_cached_observations(self):
        b=make_bridge(profiles()[0],dict(game='connect_four'));w=b.start();c=b.observe(w)
        state=copy.deepcopy(c['state']);state['intent_action']='DROP:4'
        changed=b.observe(w,state);changed['facts']['own_discs']='modified'
        self.assertEqual(b.observe(w)['state'],c['state'])
        self.assertEqual(b.observe(w)['facts'],c['facts'])

    def test_mutable_game_identity_is_not_mistaken_for_immutable_position(self):
        b=make_bridge(profiles()[0],dict(game='navigation'));w=b.start();before=digest(b.observe(w))
        w.position.state['energy']=.2
        self.assertNotEqual(digest(b.observe(w)),before)
        self.assertEqual(len(b._contexts),0)

    def test_team_terminal_requires_shared_result_not_owner_survival(self):
        from dataclasses import replace
        from reflex.combat import terminal
        b=make_bridge(profiles()[0],dict(game='team_combat'));w=b.start()
        units=(replace(w.position.units[0],hp=0),)+w.position.units[1:]
        dead=Frame(replace(w.position,units=units))
        self.assertFalse(b.terminal(dead));self.assertEqual(b.goal(dead).status,'running')

    def test_rival_wall_is_not_charged_as_owners_wall_inventory(self):
        from reflex.quoridor import Position,consequences
        before=Position.start(4);after=before.play('M:4:1').play('H:0:0')
        row,audit=consequences(before,after)
        self.assertEqual(row['cost'],0.);self.assertFalse(audit['wall'])
        own=before.play('H:0:0').play('M:7:4')
        row,audit=consequences(before,own)
        self.assertEqual(row['cost'],.09);self.assertTrue(audit['wall'])

    def test_exhausted_social_position_keeps_float_numeric_contract(self):
        from reflex.contingent_games import SocialState
        for name in ('social_allocation','social_patrol'):
            b=make_bridge(profiles()[0],dict(game=name));w=Frame(SocialState(energy=0,reserve=0,health=.5))
            compile_batch([b.observe(w)])
            after,row=b.modeled_advance(w,'recover',401);compile_batch([b.observe(after)])

    def test_laboratory_forced_intent_is_consumed_for_one_transition_only(self):
        b=make_bridge(profiles()[0],dict(game='laboratory'));w=b.project(b.start())
        for _ in range(2):w,row=b.modeled_advance(w,'WORK_RISK',401)
        c=b.observe(w);compile_batch([c])
        self.assertFalse(next(a for a in c['actions'] if a['id']=='WORK_RISK')['legal'])
        self.assertIsInstance(w.position.population.policy,type(b.start().position.population.policy))

    def test_whole_deterministic_episode_replays_without_empirical_forecast_learning(self):
        case=dict(game='navigation');p=profiles()[0]
        a=play(case,p,1210,horizon=2,revisions=1,patch_budget=2)
        b=play(case,p,1210,horizon=2,revisions=1,patch_budget=2)
        self.assertEqual(digest(a),digest(b));self.assertEqual(a['goal']['status'],'success')
        self.assertTrue(all(t['conditional']['legal_roots']>0 for t in a['trace']))

    def test_caller_policy_is_used_by_future_owner_in_both_trial_methods(self):
        from dataclasses import replace
        from unittest.mock import patch
        from reflex.core import Policy
        from reflex.examples import context,action,effect
        from reflex.purpose_plan import Goal
        class RestPolicy(Policy):
            def decide(self,b,*args,**kwargs):
                d=super().decide(b,*args,**kwargs);choices=d.action.copy()
                for i,ids in enumerate(b.ids):
                    if 'rest' in ids:choices[i]=ids.index('rest')
                return replace(d,action=choices)
        class Game:
            target='probe';unit='own-decisions';seeds=(401,402)
            def start(self):return (0,0)
            def project(self,w):return w
            def terminal(self,w):return w[0]>=2
            def observe(self,w,state=None):
                c=context('policy-port',[action('go',effect(1.)),action('rest',effect())]);c['tick']=w[0]
                if state is not None:c['state']=state
                return c
            def advance(self,w,key,seed):return (w[0]+1,w[1]+int(key=='go')),effect(float(key=='go'))
            modeled_advance=actual=advance
            def record(self,w):return dict(tick=w[0],score=w[1])
            def goal(self,w):return Goal(self.target,'scored' if self.terminal(w) else 'running',w[1]/2)
        for method in ('goal-reflex','conditional'):
            with patch('reflex.contingent_trials.make_bridge',return_value=Game()):
                r=play({},profiles()[0],12,method,horizon=2,revisions=0,policy=RestPolicy())
            future=[b['actions'][1] for t in r['trace'] for b in t['branches'] if len(b['actions'])>1]
            self.assertTrue(future);self.assertEqual(set(future),{'rest'})


if __name__=='__main__':unittest.main()
