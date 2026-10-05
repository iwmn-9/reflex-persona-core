import copy
from dataclasses import replace
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.routes import RouteState,choose_route
from reflex.examples import context,action,effect
from reflex.resource_world import World,Empire,act,winners
from reflex.laboratory import profiles
from reflex.route_experiment import route_context,route_potential,routed_context,probe,run,replay


class RouteTests(unittest.TestCase):
    def test_finite_rename_invariance_and_unavailable_goal(self):
        c=context('route',[action('one',effect(.6)),action('two',effect(.3))],mode=None)
        chosen,_=choose_route(c); self.assertEqual(chosen.chosen,'one')
        renamed=copy.deepcopy(c); renamed['actions'][0]['id']='different-game-win'
        self.assertEqual(choose_route(renamed)[0].chosen,'different-game-win')
        c['actions'][0]['legal']=False
        s,d=choose_route(c,chosen); self.assertEqual(s.chosen,'two'); self.assertTrue(d['switched'])

    def test_hysteresis_rejects_small_gain_and_accepts_material_gain(self):
        c=context('route',[action('old',effect(.5)),action('new',effect(.51))],mode=None)
        c['state']['intent_action']='new'
        s,d=choose_route(c,RouteState('old'),switch_margin=.1)
        self.assertEqual(s.chosen,'old'); self.assertEqual(s.switches,0)
        self.assertEqual(d['context']['state']['intent_action'],None)
        c['actions'][1]['outcomes'][0]['objective']=.9
        s,d=choose_route(c,s,switch_margin=.1)
        self.assertEqual(s.chosen,'new'); self.assertEqual(s.switches,1)

    def test_personality_and_opportunity_both_affect_route(self):
        rows=probe(); by={(r['scenario'],r['profile']):r['chosen'] for r in rows}
        self.assertNotEqual(by['initial_world','steady'],by['initial_world','ego'])
        self.assertNotEqual(by['initial_world','growth'],by['science_capacity','growth'])
        self.assertTrue(all(r['chosen']!='power' for r in rows if r['scenario']=='military_closed'))
        with self.assertRaises(ValueError): routed_context(World.start(routes=('science',)),profiles()[0],0,0,None,'power')

    def test_spending_research_does_not_erase_completed_capability(self):
        base=World.start(); e=replace(Empire(),science=6)
        w=replace(base,empires=(e,)+base.empires[1:]); after=act(w,'research')
        self.assertGreater(route_potential(after,0,'science'),route_potential(w,0,'science'))

    def test_any_immediate_victory_has_terminal_credit(self):
        base=World.start(); e=replace(Empire(),land=4,army=4,stock=(3,4,3,4),buildings=(0,1,1,0,0,0))
        w=replace(base,empires=(e,)+base.empires[1:]); p=profiles()[0]
        c,_=routed_context(w,p,0,0,None,'culture')
        self.assertIn(0,winners(act(w,'train')))
        winning=next(a for a in c['actions'] if a['id']=='train')
        self.assertGreater(winning['outcomes'][0]['objective'],.8)
        self.assertTrue(all(-1<=a['outcomes'][0]['objective']<=1 for a in c['actions']))

    def test_complete_game_replay_and_fixed_personality(self):
        p=profiles()[0]; original=copy.deepcopy(p)
        r=run(p,0); self.assertGreater(replay(r),0); self.assertEqual(p,original)


if __name__=='__main__': unittest.main()
