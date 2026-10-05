import copy
from dataclasses import replace
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import digest,FEATURES
from reflex.examples import context,action,effect
from reflex.deliberation import JointForecast,select
from reflex.decision_loop import DecisionLoop,Request,IntentRequest
from reflex.judgment import Binding
from reflex.planning import vector
from reflex.combat import Battle,legal,resolve,battle_record
from reflex.combat_planning import TacticalControl,forecast,no_own_collision,tactical_joint
from reflex.laboratory import profiles
from reflex.validation_experiment import combat,replay


def fixtures():
    c=context('horizon-separated',[action('guard',effect(.2)),action('advance',effect(.1))],values={'security':.8},mode='principle')
    c['state']['primary_need']='physiology'
    fc=copy.deepcopy(c)
    fc['actions']=[action('idle',effect(0,values={'security':0})),action('progress',effect(.8,values={'security':-.1}))]
    f=JointForecast((fc,),{'idle':('guard',),'progress':('advance',)},{'idle':0,'progress':.8},4,.15)
    return c,f


class DeliberationTests(unittest.TestCase):
    def test_purpose_corridor_keeps_actual_legality_and_can_prefer_risky_progress(self):
        c,f=fixtures();before=copy.deepcopy(c);d,a=select([c],f)
        self.assertEqual(d[0]['action_id'],'advance');self.assertEqual(c,before)
        self.assertEqual(d[0]['context_hash'],digest(c));self.assertEqual(a['purpose_excluded'],1)

    def test_room_for_personality_and_deliberate_loss_within_corridor(self):
        c,f=fixtures();f=replace(f,purpose={'idle':.75,'progress':.8})
        d,a=select([c],f)
        self.assertEqual(d[0]['action_id'],'guard');self.assertEqual(a['purpose_excluded'],0)

    def test_joint_cooperation_does_not_override_an_individual_value_tier(self):
        c,f=fixtures();other=copy.deepcopy(c);other['scope']['npc']='other'
        other['values']['security']=0;other['values']['power']=.8
        future=copy.deepcopy(f.contexts[0]);future['values']=other['values'];future['scope']=other['scope']
        future['actions'][0]['outcomes'][0]['values']['power']=-.1
        left=copy.deepcopy(f.contexts[0]);left['actions'][1]['outcomes'][0]['values']['power']=0
        joint=JointForecast((left,future),{'idle':('guard','guard'),'progress':('advance','advance')},
                            {'idle':.75,'progress':.8},4,.15)
        decisions,audit=select([c,other],joint)
        self.assertIsNone(decisions);self.assertFalse(audit['adopted'])

    def test_future_is_never_stored_as_selected_immediate_experience(self):
        c,f=fixtures();loop=DecisionLoop(c)
        req=Request(c,{a['id']:Binding(a['id'],'same',('objective',)) for a in c['actions']},{a['id']:() for a in c['actions']})
        result=DecisionLoop.decide_batch([(loop,req)],planner=lambda cs,ds:f)[0]
        self.assertEqual(result['decision']['action_id'],'advance')
        self.assertEqual(loop.pending['prior'][0]['objective'],.1)
        loop.observe(result['ticket'],vector(effect(-.2)))
        sample=next(iter(loop.memory.entries.values()))['samples'][0]
        self.assertEqual(sample[0],-.2)
        self.assertEqual(DecisionLoop.from_record(c,loop.record()).record(),loop.record())

    def test_invalid_joint_forecast_rolls_back_all_actors(self):
        c,f=fixtures();loop=DecisionLoop(c);before=loop.record()
        req=Request(c,{a['id']:Binding(a['id'],'same',()) for a in c['actions']},{a['id']:() for a in c['actions']})
        bad=replace(f,roots={'idle':('illegal',),'progress':('advance',)})
        with self.assertRaises(ValueError):DecisionLoop.decide_batch([(loop,req)],planner=lambda cs,ds:bad)
        self.assertEqual(loop.record(),before)
        altered=copy.deepcopy(f.contexts[0]);altered['personality']['openness']=0
        with self.assertRaises(ValueError):select([c],replace(f,contexts=(altered,)))

    def test_tactical_candidates_have_legal_conflict_free_real_roots(self):
        from reflex.combat import make_context
        from reflex.core import Policy
        w=Battle.start('choke','both');actors=(0,1,2);p=profiles()[1]
        cs=[make_context(w,i,p,0,'secure',survival_security=True) for i in actors]
        f=forecast(w,actors,cs,[Policy().choose(c) for c in cs],TacticalControl(horizon=2,samples=2))
        for root in f.roots.values():
            choices=dict(zip(actors,root));self.assertTrue(no_own_collision(choices))
            self.assertTrue(all(k in legal(w,i) for i,k in choices.items()))
        self.assertGreater(f.audit['nodes'],0)
        self.assertNotIn('actual',f.audit)

    def test_victory_route_is_selected_before_its_immediate_learning_target(self):
        c,_=fixtures();loop=DecisionLoop(c);rc=copy.deepcopy(c)
        rc['actions']=[action('alpha',effect(.9)),action('beta',effect(.1))]
        fc=copy.deepcopy(rc);fc['actions']=[action('route-a',effect(0)),action('route-b',effect(.8))]
        f=JointForecast((fc,),{'route-a':('alpha',),'route-b':('beta',)}, {'route-a':0,'route-b':.8},4,.15)
        def build(route):
            cc=copy.deepcopy(c);cc['facts']['chosen_route']=route
            return Request(cc,{a['id']:Binding(a['id'],route,('objective',)) for a in cc['actions']},{a['id']:() for a in cc['actions']})
        r=DecisionLoop.decide_batch([(loop,IntentRequest(rc,build,route_planner=lambda c:f))])[0]
        self.assertEqual(loop.strategy.chosen,'beta');self.assertEqual(r['context']['facts']['chosen_route'],'beta')
        self.assertEqual(loop.pending['binding'].situation,'beta')
        loop.observe(r['ticket'],vector(effect(.3)))
        self.assertTrue(all(key[1]=='beta' for key in loop.memory.entries))

    def test_second_game_completes_rival_turns_and_respects_real_deadline(self):
        from reflex.resource_world import World
        from reflex.resource_planning import forecast,EconomyControl
        from reflex.route_experiment import routed_context
        w=World.start(limit=1);c,_=routed_context(w,profiles()[0],80,0,None,'science')
        f=forecast(w,[c],[],EconomyControl(horizon=4))
        self.assertEqual(f.audit['nodes'],len(f.roots)*4*2)
        self.assertTrue(all(x==0 for x in f.purpose.values()))
        for a in f.contexts[0]['actions']:
            for e in a['outcomes']:self.assertTrue((abs(vector(e))<=1).all())

    def test_resource_rival_projection_preserves_the_declared_tactic(self):
        from reflex.resource_world import World,Empire,economic_move
        from reflex.resource_planning import rival_move
        w=World.start()
        varied=replace(w,empires=(replace(w.empires[0],stock=(9,8,7,12),army=6,land=2),
            replace(w.empires[1],stock=(16,2,4,11),science=13,tech=2,land=2,army=2),
            replace(w.empires[2],stock=(1,17,6,14),culture=19,monuments=1),w.empires[3]))
        for state in (w,varied,replace(varied,turn=1),replace(varied,food_yield=1,ore_yield=0)):
            for h,t in ((1,'near'),(4,'far')):
                self.assertEqual(rival_move(state,t),economic_move(state,h))

    def test_planning_episode_uses_real_rules_and_fixed_personality(self):
        p=profiles()[1];before=copy.deepcopy(p)
        r=combat(p,80,'open','secure','baseline',record=True,learn=True,survival_security=True,
                 rival='switch',planner=TacticalControl(horizon=3,samples=2,max_plans=6))
        self.assertGreater(replay(r),0);self.assertGreater(r['planning_nodes'],0)
        self.assertEqual(r['own_proposal_collisions'],0);self.assertEqual(p,before)


if __name__=='__main__':unittest.main()
