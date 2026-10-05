import copy
from dataclasses import replace
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reflex.core import Policy, compile_batch
from reflex.laboratory import profiles
from reflex.resource_world import (World, Empire, COSTS, legal, act, step, income,
    achieved, winners, terminal, forecast, make_context, world_record, world_from_record)
from reflex.resource_experiment import public_event, run, replay


class ResourceWorldTests(unittest.TestCase):
    def test_costs_are_paid_and_initial_illegal_prerequisites_rejected(self):
        w = World.start(); e = w.empires[0]
        for name, cost in COSTS.items():
            key = 'build:'+name
            if key in legal(w):
                after = act(w,key).empires[0]
                self.assertEqual(after.stock,tuple(a-b for a,b in zip(e.stock,cost)))
        for key in ('research','monument','claim','invade:1'):
            self.assertNotIn(key,legal(w))
            with self.assertRaises(ValueError): act(w,key)

    def test_production_and_maintenance_run_once_per_round(self):
        w = World.start()
        initial = w.empires
        for _ in range(3): w = step(w,'wait')
        self.assertEqual(w.empires,initial)
        w = step(w,'wait')
        self.assertEqual(w.round,1)
        self.assertEqual(w.empires,tuple(income(e,World.start()) for e in initial))

    def test_no_unpaid_upkeep_or_infinite_stock(self):
        e = replace(Empire(),stock=(0,18,18,0),buildings=(0,3,3,0,0,0),land=4,army=6)
        after = income(e,World.start())
        self.assertEqual(after.stock,(0,18,18,0))
        self.assertEqual(after.army,1)
        self.assertEqual(after.shortages,5)
        self.assertEqual(after.overflow,6)

    def test_distinct_routes_require_all_components(self):
        base = Empire()
        for e, route in ((replace(base,tech=3,science=14),'science'),
                         (replace(base,monuments=2,culture=20),'culture'),
                         (replace(base,land=4,army=6),'power')):
            self.assertEqual(achieved(e),(route,))
            w = replace(World.start(),empires=(e,base,base,base))
            self.assertEqual(winners(w),(0,))
            other = next(r for r in ('science','culture','power') if r!=route)
            self.assertEqual(winners(replace(w,routes=(other,))),())
        self.assertEqual(achieved(replace(base,tech=3)),())

    def test_shared_land_is_consumed_and_invading_transfers_without_creation(self):
        e = replace(Empire(),stock=(18,18,18,18),army=8,land=2)
        rival = replace(Empire(),land=2)
        w = replace(World.start(),empires=(e,rival,Empire(),Empire()))
        after = act(w,'claim')
        self.assertEqual(after.frontier,w.frontier-1)
        after = act(w,'invade:1')
        self.assertEqual(sum(x.land for x in after.empires),sum(x.land for x in w.empires))
        self.assertEqual(after.empires[1].land,1)

    def test_forecast_is_nonmutating_and_deadline_bounded(self):
        w = replace(World.start(),round=35)
        old = world_record(w)
        for key in legal(w):
            projected,nodes = forecast(w,key,4)
            self.assertLessEqual(projected.round,36)
            self.assertEqual(nodes,1)
        self.assertEqual(world_record(w),old)

    def test_future_event_is_absent_until_public_and_current_yields_are_used(self):
        p = profiles()[0]
        w = replace(World.start(),round=7)
        before,_ = make_context(w,p,0,0,'test',horizon=4)
        self.assertNotIn('event',str(before))
        self.assertNotIn('shock',str(before))
        changed = public_event(replace(w,round=8),'food')
        c,_ = make_context(changed,p,0,0,'test',horizon=4)
        self.assertEqual(c['facts']['public_food_yield'],'1')
        self.assertNotEqual(before['actions'],c['actions'])

    def test_public_snapshot_roundtrip_and_batch_order_independence(self):
        w = World.start()
        self.assertEqual(world_from_record(world_record(w)),w)
        contexts = [make_context(w,p,1,0,'test',horizon=4)[0] for p in profiles()]
        original = copy.deepcopy(contexts)
        batch = compile_batch(contexts)
        decisions = Policy().decide(batch).records(batch)
        self.assertEqual(decisions,[Policy().choose(c) for c in contexts])
        reverse = compile_batch(contexts[::-1])
        self.assertEqual(decisions,list(reversed(Policy().decide(reverse).records(reverse))))
        self.assertEqual(contexts,original)

    def test_game_ends_with_exact_legal_win_and_replays_personality(self):
        p = profiles()[0]
        r = run(p,0,'mixed',1)
        self.assertEqual(replay(r),r['focal_actions'])
        final = world_from_record(r['final'])
        self.assertTrue(terminal(final))
        self.assertEqual(legal(final),())
        for row in r['trace']:
            w = world_from_record(row['after'])
            self.assertEqual(sum(e.land for e in w.empires)+w.frontier,12)
            for e in w.empires:
                self.assertTrue(all(0<=x<=18 for x in e.stock))

    def test_spending_research_preserves_capability_and_immediate_win(self):
        e = replace(Empire(),stock=(18,18,18,18),tech=2,science=30,buildings=(2,2,2,3,3,1))
        w = replace(World.start(routes=('science',)),empires=(e,Empire(),Empire(),Empire()))
        after = act(w,'research')
        self.assertIn(0,winners(after))
        for p in profiles()[:3]:
            c,_ = make_context(w,p,9,0,'factual-win',horizon=4)
            research = next(a for a in c['actions'] if a['id']=='research')
            self.assertGreater(research['outcomes'][0]['values']['self_direction'],0)
            self.assertEqual(Policy().choose(c,stochastic=False)['action_id'],'research')

    def test_root_cost_and_persona_alignment_do_not_include_future_greedy_choices(self):
        w = World.start()
        a,_ = make_context(w,profiles()[0],0,0,'test',horizon=1)
        b,_ = make_context(w,profiles()[0],0,0,'test',horizon=4)
        for x,y in zip(a['actions'],b['actions']):
            self.assertEqual(x['id'],y['id'])
            for k in ('values','style','cost'):
                self.assertEqual(x['outcomes'][0][k],y['outcomes'][0][k])


if __name__=='__main__': unittest.main()
