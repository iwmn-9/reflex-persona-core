import copy
import json
import unittest
import numpy as np
from reflex.examples import context,action,effect
from reflex.pressure import NeedPressure,PressureConfig
from reflex.decision_loop import DecisionLoop,Request,IntentRequest
from reflex.judgment import Binding
from reflex.planning import vector


def scene(npc='actor'):
    c=context('pressure',[action('wait',effect(.05)),action('develop',effect(.04,needs={'growth':.4}))],{'growth':.1})
    c['scope']['npc']=npc
    for n in c['needs']:
        if n!='growth':c['needs'][n]=dict(supported=False,enabled=False,deficit=None)
    return c


def req(c):return Request(c,{a['id']:Binding(a['id'],'same',()) for a in c['actions']},{a['id']:() for a in c['actions']})


class PressureTests(unittest.TestCase):
    def test_stall_grace_cap_and_progress_relief(self):
        c=scene();p=NeedPressure(c['scope'])
        for t in range(3):p.observe(t,{'growth':0})
        self.assertEqual(p.debt['growth'],0)
        for t in range(3,25):p.observe(t,{'growth':0})
        self.assertEqual(p.debt['growth'],.7)
        p.observe(25,{'growth':.1});self.assertAlmostEqual(p.debt['growth'],.45)
        self.assertEqual(p.stalls['growth'],0)
        p.observe(26,{'growth':0},completed=('growth',));self.assertEqual(p.debt['growth'],0)

    def test_missing_channels_and_useful_maintenance_are_not_failure(self):
        p=NeedPressure(scene()['scope'])
        for t in range(12):p.observe(t,{'growth':0},maintained=('growth',))
        p.observe(12,{})
        self.assertEqual(sum(p.stalls.values()),0);self.assertEqual(sum(p.debt.values()),0)
        for t in range(13,18):p.observe(t,{'growth':0})
        prior=p.record();p.observe(18,{'safety':0})
        self.assertEqual(p.debt['growth'],prior['debt']['growth']);self.assertEqual(p.stalls['growth'],prior['stalls']['growth'])

    def test_apply_only_supported_needs_and_fixed_axes(self):
        c=scene();p=NeedPressure(c['scope'])
        for t in range(10):p.observe(t,{'growth':0,'safety':0})
        before=copy.deepcopy(c);a=p.apply(c)
        self.assertEqual(c,before);self.assertGreater(a['needs']['growth']['deficit'],c['needs']['growth']['deficit'])
        self.assertEqual(a['needs']['safety'],c['needs']['safety'])
        for k in ('personality','values','actions','state'):self.assertEqual(a[k],c[k])

    def test_invalid_observation_is_atomic(self):
        p=NeedPressure(scene()['scope']);prior=p.record()
        for progress,labels in [({'growth':np.nan},()),({'growth':2},()),({'unknown':0},()),({'growth':True},()),({},('growth',))]:
            with self.assertRaises(ValueError):p.observe(0,progress,maintained=labels)
            self.assertEqual(prior,p.record())
        p.observe(0,{'growth':0});prior=p.record()
        with self.assertRaises(ValueError):p.observe(0,{'growth':0})
        self.assertEqual(prior,p.record())

    def test_owner_and_checkpoint_validation(self):
        c=scene();p=NeedPressure(c['scope']);p.observe(0,{'growth':0})
        restored=NeedPressure.from_record(c['scope'],json.loads(json.dumps(p.record())))
        self.assertEqual(p.record(),restored.record())
        with self.assertRaises(ValueError):p.apply(scene('other'))
        rc=copy.deepcopy(c);rc['scope']['game']='route-phase';self.assertGreaterEqual(p.apply_route(rc)['needs']['growth']['deficit'],.1)
        rc['scope']['npc']='other'
        with self.assertRaises(ValueError):p.apply_route(rc)
        for field,value in [('last_tick',True),('last_tick',-2)]:
            r=p.record();r[field]=value
            with self.assertRaises(ValueError):NeedPressure.from_record(c['scope'],r)
        r=p.record();r['debt']['growth']=.9
        with self.assertRaises(ValueError):NeedPressure.from_record(c['scope'],r)

    def test_config_bounded(self):
        for kw in ({'grace':True},{'grace':101},{'rise':0},{'limit':float('inf')},{'progress_floor':False}):
            with self.assertRaises(ValueError):PressureConfig(**kw)

    def test_unconfigured_loop_keeps_original_checkpoint_and_feedback_shape(self):
        c=scene();loop=DecisionLoop(c)
        self.assertNotIn('pressure',loop.record())
        r=loop.decide(req(c));update=loop.observe(r['ticket'],vector(effect()))
        self.assertNotIn('pressure',update);self.assertNotIn('pressure',loop.record())
        saved=loop.record();self.assertEqual(saved,DecisionLoop.from_record(c,saved).record())

    def test_loop_observation_transaction_and_restore(self):
        c=scene();loop=DecisionLoop(c,pressure=NeedPressure(c['scope']))
        for tick in range(6):
            c['tick']=tick;r=loop.decide(req(c));old=copy.deepcopy(loop.pending);debt=loop.pressure.record()
            with self.assertRaises(ValueError):loop.observe(r['ticket'],vector(effect()),need_progress={'growth':float('nan')})
            self.assertEqual(loop.pending,old);self.assertEqual(loop.pressure.record(),debt)
            loop.observe(r['ticket'],vector(effect()),need_progress={'growth':0})
        saved=json.loads(json.dumps(loop.record()));restored=DecisionLoop.from_record(c,saved)
        self.assertEqual(saved,restored.record());c['tick']=6
        self.assertEqual(loop.decide(req(c)),restored.decide(req(c)))

    def test_censored_effects_can_update_actual_goal_progress_only(self):
        c=scene();loop=DecisionLoop(c,pressure=NeedPressure(c['scope']))
        for tick in range(9):
            c['tick']=tick;r=loop.decide(req(c));loop.abandon(r['ticket'],need_progress={'growth':0})
        self.assertGreater(loop.pressure.debt['growth'],0);self.assertEqual(len(loop.memory.entries),0)
        r=loop.record();self.assertEqual(r,DecisionLoop.from_record(c,r).record())

    def test_bad_censor_progress_does_not_discard_outstanding_ticket(self):
        c=scene();loop=DecisionLoop(c,pressure=NeedPressure(c['scope']));r=loop.decide(req(c))
        before=copy.deepcopy(loop.pending)
        with self.assertRaises(ValueError):loop.abandon(r['ticket'],need_progress={},completed=('growth',))
        self.assertEqual(loop.pending,before);self.assertIsNotNone(loop.memory.pending)

    def test_pressure_batch_single_identity_and_route_application(self):
        cs=[scene(str(i)) for i in range(3)];batch=[DecisionLoop(c,pressure=NeedPressure(c['scope'])) for c in cs]
        single=[DecisionLoop(c,pressure=NeedPressure(c['scope'])) for c in cs]
        for tick in range(7):
            requests=[]
            for c in cs:
                c['tick']=tick;rc=copy.deepcopy(c);rc['scope']['game']='routes';requests.append(IntentRequest(rc,lambda key,c=c:req(c)))
            rr=DecisionLoop.decide_batch(list(zip(batch,requests)))
            for l,s,r,request in zip(batch,single,rr,requests):
                self.assertEqual(r,s.decide(request));self.assertEqual(r['strategy']['context']['needs']['growth']['deficit'],r['context']['needs']['growth']['deficit'])
                l.abandon(r['ticket'],need_progress={'growth':0});s.abandon(r['ticket'],need_progress={'growth':0})
                self.assertEqual(l.record(),s.record())

    def test_batch_invalid_actor_does_not_apply_pressure_to_others(self):
        a,b=scene('a'),scene('b');la,lb=DecisionLoop(a,pressure=NeedPressure(a['scope'])),DecisionLoop(b,pressure=NeedPressure(b['scope']))
        original=(la.record(),lb.record());b['seed']+=1
        with self.assertRaises(ValueError):DecisionLoop.decide_batch([(la,req(a)),(lb,req(b))])
        self.assertEqual(original,(la.record(),lb.record()))


if __name__=='__main__':unittest.main()
