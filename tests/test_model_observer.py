import copy
from dataclasses import asdict,replace
import unittest
from unittest.mock import patch
from reflex.core import compile_batch,Policy,digest
from reflex.examples import context,action,effect
from reflex.flow_rollout import rollout
from reflex.model_observer import ModelObserver,ModelFeedback
from reflex.progress import PurposeFeedback,ProgressConfig
from reflex.pressure import NeedPressure,PressureConfig
from reflex.purpose_plan import PersonaProgressWatch
from reflex.delivery_world import Depot,DeliveryProbe,step,legal,terminal,goal
from reflex.laboratory import profiles
from reflex.observed_transfer import feedback,can_finish,run,jobs
from reflex.decision_loop import DecisionLoop,Request
from reflex.judgment import Binding


class ModelObserverTests(unittest.TestCase):
    def fixture(self):
        p=DeliveryProbe(dict(genre='delivery',limit=5),profiles()[0]);w=p.start();c=p.observe(w)
        watch=PersonaProgressWatch(c['scope'],ProgressConfig(grace=2,repeat_limit=2))
        pressure=NeedPressure(c['scope'],PressureConfig(grace=2))
        m=ModelObserver(c['scope'],purpose=p.purpose,feedback=lambda a,b,k:feedback(p,a,b,k,True,True),
            progress=watch,pressure=pressure)
        return p,w,c,watch,pressure,m

    def test_branch_forks_do_not_update_real_owner_or_each_other(self):
        p,w,c,watch,pressure,m=self.fixture();original=copy.deepcopy((watch.record(),pressure.record(),m.record()))
        paths,audit=rollout(c,w,observe=p.observe,advance=p.advance,terminal=p.terminal,horizon=3,seeds=(0,1),observer=m)
        self.assertEqual((watch.record(),pressure.record(),m.record()),original)
        for root,rows in audit.items():
            self.assertEqual(rows[0],{**rows[1],'seed':0})
            self.assertEqual(rows[0]['modeled_observations'][0]['checkpoint']['progress']['last_tick'],0)

    def test_prepared_root_does_not_apply_pressure_twice(self):
        p,w,c,watch,pressure,m=self.fixture();m.pressure.debt['growth']=.2
        c=m.pressure.apply(c)
        effective,_,_,_=m.prepare(w,c,prepared=True)
        self.assertEqual(effective['needs']['growth']['deficit'],.5)
        raw=p.observe(w);effective,_,_,_=m.prepare(w,raw)
        self.assertEqual(effective['needs']['growth']['deficit'],.5)

    def test_model_reflex_actions_and_checkpoints_match_actual_reflex_loop(self):
        p,w,c,watch,pressure,m=self.fixture();loop=DecisionLoop(c,progress=watch,pressure=pressure)
        start=w
        req=lambda cc:Request(cc,{k:Binding(k,'public',()) for k in p.keys(w)},
            {k:() for k in p.keys(w)},purpose=p.purpose(w,cc))
        first=DecisionLoop.decide_batch([(loop,req(c))],False)[0]
        paths,audit=rollout(c,start,observe=p.observe,advance=p.advance,terminal=p.terminal,horizon=3,observer=m)
        root=first['decision']['action_id'];predicted=audit[root][0]
        actual=[];previous=None
        for tick in range(3):
            if tick:
                cc=p.observe(w,loop.state)
                request=Request(cc,{k:Binding(k,'public',()) for k in p.keys(w)},
                    {k:() for k in p.keys(w)},purpose=p.purpose(w,cc,previous))
                result=DecisionLoop.decide_batch([(loop,request)],False)[0]
            else:result=first
            key=result['decision']['action_id'];before=w;w,_=p.actual(w,key,0)
            fb=feedback(p,before,w,key,True,True)
            loop.abandon(result['ticket'],need_progress=fb.needs,maintained=fb.maintained,
                completed=fb.completed,purpose_feedback=fb.purpose)
            actual.append(key);observed=predicted['modeled_observations'][tick]['checkpoint']
            self.assertEqual(observed['progress'],loop.progress.record())
            self.assertEqual(observed['pressure'],loop.pressure.record());previous=before
        self.assertEqual(actual,predicted['actions'])

    def test_default_rollout_preserves_no_observer_trace_contract(self):
        p,w,c,_,_,_=self.fixture()
        _,audit=rollout(c,w,observe=p.observe,advance=p.advance,terminal=p.terminal,horizon=2)
        self.assertTrue(all('modeled_observations' not in rows[0] for rows in audit.values()))

    def test_failed_feedback_is_transactional_and_owner_is_enforced(self):
        p,w,c,_,_,m=self.fixture();c,req,_,_=m.prepare(w,c)
        before=m.record();m.feedback=lambda a,b,k:ModelFeedback(PurposeFeedback(0.,0.),{'unknown':0.})
        with self.assertRaises(ValueError):m.advance(w,step(w,'rest'),'rest',c,req)
        self.assertEqual(m.record(),before)
        bad=copy.deepcopy(c);bad['scope']['npc']='other'
        with self.assertRaises(ValueError):m.prepare(w,bad)
        with self.assertRaises(ValueError):m.advance(w,step(w,'rest'),'rest',bad,req)

    def test_terminal_does_not_consume_modeled_observations_after_settlement(self):
        p,w,c,_,_,m=self.fixture();w=replace(w,limit=1);c=p.observe(w)
        _,audit=rollout(c,w,observe=p.observe,advance=p.advance,terminal=p.terminal,horizon=4,observer=m)
        self.assertTrue(all(len(r[0]['modeled_observations'])==1 for r in audit.values()))

    def test_modeled_mask_is_enforced_by_custom_continuation(self):
        p,w,c,watch,pressure,m=self.fixture();w=replace(w,energy=9);c=p.observe(w)
        watch.stalls=3;watch.peak=0.;watch.ready_peak=0.;m.progress=copy.deepcopy(watch)
        with self.assertRaisesRegex(ValueError,'modeled progress mask'):
            rollout(c,w,observe=p.observe,advance=p.advance,terminal=p.terminal,horizon=2,
                observer=m,continuation=lambda w,c:'rest')

    def test_pressure_only_and_watch_only_require_matching_feedback(self):
        p,w,c,watch,pressure,_=self.fixture()
        for watched,pressured in ((True,False),(False,True)):
            m=ModelObserver(c['scope'],purpose=p.purpose,
                feedback=lambda a,b,k:feedback(p,a,b,k,watched,pressured),
                progress=watch if watched else None,pressure=pressure if pressured else None)
            rollout(c,w,observe=p.observe,advance=p.advance,terminal=p.terminal,horizon=2,observer=m)
        with self.assertRaises(ValueError):ModelObserver(c['scope'],purpose=p.purpose,feedback=p.feedback)


class DeliveryRulesTests(unittest.TestCase):
    def test_resource_conservation_and_two_routes(self):
        w=Depot(energy=9)
        for k in ('collect','collect','load','aid'):
            before=w;w=step(w,k)
            if k=='load':self.assertEqual(w.stock+w.cargo,before.stock+before.cargo)
        self.assertEqual((w.stock,w.cargo,w.community,w.delivered),(1,0,3,0))
        for k in ('rest','load','aid'):w=step(w,k)
        self.assertEqual(goal(w).status,'success');self.assertFalse(legal(w))

    def test_deadline_and_actual_completion_are_distinct(self):
        self.assertEqual(goal(Depot(tick=8,limit=8,delivered=5)).value,0.)
        self.assertEqual(goal(Depot(tick=8,limit=8,delivered=6)).value,1.)
        with self.assertRaises(ValueError):step(Depot(energy=0),'collect')

    def test_reachability_oracle_is_not_used_by_any_decision(self):
        p=DeliveryProbe(dict(genre='delivery',limit=4),profiles()[0]);c=p.observe(p.start())
        with patch('reflex.observed_transfer.can_finish',side_effect=AssertionError('oracle reached decision')):
            rollout(c,p.start(),observe=p.observe,advance=p.advance,terminal=p.terminal,horizon=3,
                assess=lambda w:p.goal(w).record())
        self.assertTrue(can_finish(Depot(limit=10)))
        self.assertFalse(can_finish(Depot(limit=1)))

    def test_all_transfer_conditions_are_declared_before_execution(self):
        tasks=list(jobs());self.assertEqual(len(tasks),204)
        self.assertTrue(all(t[4] not in (820,821,822,823) for t in tasks if t[1]['genre']=='combat'))
        self.assertEqual({t[3] for t in tasks if t[0]=='diagnostic'},
            {'goal','watch','pressure','release','observed','verified'})
        self.assertEqual({t[4] for t in tasks if t[0]=='holdout' and t[1]['genre']=='combat'},{250,251})

    def test_verified_route_uses_existing_strict_positive_gain_gate(self):
        r=run(dict(genre='delivery',limit=2),profiles()[0],'verified',250)
        self.assertTrue(all(t['watch']['config']['proof_margin']==0. for t in r['trace']))
        self.assertTrue(all('modeled_observations' not in e for t in r['trace'] for e in t['endpoints']))


if __name__=='__main__':unittest.main()
