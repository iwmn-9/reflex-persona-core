import copy
import json
from dataclasses import replace
import unittest
from reflex.examples import context,action,effect
from reflex.progress import Activity,PurposeRequest,PurposeFeedback,ProgressWatch,ProgressConfig
from reflex.decision_loop import DecisionLoop,Request
from reflex.judgment import Binding
from reflex.deliberation import JointForecast,select
from reflex.planning import vector


def scene():
    return context('purpose-wait',[action('wait',effect(.2,values={'security':.3})),action('work',effect(.1,values={'security':-.1}))],
        values={'security':.8},mode='principle')


def purpose(kind='wait',patience=3,readiness=0.,condition='opening'):
    a=Activity(kind,'opening-or-preparation',condition,patience,'opening-visible' if kind=='wait' else '')
    return PurposeRequest(0.,readiness,{'wait':a,'work':Activity('attempt','purpose-work')})


def req(c,p):
    return Request(c,{a['id']:Binding(a['id'],'same',()) for a in c['actions']},
        {a['id']:() for a in c['actions']},purpose=p)


class ProgressTests(unittest.TestCase):
    def test_three_strategic_waits_then_release_without_renewing_prediction(self):
        c=scene();loop=DecisionLoop(c,progress=ProgressWatch(c['scope']))
        choices=[]
        for t in range(4):
            c['tick']=t;r=loop.decide(req(c,purpose(condition='prediction-'+str(t))))
            choices.append(r['decision']['action_id'])
            loop.abandon(r['ticket'],purpose_feedback=PurposeFeedback(0,0))
        self.assertEqual(choices,['wait','wait','wait','work'])
        self.assertEqual(loop.personality,c['personality']);self.assertEqual(loop.values,c['values'])
        self.assertEqual(loop.memory.entries,{})

    def test_actual_preparation_can_extend_useful_waiting(self):
        c=scene();w=ProgressWatch(c['scope'])
        for t in range(8):
            p=purpose(readiness=t/20,patience=2)
            allowed,_=w.mask(c,p);self.assertIn('wait',allowed)
            w.observe(t,'wait',p,PurposeFeedback(0,(t+1)/20))
        self.assertEqual(w.wait_spent,0);self.assertEqual(w.stalls,0)

    def test_small_real_preparation_accumulates_instead_of_being_lost_each_turn(self):
        c=scene();w=ProgressWatch(c['scope'])
        for t in range(6):
            p=purpose(readiness=t*.003)
            w.observe(t,'wait',p,PurposeFeedback(0,(t+1)*.003))
        self.assertEqual(w.wait_spent,0);self.assertEqual(w.stalls,0)

    def test_oscillation_is_not_new_preparation(self):
        c=scene();w=ProgressWatch(c['scope'])
        for t in range(6):
            p=purpose(readiness=.4,kind='idle')
            w.observe(t,'wait',p,PurposeFeedback(0,.2 if t%2 else .4))
        allowed,_=w.mask(c,purpose(readiness=.4,kind='idle'))
        self.assertNotIn('wait',allowed)

    def test_actual_useful_maintenance_keeps_a_valid_stop(self):
        c=scene();w=ProgressWatch(c['scope']);p=purpose(kind='maintain')
        for t in range(12):
            self.assertIn('wait',w.mask(c,p)[0]);w.observe(t,'wait',p,PurposeFeedback(0,0,maintained=True))
        self.assertEqual(w.stalls,0)
        for t in range(12,15):w.observe(t,'wait',p,PurposeFeedback(0,0))
        self.assertNotIn('wait',w.mask(c,p)[0])

    def test_deterministic_failure_repetition_and_uncertain_attempt_are_different(self):
        c=scene();w=ProgressWatch(c['scope']);p=purpose(kind='attempt')
        for t in range(3):w.observe(t,'wait',p,PurposeFeedback(0,0))
        saved=json.loads(json.dumps(w.record()))
        self.assertEqual(saved,ProgressWatch.from_record(c['scope'],saved).record())
        self.assertNotIn('wait',w.mask(c,p)[0])
        stochastic=replace(p,activities={**p.activities,'wait':Activity('uncertain','positive-probability-attempt')})
        self.assertIn('wait',w.mask(c,stochastic)[0])
        changed=purpose(kind='attempt',condition='obstruction-removed')
        self.assertIn('wait',w.mask(c,changed)[0])

    def test_useful_attempt_ends_a_wait_episode_without_faking_goal_progress(self):
        c=scene();w=ProgressWatch(c['scope']);p=purpose()
        for t in range(3):w.observe(t,'wait',p,PurposeFeedback(0,0))
        self.assertNotIn('wait',w.mask(c,p)[0])
        active=replace(p,activities={**p.activities,'work':Activity('uncertain','useful-uncertain-attempt')})
        w.observe(3,'work',active,PurposeFeedback(0,0))
        self.assertEqual(w.stalls,4);self.assertEqual(w.peak,0)
        self.assertIn('wait',w.mask(c,p)[0])
        for t in range(4,7):w.observe(t,'wait',p,PurposeFeedback(0,0))
        self.assertNotIn('wait',w.mask(c,p)[0])

    def test_only_dangerous_replacements_do_not_force_departure(self):
        c=scene();w=ProgressWatch(c['scope']);p=purpose(kind='idle')
        p=replace(p,activities={**p.activities,'work':Activity('attempt','dangerous-purpose-attempt',replacement=False)})
        for t in range(3):w.observe(t,'wait',p,PurposeFeedback(0,0))
        allowed,audit=w.mask(c,p)
        self.assertIn('wait',allowed);self.assertTrue(audit['unresolved'])

    def test_missing_result_consumes_elapsed_wait_but_is_not_failure(self):
        c=scene();w=ProgressWatch(c['scope']);p=purpose(patience=2)
        for t in range(2):w.observe(t,'wait',p)
        self.assertEqual(w.stalls,0);self.assertEqual(w.failures,{})
        self.assertNotIn('wait',w.mask(c,p)[0])

    def test_no_meaningful_alternative_does_not_force_motion_or_risk(self):
        c=scene();w=ProgressWatch(c['scope']);p=purpose(kind='idle')
        p=replace(p,activities={k:Activity('idle','no-supported-purpose') for k in p.activities})
        for t in range(3):w.observe(t,'wait',p,PurposeFeedback(0,0))
        allowed,audit=w.mask(c,p)
        self.assertEqual(allowed,{'wait','work'});self.assertTrue(audit['unresolved']);self.assertFalse(audit['applied'])

    def test_checkpoint_owner_timeline_and_wait_budget_are_preserved(self):
        c=scene();loop=DecisionLoop(c,progress=ProgressWatch(c['scope']))
        for t in range(3):
            c['tick']=t;r=loop.decide(req(c,purpose()));loop.abandon(r['ticket'],purpose_feedback=PurposeFeedback(0,0))
        r=json.loads(json.dumps(loop.record()));restored=DecisionLoop.from_record(c,r);self.assertEqual(r,restored.record())
        c['tick']=3
        self.assertEqual(loop.decide(req(c,purpose())),restored.decide(req(c,purpose())))
        r['progress']['last_tick']=1
        with self.assertRaises(ValueError):DecisionLoop.from_record(c,r)

    def test_bad_purpose_feedback_keeps_ticket_memory_and_wait_budget(self):
        c=scene();loop=DecisionLoop(c,progress=ProgressWatch(c['scope']));r=loop.decide(req(c,purpose()))
        pending=copy.deepcopy(loop.pending);state=loop.progress.record()
        with self.assertRaises(ValueError):loop.observe(r['ticket'],vector(effect()),purpose_feedback={'level':0})
        self.assertEqual(loop.pending,pending);self.assertEqual(loop.progress.record(),state);self.assertIsNotNone(loop.memory.pending)
        with self.assertRaises(ValueError):PurposeFeedback(float('nan'),0)

    def test_future_planner_cannot_reintroduce_expired_wait_root(self):
        c=scene();fc=copy.deepcopy(c);fc['actions']=[action('idle-plan',effect(.8)),action('work-plan',effect(.1))]
        f=JointForecast((fc,),{'idle-plan':('wait',),'work-plan':('work',)},{'idle-plan':.8,'work-plan':.1},3)
        ds,audit=select([c],f,root_allowed=[{'work'}])
        self.assertEqual(ds[0]['action_id'],'work');self.assertEqual(audit['progress_excluded'],1)
        f=replace(f,contexts=({**fc,'actions':fc['actions'][:1]},),roots={'idle-plan':('wait',)},purpose={'idle-plan':.8})
        ds,audit=select([c],f,root_allowed=[{'work'}]);self.assertIsNone(ds)

    def test_joint_rejection_falls_back_to_a_purposeful_reflex(self):
        c=scene();other=copy.deepcopy(c);other['scope']['npc']='other'
        fc=copy.deepcopy(c);fc['actions']=[action('all-wait',effect(.8))]
        future=copy.deepcopy(fc);future['scope']=other['scope']
        f=JointForecast((fc,future),{'all-wait':('wait','wait')},{'all-wait':.8},3)
        loops=[DecisionLoop(x,progress=ProgressWatch(x['scope'])) for x in (c,other)]
        for t in range(4):
            for x in (c,other):x['tick']=t
            # Supply contexts with this same actual tick/state to the forecast.
            def planner(cs,ds):
                forecasts=[]
                for x in cs:
                    y=copy.deepcopy(x);y['actions']=copy.deepcopy(fc['actions']);forecasts.append(y)
                return replace(f,contexts=tuple(forecasts))
            rs=DecisionLoop.decide_batch([(l,req(x,purpose())) for l,x in zip(loops,(c,other))],planner=planner)
            if t==3:
                self.assertTrue(all(r['decision']['action_id']=='work' for r in rs))
                self.assertTrue(all(not r['deliberation']['adopted'] for r in rs))
            for l,r in zip(loops,rs):l.abandon(r['ticket'],purpose_feedback=PurposeFeedback(0,0))

    def test_resource_production_wait_has_real_progress_and_release_condition(self):
        from reflex.resource_world import World,step,legal
        from reflex.route_experiment import routed_context
        from reflex.progress_adapters import economy_request,economy_feedback
        from reflex.laboratory import profiles
        w=World.start(routes=('culture',));w=replace(w,turn=3,empires=w.empires[:3]+(replace(w.empires[3],culture=2,buildings=(1,1,1,0,0,1)),))
        c,_=routed_context(w,profiles()[1],92,0,None,'culture');p=economy_request(w,c)
        self.assertEqual(p.activities['wait'].kind,'wait');self.assertEqual(p.activities['wait'].patience,3)
        monitor=ProgressWatch(c['scope'])
        for t in range(3):
            c,_=routed_context(w,profiles()[1],92,t,None,'culture');p=economy_request(w,c)
            self.assertIn('wait',monitor.mask(c,p)[0]);before=w;after=step(w,'wait')
            monitor.observe(t,'wait',p,economy_feedback(before,after,3,'wait'))
            w=replace(after,turn=3)
        self.assertEqual(w.empires[3].culture,8)
        after=step(w,'wait')
        self.assertFalse(economy_feedback(w,after,3,'wait').maintained)
        c,_=routed_context(w,profiles()[1],92,3,None,'culture')
        self.assertEqual(economy_request(w,c).activities['wait'].kind,'idle')

    def test_capture_requires_valid_stationary_turns(self):
        from reflex.combat import Battle,Unit,resolve
        from reflex.progress_adapters import combat_request,combat_feedback
        from reflex.combat import make_context
        from reflex.laboratory import profiles
        w=Battle((Unit(0,4,2),Unit(1,8,4)),goal='secure')
        c=make_context(w,0,profiles()[1],92,'secure',survival_security=True);watch=ProgressWatch(c['scope'])
        for t in range(3):
            c=make_context(w,0,profiles()[1],92,'secure',survival_security=True);p=combat_request(w,0,c)
            self.assertIn('guard',watch.mask(c,p)[0]);before=w;w,_=resolve(w,{0:'guard',1:'guard'},92)
            watch.observe(t,'guard',p,combat_feedback(before,w,0,'guard'))
        self.assertTrue(w.secured[0]);self.assertEqual(watch.stalls,0)

    def test_progress_constraint_precedes_same_effect_cost_pruning(self):
        c=scene();c['actions']=[action('wait',effect(.1)),action('work',effect(.1,cost=.1))]
        for a in c['actions']:a['confidence']=1.
        p=purpose(kind='idle');loop=DecisionLoop(c,progress=ProgressWatch(c['scope']))
        for t in range(3):
            c['tick']=t;r=loop.decide(req(c,p));loop.abandon(r['ticket'],purpose_feedback=PurposeFeedback(0,0))
        from reflex.core import FEATURES
        c['tick']=3
        request=replace(req(c,p),exact={a['id']:FEATURES for a in c['actions']})
        self.assertEqual(loop.decide(request)['decision']['action_id'],'work')

    def test_team_preparation_supports_wait_but_enemy_reload_does_not(self):
        from reflex.combat import Battle,Unit,resolve
        from reflex.progress_adapters import combat_levels,combat_feedback
        w=Battle((Unit(0,0,0),Unit(0,1,0,ammo=0),Unit(1,8,4,ammo=0)),goal='eliminate')
        after,_=resolve(w,{0:'guard',1:'reload',2:'guard'},92)
        self.assertTrue(combat_feedback(w,after,0,'guard').maintained)
        self.assertGreater(combat_levels(after,0)[1],combat_levels(w,0)[1])
        after,_=resolve(w,{0:'guard',1:'guard',2:'reload'},92)
        self.assertFalse(combat_feedback(w,after,0,'guard').maintained)
        self.assertEqual(combat_levels(after,0),combat_levels(w,0))

    def test_config_and_activity_contracts(self):
        for args in ({'grace':True},{'floor':0},{'repeat_limit':17}):
            with self.assertRaises(ValueError):ProgressConfig(**args)
        with self.assertRaises(ValueError):Activity('wait','predicted-opportunity')
        with self.assertRaises(ValueError):PurposeRequest(True,0,{})

    def test_recovery_requires_material_future_gain_and_preserves_strategic_wait(self):
        for gain,expected in ((-.10,'wait'),(.03,'wait'),(.12,'work')):
            with self.subTest(gain=gain):
                c=scene();loop=DecisionLoop(c,progress=ProgressWatch(c['scope'],ProgressConfig(proof_margin=.05)))
                for t in range(3):
                    c['tick']=t;r=loop.decide(req(c,purpose()));loop.abandon(r['ticket'],purpose_feedback=PurposeFeedback(0,0))
                def planner(cs,ds):
                    fc=copy.deepcopy(cs[0]);fc['actions']=[action('waiting-plan',effect(.2,values={'security':.3})),action('recovery-plan',effect(.1,values={'security':-.1}))]
                    return JointForecast((fc,),{'waiting-plan':('wait',),'recovery-plan':('work',)},
                        {'waiting-plan':0.,'recovery-plan':gain},3)
                c['tick']=3;r=DecisionLoop.decide_batch([(loop,req(c,purpose()))],planner=planner)[0]
                self.assertEqual(r['decision']['action_id'],expected)
                self.assertEqual(r['progress']['recovery']['approved'],expected=='work')
                self.assertEqual(r['progress']['unresolved'],expected=='wait')
                self.assertEqual(loop.progress.peak,0)
                loop.abandon(r['ticket'],purpose_feedback=PurposeFeedback(0,0))
                self.assertEqual(loop.memory.entries,{})

    def test_required_recovery_proof_without_planner_does_not_force_motion(self):
        c=scene();loop=DecisionLoop(c,progress=ProgressWatch(c['scope'],ProgressConfig(proof_margin=.15)))
        for t in range(5):
            c['tick']=t;r=loop.decide(req(c,purpose()))
            self.assertEqual(r['decision']['action_id'],'wait')
            if t>=3:self.assertTrue(r['progress']['unresolved'])
            loop.abandon(r['ticket'],purpose_feedback=PurposeFeedback(0,0))

    def test_invalid_recovery_evidence_does_not_advance_loop(self):
        c=scene();w=ProgressWatch(c['scope'],ProgressConfig(proof_margin=.15));p=purpose()
        for t in range(3):w.observe(t,'wait',p,PurposeFeedback(0,0))
        saved=w.record()
        with self.assertRaises(ValueError):w.mask(c,p,dict(needed=True,base_purpose=0,candidate_purpose=float('nan')))
        self.assertEqual(w.record(),saved)
        self.assertEqual(saved,ProgressWatch.from_record(c['scope'],json.loads(json.dumps(saved))).record())

    def test_joint_recovery_obeys_the_strictest_required_margin_atomically(self):
        cs=[scene(),scene()];cs[1]['scope']['npc']='other'
        loops=[DecisionLoop(c,progress=ProgressWatch(c['scope'],ProgressConfig(proof_margin=m))) for c,m in zip(cs,(.05,.15))]
        for t in range(3):
            for c,l in zip(cs,loops):
                c['tick']=t;r=l.decide(req(c,purpose()));l.abandon(r['ticket'],purpose_feedback=PurposeFeedback(0,0))
        def planner(contexts,ds):
            future=[]
            for c in contexts:
                f=copy.deepcopy(c);f['actions']=[action('waiting-plan',effect(.2,values={'security':.3})),action('recovery-plan',effect(.1,values={'security':-.1}))];future.append(f)
            return JointForecast(tuple(future),{'waiting-plan':('wait','wait'),'recovery-plan':('work','work')},
                {'waiting-plan':0.,'recovery-plan':.12},3)
        for c in cs:c['tick']=3
        rs=DecisionLoop.decide_batch([(l,req(c,purpose())) for l,c in zip(loops,cs)],planner=planner)
        self.assertTrue(all(r['decision']['action_id']=='wait' and not r['progress']['applied'] and r['progress']['unresolved'] for r in rs))

    def test_preparing_an_alternative_victory_is_not_hidden_by_a_leading_route(self):
        from reflex.resource_world import World,step
        from reflex.route_experiment import routed_context
        from reflex.progress_adapters import economy_request,economy_feedback
        from reflex.laboratory import profiles
        for seat in range(4):
            with self.subTest(seat=seat):
                w=World.start(routes=('culture','power'));roster=list(w.empires)
                roster[seat]=replace(roster[seat],stock=(12,12,12,12),culture=2,land=3,army=6,buildings=(3,1,1,3,0,1))
                w=replace(w,turn=seat,empires=tuple(roster))
                c,_=routed_context(w,profiles()[1],92,0,None,'culture');watch=ProgressWatch(c['scope']);last_observed=None
                for t in range(3):
                    c,_=routed_context(w,profiles()[1],92,t,None,'culture');p=economy_request(w,c,last_observed)
                    self.assertIn('wait',watch.mask(c,p)[0])
                    before=w;after=step(w,'wait');watch.observe(t,'wait',p,economy_feedback(before,after,seat,'wait'))
                    w=after;last_observed=after
                    while w.turn!=seat:w=step(w,'wait')
                self.assertEqual(w.empires[seat].culture,8)

    def test_default_tactical_recovery_can_change_a_plan_inside_the_persona_corridor(self):
        from reflex.combat_planning import TacticalControl
        control=TacticalControl();c=scene()
        loop=DecisionLoop(c,progress=ProgressWatch(c['scope'],ProgressConfig(proof_margin=control.recovery_margin)))
        for t in range(3):
            c['tick']=t;r=loop.decide(req(c,purpose()));loop.abandon(r['ticket'],purpose_feedback=PurposeFeedback(0,0))
        def planner(cs,ds):
            fc=copy.deepcopy(cs[0]);fc['actions']=[action('waiting-plan',effect(.2,values={'security':.3})),action('recovery-plan',effect(.1,values={'security':-.1}))]
            return JointForecast((fc,),{'waiting-plan':('wait',),'recovery-plan':('work',)},
                {'waiting-plan':0.,'recovery-plan':.03},3,control.max_regret)
        c['tick']=3;r=DecisionLoop.decide_batch([(loop,req(c,purpose()))],planner=planner)[0]
        self.assertEqual(r['decision']['action_id'],'work')
        self.assertTrue(r['progress']['recovery']['approved'])

    def test_newly_observed_preparation_can_renew_a_wait_but_prediction_cannot(self):
        c=scene();w=ProgressWatch(c['scope']);p=purpose()
        for t in range(3):w.observe(t,'wait',p,PurposeFeedback(0,0))
        self.assertNotIn('wait',w.mask(c,p)[0])
        actual=replace(p,maintained=True)
        self.assertIn('wait',w.mask(c,actual)[0]);w.observe(3,'wait',actual)
        self.assertEqual(w.wait_spent,0);self.assertEqual(w.peak,0)
        for t in range(4,7):w.observe(t,'wait',p,PurposeFeedback(0,0))
        self.assertNotIn('wait',w.mask(c,p)[0])
        with self.assertRaises(ValueError):replace(p,maintained='predicted')


if __name__=='__main__':unittest.main()
