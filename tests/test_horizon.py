import copy
from dataclasses import replace
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.examples import context,action,effect
from reflex.horizon import purpose_return,compare_paths
from reflex.progress import Activity,PurposeRequest,PurposeFeedback,ProgressWatch,ProgressConfig
from reflex.decision_loop import DecisionLoop,Request
from reflex.deliberation import JointForecast
from reflex.judgment import Binding


def fixture():
    c=context('extra-options',[action('wait',effect(.2,values={'security':.3})),action('work',effect(.1,values={'security':-.1}))],values={'security':.8},mode='principle')
    p=PurposeRequest(0.,0.,{'wait':Activity('wait','opening',patience=3,release='after-preparation'),'work':Activity('attempt','advance')})
    def req():return Request(c,{k:Binding(k,'public',()) for k in p.activities},{k:() for k in p.activities},purpose=p)
    loop=DecisionLoop(c,progress=ProgressWatch(c['scope'],ProgressConfig(proof_margin=.02)))
    for t in range(3):
        c['tick']=t;r=loop.decide(req());loop.abandon(r['ticket'],purpose_feedback=PurposeFeedback(0,0))
    c['tick']=3
    return c,loop,req


class HorizonTests(unittest.TestCase):
    def test_equal_endpoint_prefers_earlier_actual_purpose_only_when_enabled(self):
        early=[.3,.6,1.];later=[0.,0.,1.]
        self.assertEqual(purpose_return(early),purpose_return(later))
        self.assertGreater(purpose_return(early,.3),purpose_return(later,.3))
        self.assertGreater(purpose_return([.8,.8,1.],.3),purpose_return([.8,.8,-1.],.3))

    def test_temporal_scale_rejects_invalid_values(self):
        for p,w in (([],0),([True],0),([float('nan')],0),([1.1],0),([0],True),([0],-1)):
            with self.assertRaises(ValueError):purpose_return(p,w)

    def test_audit_matches_observation_units_and_censors_missing_future(self):
        r=compare_paths([0.,.2,.3],[(0,[.1,.5,.8],True)])
        self.assertEqual(r['immediate_samples'],1);self.assertEqual(r['delayed_samples'],0)
        self.assertEqual(compare_paths([0.,.2,1.],[(0,[0.,.5,1.],True)],True)['unrealized_later_progress'],0)
        missed=compare_paths([0.,0.,0.],[(0,[0.,.5,.8],True)],True)
        self.assertEqual(missed['unrealized_later_progress'],1)

    def test_extra_candidate_hook_preserves_persona_and_immediate_memory(self):
        c,loop,req=fixture();fixed=copy.deepcopy((c['personality'],c['values']));calls=[]
        def planner(cs,ds):
            fc=copy.deepcopy(cs[0]);fc['actions']=[action('old',effect(.2,values={'security':.3}))]
            return JointForecast((fc,),{'old':('wait',)},{'old':0.},3)
        def recover(cs,ds,allowed):
            calls.append(allowed);fc=copy.deepcopy(cs[0]);fc['actions']=[action('extra',effect(.6,values={'security':-.1}))]
            return JointForecast((fc,),{'extra':('work',)},{'extra':.6},3)
        planner.recover=recover
        result=DecisionLoop.decide_batch([(loop,req())],planner=planner)[0]
        self.assertEqual(result['decision']['action_id'],'work');self.assertEqual(calls,[[{'work'}]])
        self.assertEqual((c['personality'],c['values']),fixed)
        self.assertEqual(loop.pending['prior'][0]['objective'],.1)
        loop.abandon(result['ticket'],purpose_feedback=PurposeFeedback(.1,0.))
        self.assertFalse(loop.memory.entries)

    def test_extra_forecast_cannot_change_target_or_horizon_transactionally(self):
        for changed in ({'target':'different-target'},{'horizon':4},{'max_regret':.2}):
            c,loop,req=fixture();saved=loop.record()
            def planner(cs,ds):
                f=copy.deepcopy(cs[0]);f['actions']=[action('old',effect(.2))]
                return JointForecast((f,),{'old':('wait',)},{'old':0.},3)
            def recover(cs,ds,allowed):return replace(planner(cs,ds),**changed)
            planner.recover=recover
            with self.assertRaises(ValueError):DecisionLoop.decide_batch([(loop,req())],planner=planner)
            self.assertEqual(loop.record(),saved)

    def test_expansion_cannot_force_a_nonimproving_choice(self):
        c,loop,req=fixture()
        def planner(cs,ds):
            f=copy.deepcopy(cs[0]);f['actions']=[action('old',effect(.2))]
            return JointForecast((f,),{'old':('wait',)},{'old':.5},3)
        def recover(cs,ds,allowed):
            f=copy.deepcopy(cs[0]);f['actions']=[action('extra',effect(.1))]
            return JointForecast((f,),{'extra':('work',)},{'extra':.4},3)
        planner.recover=recover
        r=DecisionLoop.decide_batch([(loop,req())],planner=planner)[0]
        self.assertEqual(r['decision']['action_id'],'wait');self.assertTrue(r['progress']['unresolved'])

    def test_supported_completions_keep_legal_conflict_free_options(self):
        from reflex.combat import Battle,make_context,legal
        from reflex.combat_planning import propose,TacticalControl,no_own_collision
        from reflex.core import Policy
        from reflex.laboratory import profiles
        w=Battle.start('choke','both');actors=(0,1,2);cs=[make_context(w,i,profiles()[1],160,'secure',survival_security=True) for i in actors]
        ds=[Policy().choose(c) for c in cs];permitted=[set(legal(w,i))-{'guard'} for i in actors]
        plans=propose(w,actors,cs,ds,TacticalControl(recovery_options=True),permitted)
        self.assertTrue(any(all(first[i] in permitted[j] for j,i in enumerate(actors)) for first,_,_ in plans))
        self.assertTrue(all(no_own_collision(first) and all(first[i] in legal(w,i) for i in actors) for first,_,_ in plans))

    def test_resource_paths_complete_rival_turns_and_have_equal_length(self):
        from reflex.resource_world import World
        from reflex.resource_planning import forecast,EconomyControl
        from reflex.route_experiment import routed_context
        from reflex.laboratory import profiles
        w=World.start(limit=2);c,_=routed_context(w,profiles()[0],160,0,None,'science')
        f=forecast(w,[c],[],EconomyControl(horizon=4,progress_weight=.3))
        for k,path in f.audit['purpose_paths'].items():
            self.assertEqual(len(path),4);self.assertEqual(path[-1],0.)
            self.assertAlmostEqual(f.purpose[k],purpose_return(path,.3))

    def test_successful_recovery_diagnostic_does_not_require_a_decline_reason(self):
        from reflex.combat import Battle,battle_record
        from reflex.recovery_experiment import path_audit
        w=Battle.start();row=dict(before=battle_record(w),after=battle_record(w),planning=dict(selected_path=None,recovery_attempt=dict(adopted=True,reason=None)))
        result=path_audit(dict(genre='combat',seed=160,trace=[row]))
        self.assertEqual(result['compatible_options_found'],1);self.assertEqual(result['compatible'],1)

    def test_trace_handles_no_living_npcs_before_opponent_completes_both_goals(self):
        from unittest.mock import patch
        from reflex.combat import resolve,Battle
        from reflex.combat_planning import TacticalControl
        from reflex.validation_experiment import combat
        from reflex.laboratory import profiles
        first=True
        def die(w,choices,seed):
            nonlocal first
            after,audit=resolve(w,choices,seed)
            if first:
                after=replace(after,units=tuple(replace(u,hp=0) if u.team==0 else u for u in after.units));first=False
            return after,audit
        start=Battle.start(limit=3,goal='both')
        with patch('reflex.combat.Battle.start',return_value=start),patch('reflex.combat.resolve',side_effect=die):
            r=combat(profiles()[0],170,'open','both','baseline',learn=True,survival_security=True,planner=TacticalControl(horizon=1,samples=1,max_plans=2,trace_execution=True),progress_watch=True)
        self.assertEqual(len(r['trace']),3)
        self.assertNotIn('planning',r['trace'][1]);self.assertEqual(r['trace'][1]['purpose'],[])


if __name__=='__main__':unittest.main()
