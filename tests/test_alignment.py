import copy
from dataclasses import replace
import json
import unittest
from unittest.mock import patch
from reflex.alignment import compare_execution


def step(index=0, before='s0', after='s1', own='guard', rival='move', terminal=False):
    return dict(index=index, before=before, after=after, self={'0':own}, opponent={'1':rival}, terminal=terminal)


def trace(steps=None, horizon=2):
    return dict(run='episode-1', group='team-0', target='purpose-v1', unit='tick', start=0, horizon=horizon,
                steps=steps if steps is not None else [step(),step(1,'s1','s2')])


class ExecutionAlignmentTests(unittest.TestCase):
    def test_separates_self_and_opponent_at_equal_public_states(self):
        for side in ('self','opponent'):
            p=trace();o=copy.deepcopy(p);o['steps'][1][side][next(iter(o['steps'][1][side]))]='different'
            r=compare_execution(p,o)
            self.assertEqual(r[side+'_differences'],1)
            self.assertEqual(r['opponent_differences' if side=='self' else 'self_differences'],0)
            self.assertEqual(r['first_'+side+'_difference'],1)
            self.assertEqual(r['state_matched_steps'],2)
        p=trace();o=copy.deepcopy(p);o['steps'][0]['self']['0']='x';o['steps'][0]['opponent']['1']='y'
        r=compare_execution(p,o)
        self.assertEqual((r['self_differences'],r['opponent_differences']),(1,1))

    def test_changed_state_is_not_attributed_to_changed_policies(self):
        p=trace();o=copy.deepcopy(p);o['steps'][0]['after']='other';o['steps'][1]['before']='other'
        o['steps'][1]['self']['0']='x';o['steps'][1]['opponent']['1']='y'
        r=compare_execution(p,o)
        self.assertEqual(r['same_inputs_successor_differences'],1)
        self.assertEqual(r['state_diverged_steps'],1)
        self.assertEqual((r['self_compared'],r['self_differences'],r['opponent_differences']),(1,0,0))
        self.assertEqual(r['steps'][1]['self'],'not-comparable')

    def test_identity_prevents_cross_run_actor_target_or_time_mixing(self):
        for key,value in [('run','other-run'),('group','team-1'),('target','other-purpose'),('unit','own-round'),('start',1),('horizon',3)]:
            p=trace();o=copy.deepcopy(p);o[key]=value
            with self.assertRaises(ValueError):compare_execution(p,o)
        self.assertEqual(compare_execution(trace(),trace())['self_differences'],0)

    def test_json_round_trip_does_not_change_action_owners(self):
        p=trace();o=copy.deepcopy(p)
        self.assertEqual(compare_execution(p,o),compare_execution(json.loads(json.dumps(p)),json.loads(json.dumps(o))))
        o['steps'][0]['self']={0:'guard'}
        with self.assertRaises(ValueError):compare_execution(p,o)

    def test_missing_actions_are_not_an_empty_team(self):
        p=trace();o=copy.deepcopy(p);o['steps'][0]['opponent']=None
        r=compare_execution(p,o)
        self.assertEqual(r['opponent_compared'],1);self.assertEqual(r['opponent_differences'],0)
        self.assertEqual(r['steps'][0]['opponent'],'unobserved')
        for t in (p,o):t['steps'][0]['self']={}
        self.assertEqual(compare_execution(p,o)['self_differences'],0)

    def test_missing_ticks_are_censored_not_compacted(self):
        p=trace();o=trace([None,step(1,'s1','s2')]);r=compare_execution(p,o)
        self.assertEqual(r['missing_observation_steps'],1);self.assertEqual(r['self_compared'],1)
        with self.assertRaises(ValueError):compare_execution(p,trace([step(1,'s1','s2')]))
        self.assertEqual(compare_execution(p,trace([]))['missing_observation_steps'],2)

    def test_omitted_observation_field_is_rejected_explicitly(self):
        for side in ('self','opponent'):
            o=trace();del o['steps'][0][side]
            with self.assertRaises(ValueError):compare_execution(trace(),o)

    def test_combat_adapter_rejects_postterminal_rows_and_unbounded_horizon(self):
        from reflex.combat import Battle,battle_record,resolve
        from reflex.alignment_experiment import audit_combat_execution
        w=Battle.start(limit=1);choices={i:'guard' for i in range(6)};after,_=resolve(w,choices,0)
        contract=dict(run='case',group='team-0',target='mission',unit='combat-tick')
        row=dict(before=battle_record(w),after=battle_record(after),choices=choices)
        run=dict(genre='combat',seed=0,execution_contract=contract,trace=[row])
        self.assertEqual(audit_combat_execution(run)['sampled_paths'],0)
        for offset in (1,3):
            broken=copy.deepcopy(run)
            before=replace(after,tick=offset);late=replace(after,tick=offset+1)
            broken['trace'].append(dict(before=battle_record(before),after=battle_record(late),choices=choices))
            with self.assertRaises(ValueError):audit_combat_execution(broken)
        for horizon in (True,0,17,1000000000):
            broken=copy.deepcopy(run)
            broken['trace'][0]['planning']=dict(adopted=True,execution=dict(**contract,start=0,horizon=horizon,samples=[]))
            with self.assertRaises(ValueError):audit_combat_execution(broken)

    def test_terminal_padding_does_not_invent_actions(self):
        p=trace([step(terminal=True)],horizon=3);r=compare_execution(p,copy.deepcopy(p))
        self.assertEqual(r['terminal_only_steps'],2);self.assertEqual(r['self_compared'],1)
        o=trace([step()],horizon=3);r=compare_execution(p,o)
        self.assertEqual(r['missing_observation_steps'],2);self.assertEqual(r['terminal_only_steps'],0)
        p=trace();o=trace([step(terminal=True)]);r=compare_execution(p,o)
        self.assertEqual(r['terminal_boundary_differences'],1)
        p,o=o,p;r=compare_execution(p,o)
        self.assertEqual(r['terminal_boundary_differences'],1)

    def test_invalid_and_discontinuous_traces_are_rejected(self):
        for edit in ('continuity','terminal','partial','overlap','index','flag'):
            p=trace()
            if edit=='continuity':p['steps'][1]['before']='broken'
            elif edit=='terminal':p['steps'][0]['terminal']=True
            elif edit=='partial':p['steps'].pop()
            elif edit=='overlap':p['steps'][0]['opponent']={'0':'guard'}
            elif edit=='index':p['steps'][0]['index']=False
            else:p['steps'][0]['terminal']=1
            with self.assertRaises(ValueError):compare_execution(p,trace())

    def test_comparator_is_pure(self):
        p=trace();o=trace();saved=copy.deepcopy((p,o));compare_execution(p,o)
        self.assertEqual((p,o),saved)

    def test_optional_tracing_does_not_change_predictions_or_public_inputs(self):
        from reflex.combat import Battle,make_context
        from reflex.combat_planning import forecast,TacticalControl
        from reflex.core import Policy
        from reflex.laboratory import profiles
        w=Battle.start('choke','eliminate',limit=2);actors=(0,1,2)
        cs=[make_context(w,i,profiles()[1],180,'eliminate',survival_security=True) for i in actors]
        saved=copy.deepcopy(cs);ds=[Policy().choose(c) for c in cs]
        control=TacticalControl(horizon=4,samples=2,max_plans=4)
        old=forecast(w,actors,cs,ds,control);new=forecast(w,actors,cs,ds,replace(control,trace_execution=True))
        self.assertEqual((old.roots,old.purpose,old.contexts),(new.roots,new.purpose,new.contexts))
        self.assertEqual(old.audit['nodes'],new.audit['nodes']);self.assertEqual(cs,saved)
        for key in old.roots:
            self.assertNotIn('execution_samples',old.audit['plans'][key])
            samples=new.audit['plans'][key]['execution_samples']
            self.assertEqual(len(samples),2)
            for sample in samples:
                self.assertEqual(len(sample),2);self.assertTrue(sample[-1]['terminal'])
                self.assertEqual(sample[0]['self'],{str(i):k for i,k in zip(actors,new.roots[key])})
        with self.assertRaises(ValueError):TacticalControl(trace_execution=1)

    def test_real_episode_trace_is_passive_and_root_is_the_committed_action(self):
        from reflex.combat import Battle
        from reflex.combat_planning import TacticalControl
        from reflex.validation_experiment import combat,replay,replay_progress
        from reflex.alignment_experiment import audit_combat_execution
        from reflex.laboratory import profiles
        profile=profiles()[1];fixed=copy.deepcopy(profile)
        control=TacticalControl(horizon=3,samples=2,max_plans=4,recovery_options=True)
        args=(profile,180,'choke','eliminate','baseline')
        kwargs=dict(learn=True,survival_security=True,rival='switch',progress_watch=True)
        with patch('reflex.combat.Battle.start',return_value=Battle.start('choke','eliminate',limit=3)):
            old=combat(*args,**kwargs,planner=control)
            new=combat(*args,**kwargs,planner=replace(control,trace_execution=True))
        self.assertEqual(profile,fixed)
        for a,b in zip(old['trace'],new['trace']):
            for key in ('before','after','choices','audit','purpose'):self.assertEqual(a[key],b[key])
        for key in ('won','lost','actions','learned_uses','regime_resets','decisions','planning_nodes'):
            self.assertEqual(old[key],new[key])
        self.assertNotIn('execution_contract',old)
        report=audit_combat_execution(new);self.assertGreater(report['sampled_paths'],0)
        for comparison in report['comparisons']:self.assertEqual(comparison['steps'][0]['self'],'same')
        self.assertEqual(report,audit_combat_execution(json.loads(json.dumps(new))))
        self.assertEqual(replay(new),3);self.assertGreater(replay_progress(new),0)
        wrong=copy.deepcopy(new);wrong['execution_contract']['run']='wrong-episode'
        with self.assertRaises(ValueError):audit_combat_execution(wrong)
        missing=copy.deepcopy(new);missing['trace'].pop(1)
        self.assertGreater(audit_combat_execution(missing)['totals']['missing_observation_steps'],0)


if __name__=='__main__':unittest.main()
