import copy
from dataclasses import replace
import unittest
import numpy as np
from reflex.core import Policy,NEEDS,FEATURES,compile_batch
from reflex.examples import context,action,effect
from reflex.intertemporal import Branch
from reflex.purpose_plan import Goal,goal_forecast
from reflex.deliberation import select
from reflex.cumulative_preference import CumulativePreference


def fixture():
    c=context('cumulative',[action(k,effect()) for k in ('split','lump')],
        needs={k:0. for k in NEEDS},values={'achievement':1.},mode='principle',
        traits={'conscientiousness':1.,'neuroticism':0.})
    c['state'].update(primary_need=None,mode_urgency=0.)
    paths={'split':(Branch(1.,(effect(values={'achievement':.4}),)*7,(1.,)*7),),
        'lump':(Branch(1.,(effect(values={'achievement':-1.}),)+(effect(values={'achievement':.2}),)*5+
                       (effect(values={'achievement':1.}),),(1.,)*7),)}
    audit={k:[dict(terminal=True,actions=[k]*7,assessment=Goal('score','scored',v).record())]
           for k,v in (('split',.28),('lump',.35))}
    return c,paths,audit


def build(c,paths,audit,basis=None):
    return goal_forecast(c,paths,audit,horizon=len(next(iter(paths.values()))[0].effects),
        unit='decisions',target='score',objective_origin=Goal('score','running',0.),
        settlement_weight='absolute',cumulative_preference=basis)


class CumulativePreferenceTests(unittest.TestCase):
    def test_recover_lump_return_without_changing_owner_weights_or_input(self):
        c,paths,audit=fixture();saved=copy.deepcopy((c,paths,audit))
        basis=CumulativePreference('score','declared-accomplishment',(('values.achievement',10.),))
        old=build(c,paths,audit);new=build(c,paths,audit,basis)
        self.assertEqual(select([c],old)[0][0]['action_id'],'split')
        self.assertEqual(select([c],new)[0][0]['action_id'],'lump')
        self.assertEqual((c,paths,audit),saved)
        self.assertEqual(new.contexts[0]['personality'],c['personality'])
        self.assertEqual(new.contexts[0]['values'],c['values'])
        self.assertEqual(new.audit['cumulative_preference']['identity'],basis.identity)
        self.assertIn('cumulative value channels',new.audit['subjective_settlement_semantics'])
        self.assertNotIn('cumulative value channels',old.audit['subjective_settlement_semantics'])
        b=compile_batch(new.contexts);scales,_=new.return_frame.numeric(b)
        self.assertLessEqual(abs(b.effects).max(),1.)
        i=FEATURES.index('achievement')
        self.assertAlmostEqual(b.effects[0,0,0,i]*scales[0,i],3.5)

    def test_unlisted_intrinsic_value_can_still_override_objective(self):
        c,paths,audit=fixture();c['values'].update(benevolence=1.,achievement=.1)
        paths['split']=(Branch(1.,(effect(values={'benevolence':1.}),)+(effect(),)*6,(1.,)*7),)
        basis=CumulativePreference('score','declared-accomplishment',(('values.achievement',10.),))
        f=build(c,paths,audit,basis)
        self.assertEqual(select([c],f)[0][0]['action_id'],'split')

    def test_distribution_and_intrinsic_flows_survive_and_mean_correction_is_not_a_new_reward(self):
        c=context('correlated',[action('risk',effect())],needs={k:0. for k in NEEDS})
        paths={'risk':(Branch(.25,(effect(-.2,needs={'safety':-.3},values={'benevolence':-.4},cost=.2),effect()),(1.,1.)),
                       Branch(.75,(effect(.3,needs={'safety':.1},values={'benevolence':.5},style={'openness':.4}),effect()),(1.,1.)))}
        audit={'risk':[dict(terminal=False,actions=['risk','next'],assessment=Goal('score','running',v).record(),purpose_mean_estimate=.8)
                       for v in (-.2,.3)]}
        basis=CumulativePreference('score','declared-accomplishment',(('values.achievement',5.),))
        old=build(c,paths,audit);new=build(c,paths,audit,basis)
        a=compile_batch(old.contexts);b=compile_batch(new.contexts)
        left,_=old.return_frame.numeric(a);right,_=new.return_frame.numeric(b)
        i=FEATURES.index('achievement');keep=[j for j in range(len(FEATURES)) if j!=i]
        np.testing.assert_array_equal(a.probability,b.probability)
        np.testing.assert_allclose((a.effects*left[:,None,None,:])[...,keep],(b.effects*right[:,None,None,:])[...,keep],rtol=0,atol=0)
        np.testing.assert_allclose((b.effects*right[:,None,None,:])[0,0,:,i],[-1.,1.5],rtol=0,atol=1e-14)
        self.assertEqual(old.purpose,new.purpose)

    def test_bounded_packet_preserves_large_losses_and_known_origin(self):
        basis=CumulativePreference('score','declared-accomplishment',(('values.achievement',256.),))
        c=context('origin',[action('loss',effect())],needs={k:0. for k in NEEDS})
        paths={'loss':(Branch(1.,(effect(),),(1.,)),)}
        audit={'loss':[dict(terminal=True,actions=['loss'],assessment=Goal('score','scored',-1.).record())]}
        f=goal_forecast(c,paths,audit,horizon=1,unit='decisions',target='score',
            objective_origin=Goal('score','running',1.),cumulative_preference=basis)
        b=compile_batch(f.contexts);scales,_=f.return_frame.numeric(b);i=FEATURES.index('achievement')
        self.assertEqual(b.effects[0,0,0,i],-1.);self.assertEqual(scales[0,i],512.)

    def test_explicit_meaning_and_supported_probability_contract_are_required(self):
        for channels in ((),(('needs.growth',1.),),(('values.achievement',0.),),
                         (('values.achievement',True),),(('values.achievement',257.),),
                         (('values.achievement',1.),('values.achievement',2.))):
            with self.assertRaises(ValueError):CumulativePreference('score','unit',channels)
        c,paths,audit=fixture();basis=CumulativePreference('score','unit',(('values.achievement',10.),))
        with self.assertRaisesRegex(ValueError,'factual origin'):
            goal_forecast(c,paths,audit,horizon=7,unit='decisions',target='score',cumulative_preference=basis)
        with self.assertRaises(ValueError):build(c,paths,audit,replace(basis,target='different'))
        paths['lump']=(replace(paths['lump'][0],confidence=(.5,)*7),)
        with self.assertRaisesRegex(ValueError,'branch probabilities'):build(c,paths,audit,basis)

    def test_same_cumulative_contract_runs_with_actual_feedback_in_every_rule_family(self):
        from reflex.core import character_key
        from reflex.decision_loop import DecisionLoop
        from reflex.receding_owner import RecedingOwner
        from reflex.closed_loop import owned_rollout
        from reflex.laboratory import PROFILES
        from tools.continuing_games import ContinuingGame,GAMES
        for game in GAMES:
            g=ContinuingGame(game,PROFILES[0],3);w=g.start();c=g.context(w)
            owner=g.actor(c,learning='segmented');fixed=character_key(c)
            basis=CumulativePreference(g.target,'per-decision-score',(('values.achievement',3.),))
            args=dict(observe=g.context,events=g.events,terminal=g.terminal,assess=lambda w:g.goal(w).record(),
                record=lambda w:w,request_for=g.request,feedback=g.feedback,model_id=g.model_id,target=g.target,
                tail_steps=2,world_model='posterior',purpose_estimator='conditional',tail_recovery=2,
                cumulative_preference=basis)
            for symbol in ('0','2','1'):
                c=g.context(w,owner.state);before=owner.record();captured={}
                ctrl=owned_rollout(owner,**args)
                f=ctrl.planner(owner,g.request(c),w,captured=captured)([c],[])
                self.assertEqual(owner.record(),before)
                self.assertEqual(f.audit['cumulative_preference']['identity'],basis.identity)
                live=RecedingOwner(owner,**args);result=live.decide(w);key=result['decision']['action_id']
                self.assertEqual(key,captured['node']['action'])
                after,flow=g.transition(w,key,symbol);live.resolve(result['ticket'],g.feedback(w,after,key,c,flow))
                self.assertEqual(character_key(g.context(after,owner.state)),fixed)
                saved=owner.record();owner=DecisionLoop.from_record(c,saved,predictor=owner.predictor)
                self.assertEqual(saved,owner.record());w=after
            self.assertTrue(g.terminal(w))

    def test_future_recovery_uses_same_semantics_and_cache_identity_changes(self):
        from unittest.mock import patch
        from reflex.closed_loop import owned_rollout
        from reflex.receding_owner import RecedingOwner
        from reflex.laboratory import PROFILES
        from tools.continuing_games import ContinuingGame
        g=ContinuingGame('combat',PROFILES[0],3);w=g.start();w['ammo']=0
        c=g.context(w);owner=g.actor(c,learning='segmented')
        basis=CumulativePreference(g.target,'per-decision-score',(('values.achievement',3.),))
        args=dict(observe=g.context,events=g.events,terminal=g.terminal,assess=lambda w:g.goal(w).record(),
            record=lambda w:w,request_for=g.request,feedback=g.feedback,model_id=g.model_id,target=g.target,
            tail_steps=2,world_model='posterior',purpose_estimator='conditional',tail_recovery=2)
        old=owned_rollout(owner,**args);new=owned_rollout(owner,**args,cumulative_preference=basis)
        self.assertNotEqual(old.model_id,new.model_id);self.assertNotEqual(old.frontier.identity,new.frontier.identity)
        with patch('reflex.closed_loop.owned_rollout',wraps=owned_rollout) as calls:
            RecedingOwner(owner,**args,cumulative_preference=basis).decide(w)
        recoveries=[call.kwargs for call in calls.call_args_list if call.kwargs.get('tree_steps')==2]
        self.assertTrue(recoveries)
        self.assertTrue(all(kw['cumulative_preference']==basis for kw in recoveries))

    def test_checkpoint_and_read_only_prefix_keep_the_cumulative_contract(self):
        import json,tempfile
        from pathlib import Path
        from tools.run_owned_value import run
        from tools.diagnose_owner_horizon import diagnose,prefix_choice
        from tools.audit_continuation_returns import audit_run
        job=['gambling',0,52998,'shift-return',3,1,2,0]
        row=run(job,'receding_recovery',cumulative_achievement=True)
        before=json.dumps(row,sort_keys=True);d=diagnose(row,0,2)
        self.assertTrue(d['cumulative_achievement'])
        self.assertEqual(d['selected_action'],row['trace'][0]['action'])
        self.assertEqual(prefix_choice(row,0,d,3)['action'],d['selected_action'])
        self.assertTrue(row['trace'][0]['cumulative_preference']['identity'])
        wrong=copy.deepcopy(d);wrong.pop('cumulative_achievement')
        with self.assertRaisesRegex(ValueError,'cumulative preference'):prefix_choice(row,0,wrong,3)
        audit_run(row,ticks=(0,))
        self.assertEqual(json.dumps(row,sort_keys=True),before)
        with tempfile.TemporaryDirectory() as directory:
            cp=Path(directory)/'checkpoint.json'
            self.assertIsNone(run(job,'receding_recovery',cumulative_achievement=True,checkpoint=cp,max_decisions=1))
            with self.assertRaisesRegex(ValueError,'contract changed'):run(job,'receding_recovery',checkpoint=cp)
            restored=run(job,'receding_recovery',cumulative_achievement=True,checkpoint=cp)
            restored.pop('seconds');row.pop('seconds');self.assertEqual(restored,row)


if __name__=='__main__':unittest.main()
