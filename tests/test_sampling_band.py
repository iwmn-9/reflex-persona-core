import copy
import math
import unittest
from dataclasses import replace
from reflex.sampling_band import PurposeSampling
from reflex.deliberation import JointForecast,select
from reflex.examples import context,action,effect
from reflex.closed_loop import owned_rollout
from reflex.receding_owner import RecedingOwner
from reflex.core import character_key
from reflex.laboratory import PROFILES
from tools.continuing_games import ContinuingGame,GAMES


class SamplingBandTests(unittest.TestCase):
    def test_common_world_noise_cancels_but_uncertain_difference_is_retained(self):
        ids=(1,2,3)
        stable=PurposeSampling(ids,(('a',(-.5,0.,.5)),('b',(-.4,.1,.6))),'same-model')
        mask,audit=stable.mask(('a','b'),(0.,.1),(True,True),.015)
        self.assertEqual(mask,[False,True])
        self.assertLess(audit['excluded']['a']['paired_standard_error'],1e-15)
        noisy=PurposeSampling(ids,(('a',(0.,0.,0.)),('b',(-.5,.1,.7))),'same-model')
        self.assertEqual(noisy.mask(('a','b'),(0.,.1),(True,True),.015)[0],[True,True])
        # A progress-blocked option cannot exclude the remaining legal choice.
        self.assertEqual(stable.mask(('a','b'),(0.,.1),(True,False),.015)[0],[True,False])

    def test_aligned_sample_contract_and_bounded_projection(self):
        for args in (((1,), (('a',(0.,)),)),((1,1),(('a',(0.,0.)),)),
                     ((1,2),(('a',(0.,)),)),((1,2),(('a',(0.,float('nan'))),)),
                     ((1,2),(('a',(0.,0.)),('a',(0.,0.))))):
            with self.assertRaises(ValueError):PurposeSampling(*args,'model')
        s=PurposeSampling((1,2),(('a',(1.1,1.3)),),'model')
        self.assertEqual(s.mask(('a',),(1.,),(True,),0.)[0],[True])
        with self.assertRaisesRegex(ValueError,'mean'):s.mask(('a',),(.9,),(True,),0.)
        with self.assertRaisesRegex(ValueError,'matching'):s.mask(('b',),(1.,),(True,),0.)

    def test_uncertainty_does_not_erase_persona_flows_or_mutate_inputs(self):
        c=context('sample-selection',[action('hold',effect()),action('act',effect())],mode='need')
        fc=copy.deepcopy(c)
        fc['actions']=[action('p-hold',effect(.2)),action('p-act',effect(-.2,cost=.2))]
        f=JointForecast((fc,),{'p-hold':('hold',),'p-act':('act',)}, {'p-hold':0.,'p-act':.1},4,.015)
        self.assertEqual(select([c],f)[0][0]['action_id'],'act')
        s=PurposeSampling((1,2,3),(('p-hold',(0.,0.,0.)),('p-act',(-.5,.1,.7))),'model')
        candidate=replace(f,purpose_sampling=s);saved=copy.deepcopy((c,candidate))
        decisions,audit=select([c],candidate)
        self.assertEqual(decisions[0]['action_id'],'hold')
        self.assertEqual((c,candidate),saved)
        self.assertEqual(audit['purpose_sampling']['samples'],3)
        stable=replace(s,samples=(('p-hold',(0.,0.,0.)),('p-act',(.1,.1,.1))))
        self.assertEqual(select([c],replace(f,purpose_sampling=stable))[0][0]['action_id'],'act')

    def test_full_principle_tier_is_retained_before_sampling_ambiguity(self):
        from tests.test_purpose_plan import fixture
        from reflex.purpose_plan import goal_forecast
        from reflex.intertemporal import Branch
        c,paths,audit=fixture();c['values']['benevolence']=1.
        c['state'].update(mode='principle',primary_need='belonging',mode_urgency=.1)
        paths['prepare']=(Branch(1.,(effect(values={'benevolence':.8}),effect()),(1.,1.)),)
        f=goal_forecast(c,paths,audit,horizon=2,unit='ticks',target='mission',max_regret=0.)
        self.assertEqual(len(f.roots),1)
        s=PurposeSampling((1,2,3),tuple((k,(q,q,q)) for k,q in f.purpose.items()),'model')
        self.assertEqual(select([c],replace(f,purpose_sampling=s))[0][0]['action_id'],'prepare')

    def test_closed_loop_samples_are_paired_by_seed_and_survive_actual_feedback(self):
        for game in GAMES:
            g=ContinuingGame(game,PROFILES[1],turns=4);w=g.start();c=g.context(w)
            own=g.actor(c,learning='segmented');axes=character_key(c)
            args=dict(observe=g.context,events=g.events,terminal=g.terminal,
                assess=lambda w:g.goal(w).record(),record=lambda w:w,request_for=g.request,
                feedback=g.feedback,model_id=g.model_id,target=g.target,tail_steps=3,
                seeds=(701,702,703),world_model='posterior',purpose_estimator='conditional',tail_recovery=2)
            loop=RecedingOwner(own,**args,sampling_standard_errors=2.)
            for symbol in ('2','0','1','2'):
                c=g.context(w,own.state);before=own.record();request=g.request(c);cap={}
                ctrl=owned_rollout(own,**args,sampling_standard_errors=2.)
                f=ctrl.planner(own,request,w,captured=cap)([c],[])
                self.assertEqual(own.record(),before)
                self.assertEqual(f.purpose_sampling.sample_ids,(701,702,703))
                for key,values in f.purpose_sampling.samples:
                    self.assertAlmostEqual(max(-1.,min(1.,math.fsum(values)/3)),f.purpose[key],places=12)
                result=loop.decide(w)
                self.assertEqual(result['decision']['action_id'],cap['node']['action'])
                key=result['decision']['action_id'];after,flow=g.transition(w,key,symbol)
                loop.resolve(result['ticket'],g.feedback(w,after,key,c,flow));w=after
                self.assertEqual(character_key(g.context(w,own.state)),axes)
            self.assertTrue(g.terminal(w))

    def test_immediate_settlement_has_zero_mc_error_and_bad_sampling_modes_reject(self):
        for game in GAMES:
            g=ContinuingGame(game,PROFILES[0],turns=1);w=g.start();c=g.context(w);own=g.actor(c,learning='segmented')
            args=dict(observe=g.context,events=g.events,terminal=g.terminal,
                assess=lambda w:g.goal(w).record(),record=lambda w:w,request_for=g.request,
                feedback=g.feedback,model_id=g.model_id,target=g.target,tail_steps=3,
                seeds=(701,702,703),world_model='posterior',purpose_estimator='conditional',tail_recovery=2)
            normal=RecedingOwner(copy.deepcopy(own),**args).decide(w)
            candidate=RecedingOwner(copy.deepcopy(own),**args,sampling_standard_errors=2.).decide(w)
            self.assertEqual(normal['decision'],candidate['decision'])
            for override in (dict(sampling='stratified'),dict(purpose_estimator='sampled'),dict(seeds=(701,)),dict(tree_steps=2)):
                with self.assertRaises(ValueError):owned_rollout(own,**{**args,**override},sampling_standard_errors=2.)

    def test_diagnostic_prefixes_and_checkpoint_keep_sampling_configuration(self):
        import json,tempfile
        from pathlib import Path
        from tools.run_owned_value import run
        from tools.diagnose_owner_horizon import diagnose,prefix_choice
        job=['combat',1,52992,'outside',3,1,2,0]
        row=run(job,'receding_recovery',sampling_standard_errors=2.)
        before=json.dumps(row,sort_keys=True)
        d=diagnose(row,0,2)
        self.assertEqual(d['sampling_standard_errors'],2.)
        self.assertEqual(d['selected_action'],row['trace'][0]['action'])
        self.assertEqual(prefix_choice(row,0,d,3)['action'],d['selected_action'])
        self.assertEqual(prefix_choice(row,0,d,2)['action'],diagnose(row,0,1)['selected_action'])
        self.assertEqual(json.dumps(row,sort_keys=True),before)
        with tempfile.TemporaryDirectory() as directory:
            cp=Path(directory)/'checkpoint.json'
            self.assertIsNone(run(job,'receding_recovery',checkpoint=cp,max_decisions=1,sampling_standard_errors=2.))
            with self.assertRaisesRegex(ValueError,'contract changed'):run(job,'receding_recovery',checkpoint=cp,sampling_standard_errors=1.)
            restored=run(job,'receding_recovery',checkpoint=cp,sampling_standard_errors=2.)
            restored.pop('seconds');row.pop('seconds');self.assertEqual(restored,row)


if __name__=='__main__':unittest.main()
