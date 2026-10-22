import copy
from dataclasses import replace
import unittest
import numpy as np
from reflex.core import Policy, FEATURES, NEEDS, compile_batch
from reflex.examples import context,action,effect
from reflex.intertemporal import Branch
from reflex.cumulative_preference import CumulativePreference
from reflex.purpose_plan import Goal,goal_forecast
from reflex.deliberation import select
from reflex.return_means import ReturnMeans


class ReturnMeanTests(unittest.TestCase):
    def fixture(self):
        c=context('means',[action(k,effect()) for k in ('a','b')],needs={k:0. for k in NEEDS},
            values={'achievement':1.},mode='principle')
        c['state'].update(primary_need=None,mode_urgency=0.)
        paths={k:tuple(Branch(.5,(effect(),effect()),(1.,1.)) for _ in range(2)) for k in ('a','b')}
        rows={k:[dict(terminal=False,actions=[k,'next'],assessment=Goal('score','running',v).record(),purpose_mean_estimate=mean)
            for v in values] for k,values,mean in [('a',(-.1,.3),.4),('b',(.25,.35),.3)]}
        kwargs=dict(horizon=2,unit='decisions',target='score',objective_origin=Goal('score','running',0.),
            cumulative_preference=CumulativePreference('score','achievement',(('values.achievement',1.),)))
        return c,paths,rows,kwargs

    def test_corrected_value_mean_retains_better_plan_before_principle_filter(self):
        c,paths,rows,args=self.fixture();unchanged=copy.deepcopy((c,paths,rows))
        old=goal_forecast(c,paths,rows,**args)
        new=goal_forecast(c,paths,rows,**args,coherent_cumulative_mean=True)
        self.assertEqual(select([c],old)[0][0]['action_id'],'b')
        self.assertEqual(select([c],new)[0][0]['action_id'],'a')
        self.assertEqual((c,paths,rows),unchanged)
        self.assertAlmostEqual(new.purpose[next(iter(new.roots))],.4)
        self.assertEqual(new.contexts[0]['personality'],c['personality'])
        self.assertEqual(new.contexts[0]['values'],c['values'])

    def test_mean_and_utility_correct_but_sampled_downside_and_other_channels_do_not(self):
        c,paths,rows,args=self.fixture();p=Policy(principle_priority='finite')
        f=goal_forecast(c,paths,rows,**args,policy=p,coherent_cumulative_mean=True)
        b=compile_batch(f.contexts);before={};after={}
        a=p.decide(b,False,return_frame=f.return_frame,outcome_audit=before)
        z=p.decide(b,False,return_frame=f.return_frame,return_means=f.return_means,outcome_audit=after)
        correction=f.return_means.numeric(b);i=FEATURES.index('achievement')
        np.testing.assert_array_equal(a.features[:,:,3],z.features[:,:,3]) # downside loss
        np.testing.assert_allclose(z.features[:,:,2]-a.features[:,:,2],correction[:,:,i],atol=1e-15)
        for j in (0,1,4,5,6,7):np.testing.assert_array_equal(a.features[:,:,j],z.features[:,:,j])
        self.assertGreater(z.scores[0,0],a.scores[0,0])
        np.testing.assert_allclose(np.einsum('nak,nak->na',b.probability,after['scores']),z.scores,atol=1e-14)
        # Candidate order is data, never positional attachment of means.
        flipped=copy.deepcopy(f.contexts[0]);flipped['actions'].reverse()
        rb=compile_batch([flipped])
        np.testing.assert_array_equal(f.return_means.numeric(rb),correction)

    def test_correction_requires_matching_owner_plans_units_and_support(self):
        c,paths,rows,args=self.fixture();f=goal_forecast(c,paths,rows,**args,coherent_cumulative_mean=True)
        b=compile_batch(f.contexts);p=Policy()
        with self.assertRaises(ValueError):p.decide(b,False,return_means=f.return_means)
        with self.assertRaises(ValueError):p.decide(b,False,return_frame=f.return_frame,
            return_means=replace(f.return_means,owners=('0'*64,)))
        with self.assertRaises(ValueError):p.decide(replace(b,confidence=b.confidence*.9),False,
            return_frame=f.return_frame,return_means=f.return_means)
        with self.assertRaises(ValueError):ReturnMeans(f.return_means.owners,('wrong',),f.return_means.corrections).numeric(b)
        bad=copy.deepcopy(f.return_means.record());bad['corrections'][0][0][0]=float('nan')
        with self.assertRaises(ValueError):ReturnMeans(tuple(bad['owners']),tuple(bad['plans']),
            tuple(tuple(tuple(r) for r in owner) for owner in bad['corrections']))

    def test_no_conditional_correction_preserves_old_decision(self):
        c,paths,rows,args=self.fixture()
        for traces in rows.values():
            for row in traces:row.pop('purpose_mean_estimate')
        a=goal_forecast(c,paths,rows,**args);b=goal_forecast(c,paths,rows,**args,coherent_cumulative_mean=True)
        self.assertEqual(select([c],a)[0],select([c],b)[0])
        self.assertTrue(np.all(b.return_means.numeric(compile_batch(b.contexts))==0))

    def test_tiled_scoring_and_plan_restriction_keep_the_same_owned_correction(self):
        from reflex.scaling import TiledPolicy
        from reflex.valuation import ReturnFrame
        c,paths,rows,args=self.fixture();p=Policy(principle_priority='finite')
        f=goal_forecast(c,paths,rows,**args,policy=p,coherent_cumulative_mean=True)
        contexts=(f.contexts[0],)*3;b=compile_batch(contexts)
        frame=ReturnFrame(f.return_frame.owners*3,f.return_frame.effects*3,f.return_frame.root_terms*3,f.return_frame.unit)
        means=ReturnMeans(f.return_means.owners*3,f.return_means.plans,f.return_means.corrections*3)
        plain=p.decide(b,False,return_frame=frame,return_means=means)
        tiled=TiledPolicy(1,principle_priority='finite').decide(b,False,return_frame=frame,return_means=means)
        np.testing.assert_array_equal(plain.scores,tiled.scores)
        kept=next(iter(f.roots));cc=copy.deepcopy(f.contexts[0]);cc['actions']=[a for a in cc['actions'] if a['id']==kept]
        restricted=replace(f,contexts=(cc,),roots={kept:f.roots[kept]},purpose={kept:f.purpose[kept]})
        self.assertEqual(select([c],restricted,p)[0][0]['action_id'],f.roots[kept][0])


if __name__=='__main__':unittest.main()
