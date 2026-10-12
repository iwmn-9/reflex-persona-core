import copy
from dataclasses import replace
import unittest
import numpy as np
from reflex.core import Policy,compile_batch,character_key,NEEDS,FEATURES,TRAITS
from reflex.examples import context,action,effect
from reflex.intertemporal import Branch,forecast
from reflex.purpose_plan import Goal,goal_forecast
from reflex.deliberation import select
from reflex.valuation import ReturnFrame
from reflex.laboratory import PROFILES
from reflex.scaling import TiledPolicy


def settled(c,value,h):
    paths={k:(Branch(1.,(flow,)+(effect(),)*(h-1),(1.,)*h),) for k,flow in
        (('pledge',effect(values={'benevolence':value})),('profit',effect()))}
    audit={k:[dict(assessment=Goal('test','scored',g).record(),terminal=True,actions=[k])] for k,g in (('pledge',.4),('profit',.52))}
    return paths,audit


class ReturnFrameTests(unittest.TestCase):
    def test_same_settled_worlds_retain_choice_and_tier_under_zero_tail(self):
        for p in PROFILES:
            for v in (.04,.06,.08,.10,.12,.16,.20,.30):
                c=context('zero-tail',[action('pledge',effect()),action('profit',effect())],
                    needs={k:0. for k in NEEDS},values={'benevolence':1.},traits=dict(zip(TRAITS,p['traits'])),mode='principle')
                c['state'].update(primary_need=None,mode_urgency=0.)
                for policy in (Policy(principle_priority='finite'),Policy(),
                    Policy(principle_priority='personal',commitments={character_key(c):1.})):
                    choices=[];scores=[];tiers=[]
                    for h in (1,2,4,8,16):
                        paths,audit=settled(c,v,h)
                        f=goal_forecast(c,paths,audit,horizon=h,unit='own-decisions',target='test',policy=policy,settlement_weight='absolute')
                        b=compile_batch(f.contexts);d=policy.decide(b,False,return_frame=f.return_frame)
                        choices.append(select([c],f,policy)[0][0]['action_id']);scores.append(d.scores);tiers.append(d.eligible)
                    self.assertEqual(len(set(choices)),1)
                    for score,tier in zip(scores[1:],tiers[1:]):
                        np.testing.assert_allclose(score,scores[0],rtol=0,atol=1e-12)
                        np.testing.assert_array_equal(tier,tiers[0])

    def test_plain_returns_preserve_root_cost_familiarity_and_losses(self):
        c=context('returns',[action('pledge',effect(),confidence=.6,familiarity=.8,switch_cost=.2),action('profit',effect())],
            needs={k:0. for k in NEEDS},values={'benevolence':1.})
        rows=[]
        for h in (1,2,8,16):
            paths,_=settled(c,.2,h)
            paths['pledge']=(Branch(1.,(effect(-.4,values={'benevolence':.2},cost=.4),)+(effect(),)*(h-1),(1.,)*h),)
            f=forecast(c,paths,horizon=h,unit='own-decisions',target='flow')
            rows.append(Policy(principle_priority='finite').decide(compile_batch(f.contexts),False,return_frame=f.return_frame).scores)
        for row in rows[1:]:np.testing.assert_allclose(row,rows[0],rtol=0,atol=1e-12)

    def test_extra_real_flow_matters_and_transport_stays_bounded(self):
        c=context('repeated',[action('work',effect())],needs={k:0. for k in NEEDS})
        scores=[]
        for h in (1,4,16):
            paths={'work':(Branch(1.,(effect(.5,cost=.1),)*h,(1.,)*h),)}
            f=forecast(c,paths,horizon=h,unit='own-decisions',target='flow');b=compile_batch(f.contexts)
            self.assertLessEqual(abs(b.effects).max(),1.)
            scores.append(Policy().decide(b,False,return_frame=f.return_frame).scores[0,0])
        self.assertLess(scores[0],scores[1]);self.assertLess(scores[1],scores[2])

    def test_settlement_cannot_hide_extra_rewards_or_unknown_truncation(self):
        c=context('invalid-tail',[action('pledge',effect()),action('profit',effect())])
        paths,audit=settled(c,.2,2)
        paths['pledge']=(Branch(1.,(effect(),effect(.5)),(1.,1.)),)
        with self.assertRaisesRegex(ValueError,'settlement'):
            goal_forecast(c,paths,audit,horizon=2,unit='steps',target='test')
        paths,audit=settled(c,.2,2);audit['pledge'][0].update(assessment=Goal('test','running',.2).record(),terminal=False)
        with self.assertRaisesRegex(ValueError,'truncated'):
            goal_forecast(c,paths,audit,horizon=2,unit='steps',target='test')

    def test_return_frame_does_not_bypass_owner_or_goal(self):
        c=context('frame',[action('work',effect())]);frame=ReturnFrame.flows([c],2.,'steps')
        for key in ('objective','seed'):
            changed=copy.deepcopy(c);changed[key]='another' if key=='objective' else 43
            with self.assertRaisesRegex(ValueError,'owner'):Policy().decide(compile_batch([changed]),False,return_frame=frame)
        with self.assertRaises(ValueError):replace(frame,effects=[frame.effects[0]])
        with self.assertRaises(ValueError):replace(frame,root_terms=(True,))
        with self.assertRaises(ValueError):ReturnFrame.flows([c],17.,'steps')

    def test_tiled_return_scoring_is_identical_and_input_is_immutable(self):
        cs=[context(str(i),[action('a',effect(.2)),action('b',effect(-.1))]) for i in range(7)]
        b=compile_batch(cs);frame=ReturnFrame.flows(cs,4.,'steps');saved=b.effects.copy()
        a=Policy().decide(b,False,return_frame=frame);z=TiledPolicy(3).decide(b,False,return_frame=frame)
        np.testing.assert_array_equal(a.action,z.action);np.testing.assert_array_equal(a.scores,z.scores)
        np.testing.assert_array_equal(saved,b.effects)

    def test_immediate_policy_is_unchanged_without_a_frame(self):
        c=context('one-step',[action('work',effect(.3,cost=.1)),action('rest',effect(.1))])
        b=compile_batch([c]);a=Policy().decide(b,False);z=Policy().decide(b,False,return_frame=ReturnFrame.flows([c],1.,'steps'))
        np.testing.assert_array_equal(a.action,z.action);np.testing.assert_array_equal(a.scores,z.scores)

    def test_tiled_social_appraisal_with_return_units_retains_all_policy_arguments(self):
        cs=[context(str(i),[action('a',effect(.2)),action('b',effect(.19))]) for i in range(7)]
        b=compile_batch(cs);frame=ReturnFrame.flows(cs,4.,'steps')
        bias=np.tile([-.1,.1],(7,1));purpose=np.tile([.5,.45],(7,1))
        for stochastic in (False,True):
            a=Policy().decide(b,stochastic,return_frame=frame,appraisal=bias,purpose=purpose,max_social_regret=.04)
            z=TiledPolicy(3).decide(b,stochastic,return_frame=frame,appraisal=bias,purpose=purpose,max_social_regret=.04)
            np.testing.assert_array_equal(a.action,z.action);np.testing.assert_array_equal(a.scores,z.scores)
            np.testing.assert_array_equal(a.eligible,z.eligible)
        with self.assertRaises(ValueError):TiledPolicy(3).decide(b,appraisal=np.zeros((8,2)))
