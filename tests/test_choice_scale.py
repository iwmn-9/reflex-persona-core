import copy
from dataclasses import replace
import unittest
import numpy as np
from reflex.core import Policy,NEEDS,compile_batch
from reflex.examples import context,action,effect
from reflex.intertemporal import Branch
from reflex.purpose_plan import Goal,goal_forecast
from reflex.deliberation import select
from reflex.forecast_control import ForecastBudget


def risk_fixture(offset=0.,unit=1.,horizon=1):
    c=context('score-origin',[action('safe',effect()),action('risk',effect())],needs={k:0. for k in NEEDS})
    paths={k:tuple(Branch(p,(effect(),)*horizon,(1.,)*horizon) for p,v in values)
           for k,values in {'safe':[(1.,.08)],'risk':[(.5,-.1),(.5,.3)]}.items()}
    audit={k:[dict(terminal=True,actions=[k],assessment=Goal('mission','scored',offset+unit*v).record()) for p,v in values]
           for k,values in {'safe':[(1.,.08)],'risk':[(.5,-.1),(.5,.3)]}.items()}
    return c,paths,audit


def evaluate(c,paths,audit,**kwargs):
    return goal_forecast(c,paths,audit,horizon=len(next(iter(paths.values()))[0].effects),unit='ticks',target='mission',
                         settlement_weight='absolute',policy=Policy(principle_priority='finite'),**kwargs)


class ChoiceScaleTests(unittest.TestCase):
    def test_existing_absolute_loss_depends_on_common_score_origin(self):
        decisions=[]
        for offset in (0.,.5):
            c,paths,audit=risk_fixture(offset);f=evaluate(c,paths,audit)
            decisions.append(select([c],f,Policy(principle_priority='finite'))[0][0]['action_id'])
        self.assertEqual(decisions,['safe','risk'])

    def test_gain_origin_preserves_risk_and_all_numeric_scores_under_translation(self):
        rows=[]
        for offset in (0.,.5):
            c,paths,audit=risk_fixture(offset);saved=copy.deepcopy((c,paths,audit))
            f=evaluate(c,paths,audit,objective_origin=Goal('mission','running',offset))
            d=Policy(principle_priority='finite').decide(compile_batch(f.contexts),False,return_frame=f.return_frame)
            self.assertEqual(select([c],f,Policy(principle_priority='finite'))[0][0]['action_id'],'safe')
            self.assertEqual((c,paths,audit),saved);rows.append(d.scores)
        np.testing.assert_allclose(rows[0],rows[1],rtol=0.,atol=1e-14)

    def test_declared_unit_conversion_preserves_gain_scores_and_opportunity_choice(self):
        rows=[];bands=[]
        for unit,offset in ((1.,0.),(.5,.2),(.125,.7)):
            c,paths,audit=risk_fixture(offset,unit)
            f=evaluate(c,paths,audit,objective_origin=Goal('mission','running',offset),objective_unit=unit,opportunity_regret=True)
            rows.append(Policy(principle_priority='finite').decide(compile_batch(f.contexts),False,return_frame=f.return_frame).scores)
            bands.append(f.max_regret/unit)
            self.assertEqual(select([c],f,Policy(principle_priority='finite'))[0][0]['action_id'],'risk')
        for row in rows[1:]:np.testing.assert_allclose(rows[0],row,rtol=0.,atol=1e-14)
        np.testing.assert_allclose(bands,[.003]*3,rtol=0.,atol=1e-14)

    def test_absorbing_padding_does_not_invent_gains(self):
        rows=[]
        for horizon in (1,4,16):
            c,paths,audit=risk_fixture(.5,horizon=horizon)
            f=evaluate(c,paths,audit,objective_origin=Goal('mission','running',.5))
            rows.append(Policy(principle_priority='finite').decide(compile_batch(f.contexts),False,return_frame=f.return_frame).scores)
        for row in rows[1:]:np.testing.assert_allclose(rows[0],row,rtol=0.,atol=1e-14)

    def test_known_losses_do_not_disappear_under_zero_positive_confidence(self):
        c,paths,audit=risk_fixture(.5);paths={k:tuple(replace(b,confidence=(0.,)) for b in bs) for k,bs in paths.items()}
        f=evaluate(c,paths,audit,objective_origin=Goal('mission','running',.5))
        a=next(a for a in f.contexts[0]['actions'] if f.roots[a['id']]==('risk',))
        self.assertAlmostEqual(a['outcomes'][0]['objective'],-.05)
        self.assertEqual(a['outcomes'][1]['objective'],0.)

    def test_opportunity_band_uses_supported_personality_tier(self):
        c=context('support',[action(k,effect()) for k in ('care','goal','excluded')],needs={k:0. for k in NEEDS},values={'benevolence':1.},mode='principle')
        paths={k:(Branch(1.,(effect(values={'benevolence':v}),),(1.,)),) for k,v in [('care',.6),('goal',0.),('excluded',.9)]}
        audit={k:[dict(terminal=True,actions=[k],assessment=Goal('mission','scored',v).record())] for k,v in [('care',.1),('goal',.11),('excluded',-1.)]}
        p=Policy(principle_priority='finite')
        f=evaluate(c,paths,audit,opportunity_regret=True,competence_support={'care','goal'})
        self.assertAlmostEqual(f.max_regret,.0015)
        self.assertEqual(select([c],f,p,[{'care','goal'}])[0][0]['action_id'],'goal')
        rigid=goal_forecast(c,paths,audit,horizon=1,unit='ticks',target='mission',policy=Policy(),opportunity_regret=True)
        self.assertEqual(select([c],rigid)[0][0]['action_id'],'excluded')

    def test_fraction_still_allows_personal_cost_within_the_opportunity(self):
        c=context('personal-cost',[action(k,effect()) for k in ('none','principle','best')],needs={k:0. for k in NEEDS},values={'benevolence':1.},mode='principle')
        paths={k:(Branch(1.,(effect(values={'benevolence':v}),),(1.,)),) for k,v in [('none',0.),('principle',.4),('best',0.)]}
        audit={k:[dict(terminal=True,actions=[k],assessment=Goal('mission','scored',v).record())] for k,v in [('none',0.),('principle',.9),('best',1.)]}
        f=evaluate(c,paths,audit,opportunity_regret=True)
        self.assertEqual(select([c],f,Policy(principle_priority='finite'))[0][0]['action_id'],'principle')

    def test_opportunity_fraction_erases_a_small_deterministic_two_option_tradeoff(self):
        c=context('two-options',[action('care',effect()),action('best',effect())],needs={k:0. for k in NEEDS},values={'benevolence':1.},mode='principle')
        paths={k:(Branch(1.,(effect(values={'benevolence':v}),),(1.,)),) for k,v in [('care',.4),('best',0.)]}
        audit={k:[dict(terminal=True,actions=[k],assessment=Goal('mission','scored',v).record())] for k,v in [('care',.1),('best',.11)]}
        p=Policy(principle_priority='finite')
        self.assertEqual(select([c],evaluate(c,paths,audit),p)[0][0]['action_id'],'care')
        self.assertEqual(select([c],evaluate(c,paths,audit,opportunity_regret=True),p)[0][0]['action_id'],'best')

    def test_invalid_units_target_or_fraction_are_rejected(self):
        c,paths,audit=risk_fixture()
        for kw in (dict(objective_unit=.01),dict(objective_unit=True),dict(objective_origin=Goal('other','running',0.)),
                   dict(opportunity_regret=1),dict(opportunity_regret=True,max_regret=1.1)):
            with self.assertRaises(ValueError):evaluate(c,paths,audit,**kw)
        for kw in (dict(objective_gain=1),dict(objective_unit=.01),dict(opportunity_regret=True,max_regret=1.1)):
            with self.assertRaises(ValueError):ForecastBudget(**kw)

    def test_uncertain_gains_do_not_discount_known_stock_or_double_root_support(self):
        c,paths,audit=risk_fixture(.5)
        next(a for a in c['actions'] if a['id']=='risk')['confidence']=.5
        paths['risk']=tuple(replace(b,confidence=(.5,)) for b in paths['risk'])
        f=evaluate(c,paths,audit,objective_origin=Goal('mission','running',.5))
        key=next(k for k,v in f.roots.items() if v==('risk',))
        self.assertAlmostEqual(f.purpose[key],.525)
        row=next(a for a in f.contexts[0]['actions'] if a['id']==key)
        b=compile_batch(f.contexts);scales,_=f.return_frame.numeric(b)
        # The transported +.3 gain receives .5 support once: +.15, not +.075.
        self.assertAlmostEqual(row['outcomes'][1]['objective']*scales[0,0]*row['confidence'],.15)

    def test_full_decision_and_corridor_respect_declared_unit_even_with_uncertainty(self):
        frames=[]
        for unit,offset in ((1.,0.),(.5,.2),(.125,.7)):
            c,paths,audit=risk_fixture(offset,unit)
            next(a for a in c['actions'] if a['id']=='risk')['confidence']=.5
            f=evaluate(c,paths,audit,objective_origin=Goal('mission','running',offset),objective_unit=unit)
            decisions,audit_row=select([c],f,Policy(principle_priority='finite'))
            frames.append((decisions[0]['action_id'],{f.roots[k][0]:(v-offset)/unit for k,v in f.purpose.items()},f.max_regret/unit))
        self.assertEqual([r[0] for r in frames],['safe']*3)
        for r in frames:
            self.assertAlmostEqual(r[1]['risk'],.025);self.assertAlmostEqual(r[1]['safe'],.08);self.assertAlmostEqual(r[2],.15)

    def test_unit_conversion_keeps_both_allowed_and_excluded_personal_costs(self):
        for loss,expected in ((.1,'care'),(.2,'best')):
            for unit,offset in ((1.,0.),(.5,.1),(.125,.3)):
                c=context('unit-band',[action('care',effect()),action('best',effect())],needs={k:0. for k in NEEDS},values={'benevolence':1.},mode='principle')
                paths={k:(Branch(1.,(effect(values={'benevolence':v}),),(1.,)),) for k,v in [('care',.4),('best',0.)]}
                audit={k:[dict(terminal=True,actions=[k],assessment=Goal('mission','scored',offset+unit*v).record())] for k,v in [('care',.6-loss),('best',.6)]}
                f=evaluate(c,paths,audit,objective_origin=Goal('mission','running',offset),objective_unit=unit)
                self.assertEqual(select([c],f,Policy(principle_priority='finite'))[0][0]['action_id'],expected)


if __name__=='__main__':unittest.main()
