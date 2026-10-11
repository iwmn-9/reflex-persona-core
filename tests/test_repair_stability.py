import copy
from dataclasses import replace
import unittest
from reflex.core import Policy,digest
from reflex.examples import context,action,effect
from reflex.deliberation import JointForecast,select
from reflex.plan_continuity import PlanIntention
from reflex.continuation_search import search_forecast
from test_plan_repair import corridor


def pair():
    c=context('generic-check',[action('x',effect()),action('y',effect())],needs={'physiology':0.,'safety':0.})
    def make(ids,values):
        cc=copy.deepcopy(c);cc['actions']=[action(k,effect(.9 if k=='b' else .6)) for k in ids]
        endpoint={k:[dict(target='same-goal',value=v,positive_confidence=1.,status='running',settlement_step=2) for v in values[k]] for k in ids}
        plans={k:dict(proposal='candidate-'+k,root='x' if k=='a' else 'y') for k in ids}
        proposals={'candidate-'+k:dict(root=plans[k]['root'],schedule=['x'],branches=[dict(seed=s,actions=[plans[k]['root'],'x']) for s in (30,31)]) for k in ids}
        return JointForecast((cc,),{k:(plans[k]['root'],) for k in ids},
            {k:sum(values[k])/2 for k in ids},2,1.,dict(unit='ticks',endpoint=endpoint,plans=plans,
                continuation_search=dict(proposals=proposals)),target='same-goal')
    reference=make(('a',),{'a':(.6,.6)})
    expanded=make(('a','b'),{'a':(.6,.6),'b':(1.,.3)})
    return c,reference,expanded


class RepairStabilityTests(unittest.TestCase):
    def test_higher_average_does_not_hide_a_worse_paired_branch(self):
        c,reference,f=pair();saved=digest(c)
        self.assertEqual(select([c],f)[0][0]['action_id'],'y')
        d,a=select([c],replace(f,stability_reference=reference))
        self.assertEqual(d[0]['action_id'],'x');self.assertEqual(a['selected_plan'],'a')
        guard=a['metadata']['repair_stability']
        self.assertFalse(guard['accepted']);self.assertTrue(guard['protected'])
        self.assertEqual(guard['branch_deltas'],[.4,-.3]);self.assertEqual(saved,digest(c))
        m=PlanIntention(c);m.remember(c,f,a['selected_plan'],d[0]['action_id'])
        self.assertEqual(m.remaining,('x',))

    def test_improvement_on_every_branch_is_allowed(self):
        c,r,f=pair();f.audit['endpoint']['b'][1]['value']=.8
        d,a=select([c],replace(f,stability_reference=r))
        self.assertEqual(d[0]['action_id'],'y');self.assertTrue(a['metadata']['repair_stability']['accepted'])

    def test_actual_root_mask_is_used_by_both_votes(self):
        c,r,f=pair();d,a=select([c],replace(f,stability_reference=r),root_allowed=[{'x'}])
        self.assertEqual(d[0]['action_id'],'x');self.assertEqual(a['metadata']['repair_stability']['branch_deltas'],[0.,0.])

    def test_misaligned_model_bank_is_rejected(self):
        c,r,f=pair();f.audit['continuation_search']['proposals']['candidate-b']['branches'][0]['seed']=99
        with self.assertRaises(ValueError):select([c],replace(f,stability_reference=r))

    def test_protection_does_not_override_an_expanded_value_tier(self):
        c,r,f=pair();cc=copy.deepcopy(f.contexts[0]);cc['actions']=[cc['actions'][1]]
        f=replace(f,contexts=(cc,),roots={'b':('y',)},purpose={'b':.65},stability_reference=r)
        d,a=select([c],f)
        self.assertEqual(d[0]['action_id'],'y');self.assertFalse(a['metadata']['repair_stability']['protected'])

    def test_reference_is_selected_without_repair_bank_and_defaults_unchanged(self):
        c,args=corridor();old=('start','left','left','left','finish')
        plain=search_forecast(c,(0,False),retained=old,repairs=24,**args)
        guarded=search_forecast(c,(0,False),retained=old,repairs=24,repair_guard=True,**args)
        self.assertIsNone(plain.stability_reference)
        self.assertFalse(any(k.startswith('proposal-repair-') for k in guarded.stability_reference.audit['continuation_search']['proposals']))
        d,a=select([c],guarded)
        self.assertEqual(a['selected_purpose'],1.);self.assertTrue(a['metadata']['repair_stability']['accepted'])
        self.assertEqual(d[0]['action_id'],'start')

    def test_nonrecursive_owner_reference_and_controls(self):
        c,r,f=pair()
        with self.assertRaises(ValueError):select([c],replace(f,stability_reference=replace(r,stability_reference=r)))
        c,args=corridor()
        for value in (1,'yes'):
            with self.assertRaises(ValueError):search_forecast(c,(0,False),repairs=24,repair_guard=value,**args)
        with self.assertRaises(ValueError):search_forecast(c,(0,False),repair_guard=True,**args)


if __name__=='__main__':unittest.main()
