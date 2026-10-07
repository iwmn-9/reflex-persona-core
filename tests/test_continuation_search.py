import copy
import unittest
from reflex.core import digest
from reflex.examples import context,action,effect
from reflex.delivery_world import DeliveryProbe
from reflex.laboratory import profiles
from reflex.flow_rollout import rollout
from reflex.purpose_plan import Goal,goal_forecast
from reflex.continuation_search import search_forecast
from reflex.deliberation import select
from reflex.intertemporal import Branch


class ContinuationSearchTests(unittest.TestCase):
    def test_competing_proposals_preserve_principle_and_real_intent(self):
        c=context('values',[action('help',effect()),action('take',effect())],
            needs={'physiology':0.,'safety':0.},values={'benevolence':1.},mode='principle')
        c['state'].update(primary_need='belonging',mode_urgency=.1)
        paths={'help-short':(Branch(1.,(effect(values={'benevolence':.6}),),(1.,)),),
               'help-strong':(Branch(1.,(effect(values={'benevolence':.9}),),(1.,)),),
               'take-best':(Branch(1.,(effect(),),(1.,)),)}
        roots={'help-short':'help','help-strong':'help','take-best':'take'}
        audit={k:[dict(terminal=True,actions=[roots[k]],assessment=Goal('finish',
            'success' if roots[k]=='take' else 'failure',1. if roots[k]=='take' else -1.).record())] for k in paths}
        f=goal_forecast(c,paths,audit,horizon=1,unit='turns',target='finish',max_regret=0.,plan_roots=roots)
        chosen,info=select([c],f)
        self.assertEqual(chosen[0]['action_id'],'help')
        self.assertEqual(chosen[0]['next_state']['intent_action'],'help')
        self.assertEqual(len(f.roots),1)
        self.assertEqual(f.audit['plans'][info['selected_plan']]['proposal'],'help-strong')

    def test_depth_zero_preserves_reflex_forecast_and_owner(self):
        p=DeliveryProbe(dict(genre='delivery',limit=8,capacity=2,supply=3),profiles()[0])
        w=p.start();c=p.observe(w);saved=digest([c,w.__dict__])
        paths,audit=rollout(c,w,observe=p.observe,advance=p.advance,terminal=p.terminal,
            assess=lambda s:p.goal(s).record(),horizon=6)
        old=goal_forecast(c,paths,audit,horizon=6,unit='turns',target=p.target)
        new=search_forecast(c,w,observe=p.observe,advance=p.advance,terminal=p.terminal,
            assess=lambda s:p.goal(s).record(),horizon=6,depth=0,unit='turns',target=p.target)
        self.assertEqual(select([c],old)[0][0],select([c],new)[0][0])
        self.assertEqual(saved,digest([c,w.__dict__]))

    def test_shared_schedule_cannot_choose_after_seeing_seed(self):
        # A seed-clairvoyant left/right choice would score 1. Both realizable
        # shared plans score 0: one wins and one loses with probability 1/2.
        c=context('blind',[action('start',effect())],needs={'physiology':0.,'safety':0.})
        def observe(s,m):
            cc=copy.deepcopy(c);cc['tick']=s[0];cc['state']=m
            cc['actions']=[action(k,effect()) for k in ('left','right')];return cc
        def advance(s,k,seed):return (s[0]+1,seed,k),effect()
        def assess(s):return Goal('finish','success' if (s[1]==0)==(s[2]=='left') else 'failure',
            1. if (s[1]==0)==(s[2]=='left') else -1.).record()
        f=search_forecast(c,(0,None,None),observe=observe,advance=advance,terminal=lambda s:s[0]>=2,
            assess=assess,horizon=2,seeds=(0,1),depth=1,width=2,target='finish')
        self.assertTrue(all(v==0. for v in f.purpose.values()))
        for p in f.audit['continuation_search']['proposals'].values():
            if p['schedule']:self.assertEqual(p['branches'][0]['actions'],p['branches'][1]['actions'])

    def test_unavailable_schedule_uses_observed_reflex_and_legal_steps(self):
        p=DeliveryProbe(dict(genre='delivery',limit=5),profiles()[0]);w=p.start();c=p.observe(w)
        paths,audit=rollout(c,w,observe=p.observe,advance=p.advance,terminal=p.terminal,
            horizon=4,roots=('rest',),schedule=('not-present',),record_choices=True)
        self.assertEqual(audit['rest'][0]['schedule_fallbacks'],[1])
        for k in audit['rest'][0]['actions']:w,_=p.advance(w,k,0)
        self.assertEqual(set(paths),{'rest'})

    def test_complete_root_coverage_and_capacity_are_enforced(self):
        p=DeliveryProbe(dict(genre='delivery',limit=4),profiles()[1]);w=p.start();c=p.observe(w)
        paths,audit=rollout(c,w,observe=p.observe,advance=p.advance,terminal=p.terminal,
            assess=lambda s:p.goal(s).record(),horizon=3)
        with self.assertRaises(ValueError):goal_forecast(c,{'p':paths['rest']},{'p':audit['rest']},
            horizon=3,unit='turns',target=p.target,plan_roots={'p':'rest'})
        huge=copy.deepcopy(c);huge['actions']=[action(f'a-{i}',effect()) for i in range(90)]
        with self.assertRaises(ValueError):search_forecast(huge,w,observe=p.observe,advance=p.advance,
            terminal=p.terminal,assess=lambda s:p.goal(s).record(),horizon=3,target=p.target)

    def test_terminal_padding_never_executes_a_schedule_after_settlement(self):
        p=DeliveryProbe(dict(genre='delivery',limit=1),profiles()[0]);w=p.start();c=p.observe(w)
        f=search_forecast(c,w,observe=p.observe,advance=p.advance,terminal=p.terminal,
            assess=lambda s:p.goal(s).record(),horizon=4,depth=2,target=p.target)
        for n in f.audit['continuation_search']['proposals'].values():
            self.assertEqual(len(n['branches'][0]['actions']),1)
            self.assertEqual(n['schedule'],[])


if __name__=='__main__':unittest.main()
