import copy
from dataclasses import replace
import unittest
from reflex.examples import context,action,effect
from reflex.intertemporal import Branch
from reflex.purpose_plan import Goal,goal_forecast,PersonaProgressWatch
from reflex.deliberation import select
from reflex.progress import Activity,PurposeRequest,PurposeFeedback,ProgressConfig
from reflex.purpose_recovery import GoalProbe,run
from reflex.laboratory import profiles


def fixture(horizon=2):
    paths={'prepare':(Branch(1.,(effect(-.1),)+(effect(),)*(horizon-1),(1.,)*horizon),),
           'execute':(Branch(1.,(effect(.1),)+(effect(),)*(horizon-1),(1.,)*horizon),)}
    c=context('goals',[action(k,v[0].effects[0]) for k,v in paths.items()],needs={'physiology':0.,'safety':0.})
    audit={k:[dict(terminal=True,actions=[k],assessment=Goal('mission','draw' if k=='prepare' else 'success',0. if k=='prepare' else 1.).record())] for k in paths}
    return c,paths,audit


class PurposePlanTests(unittest.TestCase):
    def test_draw_cannot_keep_unfinished_progress_and_numeric_contract(self):
        for args in (('goal','draw',.5),('goal','failure',0.),('goal','success',.9),('goal','running',True),('goal','scored',float('nan'))):
            with self.assertRaises(ValueError):Goal(*args)

    def test_endpoint_return_is_not_diluted_by_horizon_padding(self):
        for h in (2,6,16):
            c,paths,audit=fixture(h)
            f=goal_forecast(c,paths,audit,horizon=h,unit='ticks',target='mission')
            vals={f.roots[k][0]:v for k,v in f.purpose.items()}
            self.assertEqual(vals,{'execute':1.,'prepare':0.})
            self.assertEqual(select([c],f)[0][0]['action_id'],'execute')

    def test_endpoint_coverage_goal_and_terminal_mismatches_are_rejected(self):
        c,paths,audit=fixture()
        for bad in ({'execute':audit['execute']},{**audit,'prepare':[]},
                    {**audit,'execute':[dict(terminal=False,assessment=Goal('mission','success',1.).record())]},
                    {**audit,'execute':[dict(terminal=True,assessment=Goal('another','success',1.).record())]}):
            with self.assertRaises(ValueError):goal_forecast(c,paths,bad,horizon=2,unit='ticks',target='mission')

    def test_goal_corridor_cannot_overwrite_original_strongest_principle(self):
        c,paths,audit=fixture();c['values']['benevolence']=1.
        c['state'].update(mode='principle',primary_need='belonging',mode_urgency=.1)
        paths['prepare']=(Branch(1.,(effect(values={'benevolence':.8}),effect()),(1.,1.)),)
        f=goal_forecast(c,paths,audit,horizon=2,unit='ticks',target='mission',max_regret=0.)
        self.assertEqual(select([c],f)[0][0]['action_id'],'prepare')
        self.assertEqual(len(f.roots),1)

    def test_persona_flow_losses_and_costs_are_not_replaced_with_goal(self):
        c,paths,audit=fixture();paths['execute']=(Branch(1.,(effect(-.1,needs={'safety':-.6},cost=.4),effect()),(1.,1.)),)
        before=copy.deepcopy((c,paths,audit));f=goal_forecast(c,paths,audit,horizon=2,unit='ticks',target='mission')
        a=next(a for a in f.contexts[0]['actions'] if f.roots[a['id']]==('execute',))
        self.assertLess(a['outcomes'][0]['needs']['safety'],0.)
        self.assertGreater(a['outcomes'][0]['cost'],0.)
        self.assertEqual((c,paths,audit),before)

    def test_persona_patience_still_changes_current_vs_later_goal_tradeoff(self):
        c,paths,audit=fixture(3)
        audit['prepare']=[dict(terminal=True,actions=['prepare'],assessment=Goal('mission','scored',.6).record())]
        audit['execute']=[dict(terminal=True,actions=['execute','execute','execute'],assessment=Goal('mission','scored',1.).record())]
        c['personality'].update(conscientiousness=.95,neuroticism=.1)
        f=goal_forecast(c,paths,audit,horizon=3,unit='ticks',target='mission',max_regret=1.)
        self.assertEqual(select([c],f)[0][0]['action_id'],'execute')
        c['personality'].update(conscientiousness=.05,neuroticism=.9)
        f=goal_forecast(c,paths,audit,horizon=3,unit='ticks',target='mission',max_regret=1.)
        self.assertEqual(select([c],f)[0][0]['action_id'],'prepare')

    def test_low_future_confidence_shrinks_wins_without_erasing_losses(self):
        c,paths,audit=fixture()
        paths['execute']=(replace(paths['execute'][0],confidence=(1.,0.)),)
        audit['execute'][0]['actions']=['execute','execute']
        f=goal_forecast(c,paths,audit,horizon=2,unit='ticks',target='mission')
        k=next(k for k,v in f.roots.items() if v==('execute',))
        self.assertEqual(f.purpose[k],0.)
        audit['execute'][0]['assessment']=Goal('mission','failure',-1.).record()
        f=goal_forecast(c,paths,audit,horizon=2,unit='ticks',target='mission')
        k=next(k for k,v in f.roots.items() if v==('execute',))
        self.assertEqual(f.purpose[k],-1.)

    def test_progress_expiry_does_not_strip_entire_principle_tier(self):
        c=context('tier',[action('hold',effect(values={'benevolence':1.})),action('other',effect(.8))],values={'benevolence':1.},mode='principle')
        c['state'].update(primary_need='physiology',mode_urgency=.1)
        p=PurposeRequest(0.,0.,{'hold':Activity('idle','unsupported-hold'),'other':Activity('uncertain','purpose-attempt')})
        watch=PersonaProgressWatch(c['scope'],ProgressConfig(grace=1))
        watch.observe(0,'hold',p,PurposeFeedback(0.,0.));c['tick']=1
        allowed,audit=watch.mask(c,p)
        self.assertIn('hold',allowed);self.assertFalse(audit['applied']);self.assertTrue(audit['unresolved'])
        from reflex.progress import ProgressWatch
        restored=ProgressWatch.from_record(c['scope'],watch.record())
        self.assertEqual(restored.mask(c,p),watch.mask(c,p))

    def test_actual_threshold_releases_science_wait(self):
        s=dict(genre='resources',route='science',opening='conversion',limit=5)
        p=GoalProbe(s,profiles()[0]);w=p.start();c=p.observe(w)
        self.assertEqual(p.purpose(w,c).activities['wait'].kind,'idle')
        e=replace(w.empires[0],science=10);w=replace(w,empires=(e,)+w.empires[1:]);c=p.observe(w)
        a=p.purpose(w,c).activities['wait']
        self.assertEqual(a.kind,'wait');self.assertEqual(a.release,'science-threshold-ready');self.assertEqual(a.patience,1)

    def test_unobserved_events_do_not_change_initial_context_or_actor_stream(self):
        s=dict(genre='resources',route='mixed',limit=8)
        a=GoalProbe(s,profiles()[0]);b=GoalProbe({**s,'famine':True},profiles()[0])
        self.assertEqual(a.observe(a.start()),b.observe(b.start()))
        s=dict(genre='auction',prizes=[8,10],budget=6)
        a=GoalProbe(s,profiles()[1]);b=GoalProbe({**s,'rival_shift':True,'revised_prize':True},profiles()[1])
        self.assertEqual(a.observe(a.start()),b.observe(b.start()))

    def test_deadline_returns_settled_draw_instead_of_combat_proxy(self):
        p=GoalProbe(dict(genre='combat',hp=6,limit=1),profiles()[0]);w=p.start()
        after,_=p.advance(w,'guard',820)
        self.assertEqual(p.goal(after),Goal(p.target,'draw',0.))

    def test_mutual_elimination_keeps_game_draw_separate_from_personal_death(self):
        p=GoalProbe(dict(genre='combat',hp=3,limit=8),profiles()[0]);w=p.start()
        w=replace(w,units=tuple(replace(u,hp=0) for u in w.units),tick=2)
        self.assertEqual(p.goal(w),Goal(p.target,'draw',0.))

    def test_release_loop_spends_reserved_budget_without_hypothetical_learning(self):
        r=run(dict(genre='auction',prizes=[8,10],budget=6),profiles()[1],'release',200)
        self.assertEqual(r['actions'],['bid:0','bid:4']);self.assertEqual(r['status'],'scored')
        self.assertEqual(len({t['persona_hash'] for t in r['trace']}),1)
        self.assertEqual([t['watch']['last_tick'] for t in r['trace']],[0,1])


if __name__=='__main__':unittest.main()
