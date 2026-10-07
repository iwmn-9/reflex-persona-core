import copy
import unittest
from reflex.examples import context,action,effect
from reflex.flow_rollout import rollout
from reflex.intertemporal import forecast
from reflex.deliberation import select
from reflex.payback_cycle import Probe,run
from reflex.laboratory import profiles


class FlowRolloutTests(unittest.TestCase):
    def toy(self):
        def observe(s,memory=None):
            c=context('finite',[action('take',effect(.2)),action('prepare',effect(-.1))],mode='need')
            c['tick']=s['tick']
            if memory is not None:c['state']=copy.deepcopy(memory)
            return c
        def advance(s,k,seed):
            s['tick']+=1
            return s,effect(.2 if k=='take' else -.1)
        initial={'tick':0}
        return observe(initial),initial,dict(observe=observe,advance=advance,terminal=lambda s:s['tick']>=2,horizon=4,seeds=(800,801))

    def test_each_root_is_followed_by_persona_and_absorbing_zero_flow(self):
        c,s,kw=self.toy();paths,audit=rollout(c,s,**kw)
        self.assertEqual(set(paths),{'take','prepare'})
        for root in paths:
            self.assertEqual(audit[root][0]['actions'],[root,'take'])
            self.assertEqual(audit[root][0]['absorbed_at'],2)
            self.assertEqual([r['objective'] for r in paths[root][0].effects[2:]],[0,0])
            self.assertAlmostEqual(sum(b.probability for b in paths[root]),1)

    def test_branch_input_state_and_owner_are_not_mutated(self):
        c,s,kw=self.toy();before=copy.deepcopy((c,s));rollout(c,s,**kw)
        self.assertEqual((c,s),before)

    def test_hypothetical_intent_is_carried_without_changing_real_owner(self):
        c,s,kw=self.toy();seen=[];original=kw['observe']
        def observe(s,m):seen.append(copy.deepcopy(m));return original(s,m)
        kw['observe']=observe;rollout(c,s,**kw)
        self.assertEqual({m['intent_action'] for m in seen},{'take','prepare'})
        self.assertIsNone(c['state']['intent_action'])

    def test_changed_persona_cannot_enter_future_self_model(self):
        c,s,kw=self.toy();original=kw['observe']
        def observe(s,m):
            cc=original(s,m);cc['personality']['openness']=1.;return cc
        kw['observe']=observe
        with self.assertRaises(ValueError):rollout(c,s,**kw)

    def test_unknown_nonterminal_future_is_not_padded(self):
        c,s,kw=self.toy()
        def failed(*args):raise RuntimeError('model unavailable')
        kw['advance']=failed
        with self.assertRaisesRegex(RuntimeError,'model unavailable'):rollout(c,s,**kw)

    def test_invalid_seed_horizon_and_illegal_continuation_are_rejected(self):
        c,s,kw=self.toy()
        for change in (dict(seeds=(800,800)),dict(seeds=()),dict(horizon=17),dict(horizon=True),dict(continuation=lambda s,c:'illegal')):
            with self.assertRaises(ValueError):rollout(c,s,**{**kw,**change})

    def test_one_step_rollout_selects_same_actual_immediate_effects(self):
        from reflex.core import Policy
        c,s,kw=self.toy();kw['horizon']=1;paths,_=rollout(c,s,**kw)
        f=forecast(c,paths,horizon=1,unit='turns',target='goal')
        ds,_=select([c],f)
        self.assertEqual(ds[0]['action_id'],Policy().choose(c,False)['action_id'])

    def test_public_revision_is_not_visible_before_it_occurs(self):
        profile=profiles()[1];base=dict(genre='auction',prizes=[8,10],budget=6)
        a=Probe(base,profile);b=Probe({**base,'revised_prize':True},profile)
        sa=a.start();sb=b.start()
        self.assertEqual(a.advance(sa,'bid:0',800),b.advance(sb,'bid:0',800))
        self.assertEqual(a.observe(sa),b.observe(sb))
        shifted=Probe({**base,'rival_shift':True},profile)
        self.assertEqual(a.observe(sa),shifted.observe(shifted.start()))
        normal=Probe(dict(genre='resources',limit=8),profile)
        famine=Probe(dict(genre='resources',limit=8,famine=True),profile)
        self.assertEqual(normal.observe(normal.start()),famine.observe(famine.start()))
        self.assertEqual(b.actual(sb,'bid:0',190)[0].prize,3)
        self.assertEqual(b.advance(sb,'bid:0',800)[0].prize,10)

    def test_replanning_reserved_auction_money_is_actually_spent_at_payback(self):
        s=dict(genre='auction',prizes=[8,10],budget=6)
        r=run(s,profiles()[1],'persona')
        self.assertEqual(r['actions'],['bid:0','bid:4'])
        self.assertAlmostEqual(r['score'],10/30)
        self.assertEqual(len({t['persona_hash'] for t in r['trace']}),1)

    def test_combat_one_tick_deadline_never_collects_postmortem_tail(self):
        p=Probe(dict(genre='combat',hp=3,limit=1),profiles()[0]);w=p.start();c=p.observe(w)
        paths,audit=rollout(c,w,observe=p.observe,advance=p.advance,terminal=p.terminal,horizon=6,seeds=p.seeds)
        for bs in paths.values():
            for b in bs:self.assertEqual([e['objective'] for e in b.effects[1:]],[0]*5)
        self.assertTrue(all(t['terminal'] for ts in audit.values() for t in ts))
        self.assertTrue(set(p.seeds).isdisjoint({190,191}))


if __name__=='__main__':unittest.main()
