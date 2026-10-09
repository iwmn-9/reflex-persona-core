import copy
import unittest
from reflex.core import Policy,digest
from reflex.examples import context,action,effect
from reflex.flow_rollout import rollout
from reflex.continuation_search import search_forecast
from reflex.purpose_plan import Goal


class ScheduleSamplingTests(unittest.TestCase):
    def fixture(self):
        c=context('delayed',[action('start',effect())],needs={'physiology':0.,'safety':0.})
        def observe(s,m):
            cc=copy.deepcopy(c);cc['tick']=s[0];cc['state']=m
            cc['actions']=[action('comfort',effect(cost=0.)),action('prepare',effect(cost=.1))]
            return cc
        def advance(s,k,seed):return (s[0]+1,s[1]+int(k=='prepare')),effect(cost=.1 if k=='prepare' else 0.)
        def assess(s):return Goal('finish','success' if s[1]>=3 else 'draw',1. if s[1]>=3 else 0.).record()
        return c,dict(observe=observe,advance=advance,terminal=lambda s:s[0]>=4,assess=assess,
            horizon=4,target='finish',depth=0,width=2,policy=Policy(principle_priority='finite'))

    def test_sampling_discovers_delayed_chain_and_preserves_reflex(self):
        c,kw=self.fixture();saved=digest(c)
        old=search_forecast(c,(0,0),**kw)
        new=search_forecast(c,(0,0),samples=32,**kw)
        self.assertEqual(max(old.purpose.values()),0.)
        self.assertEqual(max(new.purpose.values()),1.)
        proposals=new.audit['continuation_search']['proposals']
        self.assertTrue(any(not n['schedule'] for n in proposals.values()))
        self.assertEqual(new.audit['continuation_search']['pilot_evaluations'],32)
        self.assertEqual(digest(c),saved)
        self.assertEqual(digest(new.audit),digest(search_forecast(c,(0,0),samples=32,**kw).audit))

    def test_frozen_sequence_is_evaluated_across_model_branches(self):
        c,kw=self.fixture()
        def advance(s,k,seed):return (s[0]+1,seed,k),effect()
        def assess(s):return Goal('finish','success' if (s[1]==0)==(s[2]=='prepare') else 'failure',
            1. if (s[1]==0)==(s[2]=='prepare') else -1.).record()
        kw.update(advance=advance,assess=assess,horizon=2,terminal=lambda s:s[0]>=2,seeds=(0,1))
        f=search_forecast(c,(0,None,None),samples=8,**kw)
        self.assertTrue(all(v==0. for v in f.purpose.values()))
        for n in f.audit['continuation_search']['proposals'].values():
            if n['schedule']:self.assertEqual(n['branches'][0]['actions'],n['branches'][1]['actions'])

    def test_pilot_has_only_legal_observation_and_cannot_mix_continuations(self):
        c,kw=self.fixture();seen=[]
        def sample(keys,tick):seen.append((keys,tick));return keys[-1]
        paths,audit=rollout(c,(0,0),**{k:v for k,v in kw.items() if k in ('observe','advance','terminal','assess','horizon','policy')},schedule_sampler=sample)
        self.assertEqual([t for _,t in seen],[1,2,3])
        self.assertTrue(all(keys==('comfort','prepare') for keys,_ in seen))
        self.assertEqual(audit['start'][0]['actions'],['start','prepare','prepare','prepare'])
        for args in (dict(seeds=(0,1)),dict(schedule=('prepare',)),dict(continuation=lambda s,c:'prepare')):
            with self.assertRaises(ValueError):rollout(c,(0,0),observe=kw['observe'],advance=kw['advance'],terminal=kw['terminal'],horizon=4,schedule_sampler=sample,**args)

    def test_disabled_and_invalid_budgets(self):
        c,kw=self.fixture()
        old=search_forecast(c,(0,0),**kw);disabled=search_forecast(c,(0,0),samples=0,**kw)
        self.assertEqual(digest(old.audit),digest(disabled.audit))
        self.assertNotIn('sampling',old.audit['continuation_search'])
        for n in (-1,65,True,1.5):
            with self.assertRaises(ValueError):search_forecast(c,(0,0),samples=n,**kw)


if __name__=='__main__':unittest.main()
