import copy
from dataclasses import replace
import unittest
import numpy as np
from reflex.core import digest
from reflex.examples import context,action,effect
from reflex.purpose_plan import Goal
from reflex.continuation_search import search_forecast
from reflex.deliberation import select
from reflex.plan_continuity import PlanIntention
from reflex.plan_repair import patches,edits,RevisionPreference


def corridor():
    c=context('repair',[action('start',effect())],needs={'physiology':0.,'safety':0.})
    def observe(s,m):
        cc=copy.deepcopy(c);cc['tick']=s[0];cc['state']=m
        keys=('finish',) if s[0]==4 else ('left','right')
        cc['actions']=[action(k,effect()) for k in keys];return cc
    def advance(s,k,seed):return (s[0]+1,k=='right' if s[0]==3 else s[1]),effect()
    def assess(s):
        done=s[0]>=5
        return Goal('finish-goal',('success' if s[1] else 'failure') if done else 'running',
            (1. if s[1] else -1.) if done else 0.).record()
    return c,dict(observe=observe,advance=advance,assess=assess,terminal=lambda s:s[0]>=5,
        horizon=5,depth=0,seeds=(1,2),target='finish-goal')


class PlanRepairTests(unittest.TestCase):
    def test_round_robin_repairs_are_single_shared_edits(self):
        prior=('a','b','c');choices=({'a','x','y'},{'b','z'},{'c','q'})
        rows=patches(prior,choices,limit=3)
        self.assertEqual(rows,(('x','b','c'),('a','z','c'),('a','b','q')))
        self.assertTrue(all(edits(prior,r)==1 for r in rows))
        self.assertEqual(patches(prior,choices,limit=0),())
        self.assertEqual(prior,('a','b','c'))

    def test_bad_bank_controls_and_tokens_are_rejected(self):
        for args in (((),(),24),(('x',),(),24),(('x',),({'y'},),True),(('x',),({'y'},),65)):
            with self.assertRaises(ValueError):patches(args[0],args[1],limit=args[2])
        with self.assertRaises(ValueError):patches(('x',),({''},))

    def test_late_repair_beats_reflex_completion_without_changing_goal(self):
        c,args=corridor();old=('start','left','left','left','finish');saved=digest(c)
        a=search_forecast(c,(0,False),retained=old,**args)
        b=search_forecast(c,(0,False),retained=old,repairs=24,**args)
        self.assertTrue(all(v==-1. for v in a.purpose.values()))
        self.assertEqual(max(b.purpose.values()),1.)
        records,audit=select([c],b);proposal=b.audit['plans'][audit['selected_plan']]['proposal']
        self.assertTrue(proposal.startswith('proposal-repair-'))
        chosen=b.audit['continuation_search']['proposals'][proposal]
        self.assertEqual(tuple([chosen['root']]+chosen['schedule']),('start','left','left','right','finish'))
        self.assertEqual(b.target,a.target);self.assertEqual(saved,digest(c))
        memory=PlanIntention(c);memory.remember(c,b,audit['selected_plan'],records[0]['action_id'])
        self.assertEqual(memory.target,'finish-goal');self.assertEqual(memory.remaining,('left','left','right','finish'))

    def test_all_branches_recheck_same_patch_and_no_future_seed_choice(self):
        c,args=corridor();f=search_forecast(c,(0,False),retained=('start','left','left','left','finish'),repairs=24,**args)
        for name,row in f.audit['continuation_search']['proposals'].items():
            if name.startswith('proposal-repair-'):
                self.assertEqual(row['branches'][0]['actions'],row['branches'][1]['actions'])
                self.assertEqual([r['seed'] for r in row['branches']],[1,2])

    def test_invalid_old_head_can_be_repaired_without_forcing_it(self):
        c,args=corridor();f=search_forecast(c,(0,False),retained=('blocked','left','left'),repairs=24,**args)
        rs=[v for k,v in f.audit['continuation_search']['proposals'].items() if k.startswith('proposal-repair-')]
        self.assertTrue(rs);self.assertTrue(all(r['root']=='start' for r in rs))
        self.assertIsNone(f.continuity)

    def test_changed_value_and_purpose_do_not_lose_to_edit_distance(self):
        names=('new','old');p=RevisionPreference('old',{'new':(None,),'old':(None,)},{'new':5,'old':0})
        enabled=np.array([True,True])
        self.assertEqual(p.resolve_tie(names,enabled,np.array([1.,.5]),np.array([1.,1.]),0),0)
        self.assertEqual(p.resolve_tie(names,enabled,np.array([1.,1.]),np.array([1.,.5]),0),0)
        self.assertEqual(p.resolve_tie(names,enabled,np.array([1.,1.]),np.array([1.,1.]),0),1)
        self.assertEqual(p.resolve_tie(names,np.array([True,False]),np.ones(2),np.ones(2),0),0)

    def test_held_out_bank_contains_every_repair(self):
        c,args=corridor();f=search_forecast(c,(0,False),retained=('start','left','left','left','finish'),repairs=24,
            validation_seeds=(3,4),**args)
        rows=f.audit['continuation_search'];v=rows['validation']
        self.assertEqual(set(v['discovery']),set(rows['proposals']))
        for name,row in rows['proposals'].items():
            self.assertEqual([b['seed'] for b in row['branches']],[3,4])
            self.assertEqual([b['seed'] for b in v['discovery'][name]],[1,2])

    def test_no_persistent_plan_means_no_repair_proposals(self):
        c,args=corridor();f=search_forecast(c,(0,False),repairs=24,**args)
        self.assertEqual(f.audit['continuation_search']['repair']['proposed'],0)
        self.assertFalse(any(k.startswith('proposal-repair-') for k in f.audit['continuation_search']['proposals']))

    def test_full_replacement_remains_possible_and_executed_prefix_is_not_rewritten(self):
        c,args=corridor();m=PlanIntention(c)
        for tick,sequence in enumerate((('start','left','left'),('right','right','finish'),('finish',))):
            cc=copy.deepcopy(c);cc['tick']=tick
            key='new';f=replace(search_forecast(c,(0,False),**args),roots={key:(sequence[0],)},target='finish-goal',
                audit={'unit':'turns','plans':{key:{'proposal':key}},'continuation_search':{'proposals':{key:{'branches':[{'actions':list(sequence)}]}}}})
            m.remember(cc,f,key,sequence[0])
            self.assertEqual(m.remaining,sequence[1:]);self.assertEqual(m.target,'finish-goal')
        self.assertEqual(m.remaining,());self.assertEqual(m.last_tick,2)

    def test_revision_contract_has_complete_coverage(self):
        c,args=corridor();f=search_forecast(c,(0,False),retained=('start','left'),repairs=24,**args)
        self.assertIsInstance(f.continuity,RevisionPreference)
        with self.assertRaises(ValueError):replace(f.continuity,distances={}).validate(f)
        with self.assertRaises(ValueError):replace(f.continuity,distances={k:True for k in f.roots}).validate(f)

    def test_repair_parameter_requires_bounded_owned_search(self):
        c,args=corridor()
        for value in (-1,True,65):
            with self.assertRaises(ValueError):search_forecast(c,(0,False),repairs=value,**args)
        from reflex.observed_transfer import run
        from reflex.laboratory import profiles
        with self.assertRaises(ValueError):run({'genre':'delivery','limit':6},profiles()[0],'verified',1,repairs=1)


if __name__=='__main__':unittest.main()
