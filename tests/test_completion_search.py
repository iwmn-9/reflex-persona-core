import copy,unittest
from reflex.examples import context,action,effect
from reflex.purpose_plan import Goal
from reflex.completion_search import completion_forecast


def chain(length):
    c=context('opaque',[action('a',effect()),action('b',effect())],needs={'physiology':0.,'safety':0.})
    def observe(s,m):
        cc=copy.deepcopy(c);cc['tick']=s[0];cc['state']=m;cc['actions']=[action('continue',effect())];return cc
    def advance(s,k,seed):return (s[0]+1,k=='b' if s[0]==0 else s[1]),effect()
    def assess(s):
        done=s[0]>=length;v=(1. if s[1] else -1.) if done else 0.
        return Goal('reach',( 'success' if s[1] else 'failure') if done else 'running',v).record()
    return c,dict(observe=observe,advance=advance,terminal=lambda s:s[0]>=length,assess=assess,
        seeds=(1,2),target='reach',width=1,depth=0,settlement_weight='absolute')


class CompletionSearchTests(unittest.TestCase):
    def test_all_root_cutoffs_are_extended_without_a_game_specific_score(self):
        c,args=chain(5);f=completion_forecast(c,(0,False),horizon=2,max_horizon=8,**args)
        self.assertEqual([r['horizon'] for r in f.audit['completion_search']['levels']],[2,4,8])
        self.assertEqual(set(v[0] for v in f.roots.values()),{'a','b'})
        self.assertTrue(f.audit['completion_search']['all_model_endpoints_settled'])
        self.assertEqual(sorted(f.purpose.values()),[-1.,1.])

    def test_unfinished_models_remain_explicit_at_budget_cap(self):
        c,args=chain(9);f=completion_forecast(c,(0,False),horizon=2,max_horizon=5,**args)
        self.assertEqual(f.horizon,5);self.assertFalse(f.audit['completion_search']['all_model_endpoints_settled'])
        self.assertTrue(all(v==0. for v in f.purpose.values()))

    def test_early_terminal_stops_escalation(self):
        c,args=chain(1);f=completion_forecast(c,(0,False),horizon=2,max_horizon=16,**args)
        self.assertEqual(len(f.audit['completion_search']['levels']),1);self.assertEqual(f.horizon,2)

    def test_same_target_and_retained_plan_survive_escalation(self):
        c,args=chain(5);f=completion_forecast(c,(0,False),horizon=2,max_horizon=8,retained=('b','continue'),repairs=4,repair_guard=True,**args)
        self.assertEqual(f.target,'reach');self.assertIsNotNone(f.stability_reference)
        self.assertEqual(f.stability_reference.horizon,f.horizon)

    def test_invalid_caps_and_adapter_use_are_rejected(self):
        c,args=chain(5)
        for cap in (True,1,17):
            with self.assertRaises(ValueError):completion_forecast(c,(0,False),horizon=2,max_horizon=cap,**args)
        from reflex.observed_transfer import run
        from reflex.laboratory import profiles
        with self.assertRaises(ValueError):run({'genre':'delivery','limit':4},profiles()[0],'verified',1,complete_to=8)

    def test_long_future_intention_expands_every_root_without_truncation(self):
        c,args=chain(7)
        kept=('b',)+('continue',)*5
        f=completion_forecast(c,(0,False),horizon=2,max_horizon=8,retained=kept,repairs=4,repair_guard=True,**args)
        audit=f.audit['completion_search']
        self.assertEqual(audit['requested_horizon'],2)
        self.assertEqual(audit['retained_length'],6)
        self.assertEqual([v['horizon'] for v in audit['levels']],[6,8])
        self.assertEqual(set(v[0] for v in f.roots.values()),{'a','b'})
        proposal=f.audit['continuation_search']['proposals']['proposal-retained']
        self.assertEqual([proposal['root']]+proposal['schedule'],list(kept))
        self.assertEqual(f.stability_reference.horizon,f.horizon)

    def test_future_intention_cannot_silently_exceed_the_completion_cap(self):
        c,args=chain(7)
        for kept in (('b',)+('continue',)*8,'b',(1,)):
            with self.assertRaises(ValueError):
                completion_forecast(c,(0,False),horizon=2,max_horizon=8,retained=kept,**args)


if __name__=='__main__':unittest.main()
