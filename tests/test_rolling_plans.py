from dataclasses import dataclass,replace
import unittest
from unittest.mock import patch
import numpy as np
from reflex.core import digest
from reflex.laboratory import profiles
from reflex.rule_baseline import Rewards,ConnectRules
from reflex.board_models import ConnectPosition,ConnectAdapter
from reflex.rolling_plans import (Plan,PlanMemory,PlanBudget,Counter,candidate_plans,sample_plans,decide,validation_gate)
from reflex.rule_baseline_experiment import play


@dataclass(frozen=True)
class Delayed:
    stage:int=0
    target:int=0
    correct:bool=True


class DelayedRules:
    """No intermediate rewards. Both own choices must follow public target."""
    game='delayed_public_goal';players=2;simultaneous=False
    def terminal(self,s):return s.stage==2
    def chance(self,s):return False
    def actor(self,s):return 0
    def legal(self,s,viewer=None):return () if self.terminal(s) else ('LEFT','RIGHT')
    def root(self,s,a,viewer):return self.step(s,a)
    def step(self,s,a):
        if a not in self.legal(s):raise ValueError('illegal')
        return replace(s,stage=s.stage+1,correct=s.correct and (a=='RIGHT')==bool(s.target))
    def rewards(self,s):
        if not self.terminal(s):raise ValueError('no intermediate score')
        c=(float(s.correct),float(not s.correct));return Rewards(c,tuple(2*x-1 for x in c),c)
    def observation(self,s):return dict(stage=s.stage,target=s.target,correct=s.correct)


SMALL=PlanBudget(horizon=2,variants=12,search_samples=8,validation_samples=16,max_nodes=4000,bootstrap_samples=16)


class RollingPlanTests(unittest.TestCase):
    def test_controls_and_owner_isolation(self):
        for kwargs in (dict(horizon=0),dict(max_nodes=-1),dict(search_samples=2),dict(margin=-1),dict(persist=1)):
            with self.assertRaises(ValueError):PlanBudget(**kwargs)
        with self.assertRaises(ValueError):Plan((1.,))
        m=PlanMemory('other',(.25,))
        with self.assertRaises(ValueError):decide(DelayedRules(),Delayed(),0,profiles()[0],'rules_persona',SMALL,1,0,'x',plan_memory=m)
        self.assertEqual(PlanMemory.from_record(m.record()),m)

    def test_root_coverage_quantiles_remain_legal(self):
        names=('a','b','c')
        plans=candidate_plans(names,SMALL,np.random.default_rng(4),Plan((.9,)))
        self.assertEqual({p.move(names,0,0) for p in plans},set(names))
        for p in plans:
            for n in (1,2,13):self.assertIn(p.move(tuple(range(n)),1,.4),tuple(range(n)))

    def test_delayed_goal_plan_beats_random_own_continuation(self):
        c,d,s,m=decide(DelayedRules(),Delayed(),0,profiles()[0],'rules_persona',SMALL,1,0,'delayed')
        long=[x['estimated_share'] for x in s['discovery'] if len(x['genes'])==2]
        flat=[x['estimated_share'] for x in s['discovery'] if len(x['genes'])==1]
        self.assertEqual(max(long),1.);self.assertLess(max(flat),1.)
        self.assertEqual(d['action_id'],'LEFT');self.assertTrue(m.remaining)
        self.assertTrue(s['validation']['used']);self.assertEqual(d['context_hash'],digest(c))
        self.assertEqual(c['actions'][0]['id'],d['action_id'])

    def test_stable_plan_is_held_but_public_change_triggers_switch(self):
        rules=DelayedRules();profile=profiles()[0]
        c,d,s,m=decide(rules,Delayed(),0,profile,'rules_persona',SMALL,1,0,'change')
        after=rules.step(Delayed(),d['action_id'])
        _,held,h,hm=decide(rules,after,0,profile,'rules_persona',SMALL,1,1,'change',d['next_state'],m)
        self.assertFalse(h['plan_changed']);self.assertEqual(held['action_id'],'LEFT')
        _,changed,x,xm=decide(rules,replace(after,target=1),0,profile,'rules_persona',SMALL,1,1,'change',d['next_state'],m)
        self.assertTrue(x['plan_changed']);self.assertTrue(x['validation']['used'])
        self.assertEqual(changed['action_id'],'RIGHT');self.assertEqual(hm.generation+1,xm.generation)
        self.assertGreater(x['validation']['persona_gain'],0)

    def test_same_personality_and_modes_in_counterfactual_change(self):
        p=profiles()[0]
        c,d,s,m=decide(DelayedRules(),Delayed(),0,p,'rules_persona',SMALL,1,0,'fixed')
        state=Delayed(stage=1,correct=True)
        a,da,sa,ma=decide(DelayedRules(),state,0,p,'rules_persona',SMALL,1,1,'fixed',d['next_state'],m)
        b,db,sb,mb=decide(DelayedRules(),replace(state,target=1),0,p,'rules_persona',SMALL,1,1,'fixed',d['next_state'],m)
        for key in ('personality','values','needs','state','seed','scope'):self.assertEqual(a[key],b[key])
        self.assertEqual(da['next_state']['mode'],db['next_state']['mode'])

    def test_cut_trials_never_become_a_loss_and_every_plan_equal_samples(self):
        rules=DelayedRules();plans=[Plan((.25,.25)),Plan((.75,.75))];count=Counter(3)
        samples,stats=sample_plans(rules,Delayed(),0,plans,8,[2],count,3,5)
        self.assertEqual(stats['completed_samples'],0);self.assertEqual(stats['discarded_terminal_samples'],1)
        self.assertTrue(all(not x for x in samples.values()));self.assertEqual(count.nodes,3)
        samples,s=sample_plans(rules,Delayed(),0,plans,8,[2],Counter(99),99,1)
        self.assertEqual(s['completed_samples'],0)

    def test_low_budget_preserves_incumbent_without_fake_confirmation(self):
        _,d,stats,m=decide(DelayedRules(),Delayed(),0,profiles()[0],'rules_persona',SMALL,1,0,'cut')
        _,hold,s,remaining=decide(DelayedRules(),Delayed(stage=1,target=1),0,profiles()[0],'rules_persona',replace(SMALL,max_nodes=0),1,1,'cut',d['next_state'],m)
        self.assertFalse(s['plan_changed']);self.assertFalse(s['validation']['used'])
        self.assertEqual(s['total_nodes'],0);self.assertEqual(hold['action_id'],'LEFT')

    def test_switch_gate_rejects_unconfirmed_gain_and_ineligible(self):
        self.assertFalse(validation_gate(.5,-.1,True,True,0.,SMALL)[0])
        self.assertFalse(validation_gate(.5,.4,False,True,0.,SMALL)[0])
        self.assertTrue(validation_gate(.5,.4,True,True,0.,SMALL)[0])

    def test_no_game_heuristic_and_deterministic_replay(self):
        rules=ConnectRules();state=ConnectPosition()
        for col in (0,6,1,6,2,5):state=state.play(f'DROP:{col}')
        with patch.object(ConnectAdapter,'consequence',side_effect=AssertionError('forbidden')):
            a=decide(rules,state,0,profiles()[0],'rules_persona',SMALL,4,6,'win')
            b=decide(rules,state,0,profiles()[0],'rules_persona',SMALL,4,6,'win')
        self.assertEqual(a,b);self.assertEqual(a[1]['action_id'],'DROP:3')
        self.assertLessEqual(a[2]['total_nodes'],SMALL.max_nodes)

    def test_discovery_validation_streams_are_separate_and_plan_order_independent(self):
        with patch('reflex.rolling_plans.sample_plans',wraps=sample_plans) as spy:
            _,_,s,_=decide(DelayedRules(),Delayed(),0,profiles()[0],'rules_persona',SMALL,1,0,'delayed')
        self.assertTrue(s['validation']['used'])
        self.assertEqual({call.args[5][-1] for call in spy.call_args_list},{'discovery','validation'})
        plans=[Plan((.25,)),Plan((.75,))]
        a,_=sample_plans(DelayedRules(),Delayed(),0,plans,8,[7],Counter(99),99,4)
        b,_=sample_plans(DelayedRules(),Delayed(),0,plans[::-1],8,[7],Counter(99),99,4)
        self.assertEqual(a,b)

    def test_three_games_finish_with_bounded_confirmed_changes(self):
        small=replace(SMALL,variants=2,search_samples=4,validation_samples=8,bootstrap_samples=8,max_nodes=6000)
        for game,n in (('connect_four',2),('goofspiel',4),('no_thanks_basic',3)):
            r=play(game,n,0,0,'rules_persona','random',planner=small)
            self.assertTrue(r['finished'])
            for t in r['trace']:
                if t['own']:
                    s=t['stats'];self.assertLessEqual(s['total_nodes'],small.max_nodes)
                    if s['plan_changed']:self.assertTrue(s['validation']['used'])


if __name__=='__main__':unittest.main()
