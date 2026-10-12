import copy
import unittest
from reflex.examples import context,action,effect
from reflex.observation_plan import ObservationSchema,ObservationPlan,View,PlanMemory
from reflex.contingent_search import search_forecast
from reflex.core import Policy
from reflex.purpose_plan import Goal
from reflex.deliberation import select
from reflex.execution_window import Directive,ExecutionWindow


class ObservationSchemaTests(unittest.TestCase):
    def test_unrelated_facts_are_omitted_but_meaningful_numeric_changes_are_not(self):
        c=context('key',[action('a',effect(.2)),action('b',effect(.1))]);schema=ObservationSchema('world-v1',('options',))
        v=View.make('',c,schema=schema);plan=ObservationPlan.make(c,'target','ticks',{v.node:'a'},schema=schema)
        annotated=copy.deepcopy(c);annotated['facts']['debug']='new annotation'
        self.assertEqual(plan.choose(View.make('',annotated,schema=schema)),'a')
        annotated['actions'][0]['outcomes'][0]['objective']=-.3
        self.assertIsNone(plan.choose(View.make('',annotated,schema=schema)))
        annotated=copy.deepcopy(c);annotated['tick']=1
        self.assertIsNone(plan.choose(View.make('',annotated,schema=schema)))
        self.assertEqual(ObservationPlan.restore(plan.record()),plan)

    def test_schema_namespace_and_missing_required_facts_are_not_silent_fallbacks(self):
        c=context('key',[action('a',effect())]);a=ObservationSchema('world-v1',('options',));b=ObservationSchema('world-v2',('options',))
        v=View.make('',c,schema=a);plan=ObservationPlan.make(c,'target','ticks',{v.node:'a'},schema=a)
        with self.assertRaises(ValueError):plan.choose(View.make('',c,schema=b))
        with self.assertRaises(ValueError):View.make('',c,schema=ObservationSchema('required',('missing',)))

    def test_search_and_owned_memory_share_one_schema(self):
        c=context('search',[action('a',effect(.2)),action('b',effect(.1))]);schema=ObservationSchema('world-v1',('options',))
        def observe(t,state):
            cc=copy.deepcopy(c);cc.update(tick=t,state=state);cc['facts']['debug']=str(t);return cc
        f=search_forecast(c,0,observe=observe,advance=lambda t,k,s:(t+1,effect(.2)),terminal=lambda t:t==2,
            assess=lambda t:Goal('target','success',1.).record(),horizon=2,unit='ticks',target='target',width=1,revisions=1,
            policy=Policy(),observation_schema=schema)
        records,audit=select([c],f);memory=PlanMemory(c,'target','ticks',schema=schema)
        memory.remember(c,f,audit['selected_plan'],records[0]['action_id'])
        self.assertEqual(PlanMemory.restore(c,memory.record()).record(),memory.record())
        with self.assertRaises(ValueError):PlanMemory(c,'target','ticks').remember(c,f,audit['selected_plan'],records[0]['action_id'])


class ExecutionWindowTests(unittest.TestCase):
    scope=dict(game='test',episode='one',npc='actor')
    def make(self):return ExecutionWindow(self.scope,(Directive('grow','prepare-capacity',3),Directive('hold','preserve-capacity',3)))
    def test_reissue_does_not_renew_deadline_and_timeout_is_distinct(self):
        w=self.make();w.issue(self.scope,'grow',0,opportunity=True)
        w.observe(self.scope,0,opportunity=True)
        self.assertEqual(w.issue(self.scope,'grow',1,opportunity=True)['deadline'],2)
        w.observe(self.scope,1,opportunity=True)
        self.assertEqual(w.observe(self.scope,2,opportunity=True)['status'],'timeout')

    def test_no_opportunity_and_interruption_are_not_success_evidence(self):
        w=self.make();self.assertFalse(w.issue(self.scope,'grow',0,opportunity=False)['issued'])
        w.issue(self.scope,'grow',0,opportunity=True)
        w.issue(self.scope,'hold',0,opportunity=True)
        self.assertEqual(w.counts['grow']['interrupted'],1)
        self.assertEqual(w.observe(self.scope,2,opportunity=False)['status'],'unobserved')
        self.assertEqual(w.counts['hold']['success'],0)

    def test_criterion_success_is_not_reported_as_causal_control(self):
        w=self.make();w.issue(self.scope,'grow',0,opportunity=True)
        r=w.observe(self.scope,0,opportunity=True,met=True)
        self.assertTrue(r['criterion_is_not_causal_control']);self.assertEqual(r['status'],'success')
        w.issue(self.scope,'hold',1,opportunity=True)
        with self.assertRaises(ValueError):w.observe(self.scope,1,opportunity=False,met=True)
        self.assertEqual(w.last_tick,0)

    def test_late_criterion_cannot_turn_an_expired_window_into_a_success(self):
        w=self.make();w.issue(self.scope,'grow',0,opportunity=True);w.observe(self.scope,0,opportunity=True)
        with self.assertRaises(ValueError):w.issue(self.scope,'grow',3,opportunity=True)
        self.assertEqual(w.observe(self.scope,3,opportunity=True,met=True)['status'],'timeout')
        w.issue(self.scope,'hold',4,opportunity=True)
        with self.assertRaises(ValueError):w.issue(self.scope,'grow',3,opportunity=True)
