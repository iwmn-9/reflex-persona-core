"""Memory copies share immutable evidence, never mutable owner state."""
import copy
from collections import OrderedDict, deque
from dataclasses import FrozenInstanceError
import json
import unittest

import numpy as np

from reflex.cloning import deepcopy as clone
from reflex.core import FEATURES, Policy, compile_batch, digest
from reflex.decision_loop import DecisionLoop, Request
from reflex.examples import action, context, effect
from reflex.judgment import Binding, OutcomeMemory, _OutcomeEntry, choose
from reflex.planning import vector


def fixture():
    c=context('immutable-memory',[action('A',effect(.2))],needs={})
    for name in c['needs']:c['needs'][name]=dict(supported=False,enabled=False,deficit=None)
    return c


def bindings(c,situation='same'):
    return {a['id']:Binding(a['id'],situation,('objective',)) for a in c['actions']}


def trial(memory,c,value,situation='same'):
    bound=bindings(c,situation);prepared,_=memory.prepare(c,bound)
    decision,_=choose(prepared,stochastic=False)
    ticket=memory.commit(prepared,decision,bound)
    result=memory.observe(ticket,vector(effect(value)))
    c['tick']+=1
    return result


class ImmutableOutcomeMemoryTests(unittest.TestCase):
    def test_entry_reading_and_sharing_do_not_share_mutable_memory(self):
        c=fixture();memory=OutcomeMemory(c['scope'],capacity=2)
        trial(memory,c,.2);trial(memory,c,.3)
        entry=memory.entries[('A','same')]
        for copier in (copy.deepcopy,clone):
            with self.subTest(copier=copier.__module__):
                copied=copier(memory)
                self.assertIsNot(copied,memory)
                self.assertIsNot(copied.entries,memory.entries)
                self.assertIs(copied.entries[('A','same')],entry)
                self.assertEqual(entry['count'],2)
                self.assertEqual(entry['estimated'],('objective',))
                self.assertTrue(all(type(row) is tuple and all(type(v) is float for v in row)
                                    for row in entry['samples']))
                copied.entries.clear()
                self.assertEqual(len(memory.entries),1)
        with self.assertRaises(FrozenInstanceError):entry.count=4
        with self.assertRaises(TypeError):entry['samples'][0][0]=.9
        with self.assertRaises(KeyError):entry['missing']

    def test_observation_replaces_entry_keeps_last_four_and_preserves_json(self):
        c=fixture();memory=OutcomeMemory(c['scope']);previous=None
        for i in range(14):
            if previous is not None:old_values=previous.samples
            trial(memory,c,i/100)
            entry=memory.entries[('A','same')]
            if previous is not None:
                self.assertIsNot(entry,previous)
                self.assertEqual(previous.samples,old_values)
            previous=entry
        expected=dict(version='outcome-memory-v1',owner=digest(c['scope']),capacity=64,last_tick=13,
            entries=[dict(method='A',situation='same',estimated=['objective'],count=12,
                          samples=[vector(effect(i/100)).tolist() for i in range(10,14)])])
        self.assertEqual(json.dumps(memory.record(),sort_keys=True),json.dumps(expected,sort_keys=True))
        last=entry
        result=trial(memory,c,1.)
        self.assertTrue(result['regime_reset'])
        self.assertEqual(result['observations'],1)
        self.assertEqual(memory.entries[('A','same')]['samples'],(tuple(vector(effect(1.)).tolist()),))
        self.assertEqual(last.count,12)
        self.assertEqual(last.samples,tuple(tuple(v) for v in expected['entries'][0]['samples']))

    def test_saved_lists_and_observation_arrays_are_detached(self):
        c=fixture();memory=OutcomeMemory(c['scope']);bound=bindings(c)
        decision,_=choose(c,stochastic=False);observed=vector(effect(.25))
        memory.observe(memory.commit(c,decision,bound),observed)
        observed[0]=-.5
        expected=memory.record()
        saved=json.loads(json.dumps(expected))
        restored=OutcomeMemory.from_record(c['scope'],saved)
        self.assertEqual(restored.record(),expected)
        self.assertIs(copy.deepcopy(restored.entries[('A','same')]),restored.entries[('A','same')])
        saved['entries'][0]['samples'][0][0]=.9
        saved['entries'][0]['estimated'][0]='cost'
        self.assertEqual(restored.record(),expected)
        self.assertEqual(memory.record(),expected)

    def test_empirical_mixture_and_policy_match_original_numpy_sample_storage(self):
        c=fixture()
        c['actions']=[action('A',effect(.6,cost=.1,p=.25),effect(-.6,cost=.7,p=.75))]
        memory=OutcomeMemory(c['scope'])
        for value in (.1,-.1,.2,-.2):trial(memory,c,value)
        legacy=copy.deepcopy(memory)
        legacy.entries=OrderedDict((key,dict(count=e['count'],estimated=e['estimated'],
            samples=deque((np.array(v,dtype=float) for v in e['samples']),maxlen=4)))
            for key,e in memory.entries.items())
        expected,old_stats=legacy.prepare(c,bindings(c))
        actual,new_stats=memory.prepare(c,bindings(c))
        self.assertEqual(digest(actual),digest(expected));self.assertEqual(new_stats,old_stats)
        old_batch,new_batch=compile_batch([expected]),compile_batch([actual])
        for name in old_batch.__dataclass_fields__:
            old,new=getattr(old_batch,name),getattr(new_batch,name)
            if isinstance(old,np.ndarray):self.assertEqual(old.tobytes(),new.tobytes(),name)
            else:self.assertEqual(old,new,name)
        self.assertEqual(Policy().decide(old_batch,False).records(old_batch),
                         Policy().decide(new_batch,False).records(new_batch))

    def test_actual_and_hypothetical_owners_update_independently(self):
        c=fixture();actual=DecisionLoop(c)
        def req():return Request(c,bindings(c),{'A':()})
        for tick in range(4):
            c['tick']=tick;selected=actual.decide(req(),False)
            actual.observe(selected['ticket'],vector(effect(.2)))
        before=actual.record();future=copy.deepcopy(actual)
        self.assertIsNot(actual.memory.entries,future.memory.entries)
        self.assertIs(actual.memory.entries[('A','same')],future.memory.entries[('A','same')])
        c['tick']=4;selected=future.decide(req(),False)
        future.observe(selected['ticket'],vector(effect(-.8)))
        future_saved=future.record()
        self.assertEqual(actual.record(),before)
        selected=actual.decide(req(),False);actual.observe(selected['ticket'],vector(effect(.3)))
        self.assertEqual(future.record(),future_saved)
        self.assertNotEqual(actual.memory.record(),future.memory.record())
        self.assertEqual(actual.personality,future.personality)
        self.assertEqual(actual.values,future.values)

    def test_pending_trials_and_eviction_remain_owner_local(self):
        c=fixture();memory=OutcomeMemory(c['scope'],capacity=2)
        trial(memory,c,.2,'one');trial(memory,c,.2,'two')
        before=memory.record();future=copy.deepcopy(memory)
        trial(future,copy.deepcopy(c),.2,'three')
        self.assertEqual(memory.record(),before)
        self.assertEqual(list(future.entries),[('A','two'),('A','three')])
        bound=bindings(c,'two');prepared,_=memory.prepare(c,bound)
        decision,_=choose(prepared,stochastic=False);ticket=memory.commit(prepared,decision,bound)
        future=copy.deepcopy(memory)
        self.assertEqual(future.ticket,ticket)
        self.assertIsNot(future.pending,memory.pending)
        self.assertIsNot(future.pending['predicted'][0],memory.pending['predicted'][0])
        future.pending['predicted'][0][0]=-.7
        self.assertNotEqual(future.pending['predicted'][0][0],memory.pending['predicted'][0][0])
        future.observe(ticket,vector(effect(.1)))
        self.assertIsNone(future.pending);self.assertIsNotNone(memory.pending)
        memory.observe(ticket,vector(effect(.2)))
        self.assertNotEqual(future.record(),memory.record())

    def test_malformed_subclass_and_replaced_fields_cannot_share_mutable_data(self):
        class MutableLabel(str):pass
        label=MutableLabel('objective');label.extra=[]
        malformed=_OutcomeEntry(1,(label,),([.2]*len(FEATURES),))
        copied=copy.deepcopy(malformed)
        self.assertIsNot(copied,malformed)
        copied.estimated[0].extra.append('changed');copied.samples[0][0]=.9
        self.assertEqual(label.extra,[]);self.assertEqual(malformed.samples[0][0],.2)
        class ExtraEntry(_OutcomeEntry):pass
        extra=ExtraEntry(1,('objective',),((.2,),));extra.notes=[]
        copied=copy.deepcopy(extra);copied.notes.append('changed')
        self.assertEqual(extra.notes,[])
        valid=_OutcomeEntry(1,('objective',),((.2,),))
        object.__setattr__(valid,'samples',([.2],))
        copied=copy.deepcopy(valid);copied.samples[0][0]=.9
        self.assertEqual(valid.samples[0][0],.2)
        extra=_OutcomeEntry(1,('objective',),((.2,),))
        object.__setattr__(extra,'notes',[])
        copied=copy.deepcopy(extra);copied.notes.append('changed')
        self.assertEqual(extra.notes,[])

    def test_defensive_copy_preserves_memo_aliases_and_cycles(self):
        values=[];malformed=_OutcomeEntry(1,('objective',),(values,values));values.append(malformed)
        copied=copy.deepcopy(malformed)
        self.assertIsNot(copied,malformed)
        self.assertIs(copied.samples[0],copied.samples[1])
        self.assertIs(copied.samples[0][0],copied)
        valid=_OutcomeEntry(1,('objective',),((.2,),));marker=object()
        self.assertIs(copy.deepcopy(valid,{id(valid):marker}),marker)


if __name__=='__main__':unittest.main()
