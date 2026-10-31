import copy
import copyreg
import unittest
from collections import OrderedDict

from reflex.cloning import deepcopy
from reflex.core import digest
from reflex.examples import context,action,effect


class CloneTests(unittest.TestCase):
    def test_instance_level_protocols_use_the_standard_library(self):
        class Owner:pass
        owner=Owner();owner.__deepcopy__=lambda memo: {'custom':True}
        self.assertEqual(deepcopy(owner),copy.deepcopy(owner))
        self.assertEqual(deepcopy({'owner':owner}),copy.deepcopy({'owner':owner}))
        owner=Owner();owner.__reduce_ex__=lambda protocol:(dict,(),None,None,iter([('reduced',True)]))
        self.assertEqual(deepcopy(owner),copy.deepcopy(owner))

    def test_ordinary_owner_graph_keeps_cycles_aliases_and_array_copy_semantics(self):
        import numpy as np
        class Owner:pass
        owner=Owner();shared=[];owner.memory=shared;owner.other=shared;shared.append(owner)
        owner.weights=np.array([.2,.7]);owner.weights.flags.writeable=False
        actual=deepcopy(owner);expected=copy.deepcopy(owner)
        self.assertIs(actual.memory,actual.other);self.assertIs(actual.memory[0],actual)
        self.assertIsNot(actual.memory,owner.memory)
        self.assertTrue(np.array_equal(actual.weights,expected.weights))
        self.assertEqual(actual.weights.flags.writeable,expected.weights.flags.writeable)
        actual.weights[0]=.8;self.assertEqual(owner.weights[0],.2)

    def test_registered_reduction_and_custom_state_are_not_bypassed(self):
        class Registered:pass
        def rebuild():
            value=Registered();value.used=True;return value
        copyreg.pickle(Registered,lambda obj:(rebuild,()))
        try:self.assertTrue(deepcopy(Registered()).used)
        finally:copyreg.dispatch_table.pop(Registered)
        class State:
            def __getstate__(self):return {'replaced':[1]}
        value=State();value.other=[2]
        self.assertEqual(vars(deepcopy(value)),vars(copy.deepcopy(value)))

    def test_custom_dictionary_key_and_value_hooks_run_in_reference_order(self):
        calls=[]
        class Hook:
            def __init__(self,name):self.name=name
            def __deepcopy__(self,memo):calls.append(self.name);return Hook(self.name)
        value={Hook('key'):Hook('value')}
        copy.deepcopy(value);expected=list(calls);calls.clear()
        deepcopy(value);self.assertEqual(calls,expected)

    def test_contexts_match_standard_copy_and_do_not_share_mutable_values(self):
        c=context('clone',[action(str(i),effect(i/10,needs={'growth':i/10})) for i in range(5)])
        copied=deepcopy(c)
        self.assertEqual(copied,copy.deepcopy(c))
        self.assertEqual(digest(copied),digest(c))
        copied['actions'][0]['outcomes'][0]['needs']['growth']=.9
        self.assertEqual(c['actions'][0]['outcomes'][0]['needs']['growth'],0.)

    def test_aliases_and_container_cycles_are_preserved(self):
        shared=[];root={'a':shared,'b':shared};shared.append(root)
        result=deepcopy(root)
        self.assertIs(result['a'],result['b']);self.assertIs(result['a'][0],result)
        self.assertIsNot(result['a'],shared)
        items=[];pair=(items,);items.append(pair)
        result=deepcopy(pair)
        self.assertIs(result[0][0],result);self.assertIsNot(result[0],items)

    def test_custom_hooks_run_once_on_the_original_graph(self):
        calls=[]
        class Custom:
            def __deepcopy__(self,memo):
                calls.append(self)
                memo[id(self)]=replacement
                return replacement
        replacement=object();obj=Custom();shared=[];root=[shared,{'x':obj,'y':obj},shared]
        result=deepcopy(root)
        self.assertEqual(calls,[obj]);self.assertIs(result[1]['x'],replacement)
        self.assertIs(result[1]['x'],result[1]['y']);self.assertIs(result[0],result[2])
        ordered=OrderedDict([('x',[1,2])]);cloned=deepcopy(ordered)
        self.assertIs(type(cloned),OrderedDict);self.assertIsNot(cloned['x'],ordered['x'])

    def test_explicit_memo_and_subclasses_keep_standard_semantics(self):
        marker=object();self.assertIs(deepcopy(1,{id(1):marker}),marker)
        class MutableText(str):pass
        value=MutableText('a');value.extra=[]
        result=deepcopy({'x':value})['x']
        self.assertEqual(result,value);self.assertIsNot(result,value)
        self.assertIsNot(result.extra,value.extra)


if __name__=='__main__':unittest.main()
