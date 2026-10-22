import copy
import unittest
from collections import OrderedDict

from reflex.cloning import deepcopy
from reflex.core import digest
from reflex.examples import context,action,effect


class CloneTests(unittest.TestCase):
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
