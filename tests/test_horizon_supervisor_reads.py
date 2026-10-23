"""Test the real read helper without starting the supervisor's top-level loop."""
import ast
import json
from pathlib import Path
import subprocess
import time
import unittest
from datetime import datetime,timezone
from types import SimpleNamespace
from unittest.mock import patch


def read_helper():
    root=Path(__file__).resolve().parents[1]
    source=root/'tools/finish_registered_worlds.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    definitions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('api','transient_read')]
    scope=dict(json=json,subprocess=subprocess,time=time,R=root)
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(source),'exec'),scope)
    return scope['api']


class HorizonSupervisorReadTests(unittest.TestCase):
    def test_exhausted_transport_read_keeps_study_pending_then_recovers(self):
        root=Path(__file__).resolve().parents[1]
        source=root/'tools/finish_registered_worlds.py'
        tree=ast.parse(source.read_text(encoding='utf8'))
        loop=next(n for n in tree.body if isinstance(n,ast.Try) and any(isinstance(x,ast.While) for x in n.body))
        state=dict(tasks=[dict(status='active',run=123,key='case',root='unused')],complete=False)
        waits=[];saved=[];calls=[]
        def api(path):
            calls.append(path)
            if len(calls)==1:
                raise subprocess.CalledProcessError(1,['gh','api',path],stderr='dial tcp 1:443: connectex: timeout')
            return {'jobs':[]}
        def sleep(seconds):
            waits.append(seconds)
            if len(waits)==1:
                self.assertFalse(state['complete'])
                self.assertIn('transport_wait',state)
                self.assertNotIn('error',state)
            else:state['tasks'][0]['status']='complete'
        classifier=read_helper().__globals__['transient_read']
        scope=dict(state=state,api=api,transient_read=classifier,R=root,subprocess=subprocess,
            datetime=datetime,timezone=timezone,time=SimpleNamespace(sleep=sleep),
            save=lambda s:saved.append(json.loads(json.dumps(s))))
        with patch('builtins.print'):
            exec(compile(ast.Module(body=[loop],type_ignores=[]),str(source),'exec'),scope)
        self.assertEqual(waits,[60,45])
        self.assertEqual(calls,['actions/runs/123/jobs?per_page=100']*2)
        self.assertTrue(state['complete'])
        self.assertNotIn('transport_wait',state)
        self.assertTrue(any('transport_wait' in s and not s['complete'] for s in saved))

    def test_observed_windows_connect_timeout_retries_the_read(self):
        failed=subprocess.CompletedProcess([],1,'','Get "https://api.github.com": dial tcp 20.27.177.116:443: connectex: A connection attempt failed')
        success=subprocess.CompletedProcess([],0,'{"jobs": []}','')
        with patch.object(subprocess,'run',side_effect=[failed,success]) as run,patch.object(time,'sleep'),patch('builtins.print'):
            self.assertEqual(read_helper()('actions/runs/123/jobs'),{'jobs':[]})
        self.assertEqual(run.call_count,2)

    def test_connection_failure_retries_only_the_same_read(self):
        failed=subprocess.CompletedProcess([],1,'','error connecting to api.github.com\n')
        success=subprocess.CompletedProcess([],0,'{"jobs": []}','')
        with patch.object(subprocess,'run',side_effect=[failed,success]) as run,patch.object(time,'sleep') as sleep,patch('builtins.print'):
            self.assertEqual(read_helper()('actions/runs/123/jobs?per_page=100'),{'jobs':[]})
        self.assertEqual(run.call_count,2)
        self.assertEqual(run.call_args_list[0],run.call_args_list[1])
        command=run.call_args.args[0]
        self.assertEqual(command,['gh','api','repos/iwmn-9/reflex-persona-core/actions/runs/123/jobs?per_page=100'])
        sleep.assert_called_once_with(15)

    def test_persistent_connection_failure_stays_a_failure_after_five_attempts(self):
        failed=subprocess.CompletedProcess([],1,'','error connecting to api.github.com\n')
        with patch.object(subprocess,'run',return_value=failed) as run,patch.object(time,'sleep') as sleep,patch('builtins.print'):
            with self.assertRaises(subprocess.CalledProcessError):
                read_helper()('actions/runs/123/jobs')
        self.assertEqual(run.call_count,5)
        self.assertEqual([c.args[0] for c in sleep.call_args_list],[15,30,45,60])

    def test_authorization_failure_is_not_retried(self):
        failed=subprocess.CompletedProcess([],1,'','HTTP 403: Resource not accessible')
        with patch.object(subprocess,'run',return_value=failed) as run,patch.object(time,'sleep') as sleep:
            with self.assertRaises(subprocess.CalledProcessError):
                read_helper()('actions/runs/123/jobs')
        run.assert_called_once()
        sleep.assert_not_called()
