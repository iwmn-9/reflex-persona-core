import subprocess
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.finish_adverse_recovery import execute, finish, ROOT
from tools.owned_checkpoint import encoded, sha


class AdverseCollectionTests(unittest.TestCase):
    def test_unknown_dispatch_is_never_retried(self):
        failed=subprocess.CompletedProcess([],1,'','connection lost after dispatch')
        with patch('tools.finish_adverse_recovery.subprocess.run',return_value=failed) as call:
            with self.assertRaises(RuntimeError):execute('gh','workflow','run','adverse-recovery.yml')
        self.assertEqual(call.call_count,1)

    def test_read_failure_can_recover(self):
        failed=subprocess.CompletedProcess([],1,'','temporary read error')
        success=subprocess.CompletedProcess([],0,'{"status":"completed"}','')
        with patch('tools.finish_adverse_recovery.subprocess.run',side_effect=[failed,success]) as call, patch('tools.finish_adverse_recovery.time.sleep'):
            self.assertEqual(execute('gh','api','repos/example/actions/runs/1'),success.stdout)
        self.assertEqual(call.call_count,2)

    def test_complete_state_requires_requested_full_audit(self):
        plan_path=ROOT/'evidence/long_horizon/adverse_recovery/plan.json'
        plan=json.loads(plan_path.read_bytes())
        names={f"adverse-{s['id']}-{arm}" for s in plan['specs'] for arm in plan['arms']}
        report=dict(complete=True,pairs_complete=10,progress=[],development_gate_passed=True)
        def read_only(gh,*args):
            self.assertEqual(args[0],'api','completion must not dispatch another study')
            route=args[1]
            if '/git/ref/' in route:return json.dumps(dict(object=dict(type='commit',sha='frozen')))
            if '/artifacts?' in route:return '{"artifacts":[]}'
            if '/jobs?' in route:return '{"jobs":[]}'
            return json.dumps(dict(head_sha='frozen',status='completed',conclusion='success'))
        for failed in (False,True):
            with self.subTest(audit_failure=failed),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary)
                (root/'state.json').write_bytes(encoded(dict(initial_run=1,current_run=1,
                    source_ref='frozen',segments=[1],collected={n:'retained' for n in names},
                    complete=False,study_plan_sha256=sha(plan_path.read_bytes()))))
                with patch('tools.finish_adverse_recovery.execute',side_effect=read_only), \
                     patch('tools.finish_adverse_recovery.summarize',return_value=report), \
                     patch('tools.analyze_adverse_results.analyze',return_value={}), \
                     patch('tools.archive_adverse_study.archive',side_effect=AssertionError('bad history') if failed else None) as audit:
                    if failed:
                        with self.assertRaises(AssertionError):
                            finish(root,1,'frozen','gh',1,'tag',archive_results=True)
                    else:finish(root,1,'frozen','gh',1,'tag',archive_results=True)
                audit.assert_called_once_with(root,plan_path,'frozen',root/'complete_audit')
                state=json.loads((root/'state.json').read_bytes())
                self.assertEqual(state['complete'],not failed)
                self.assertEqual(state['phase'],'needs_attention' if failed else 'complete_audited_needs_review')


if __name__=='__main__':unittest.main()
