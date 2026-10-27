import subprocess
import unittest
from unittest.mock import patch

from tools.finish_adverse_recovery import execute


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


if __name__=='__main__':unittest.main()
