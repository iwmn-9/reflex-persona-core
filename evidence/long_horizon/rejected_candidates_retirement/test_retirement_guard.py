import os
import unittest
from unittest.mock import patch
import audit_retired_study as audit

class Guard(unittest.TestCase):
    def test_auditor_does_not_block_itself(self):
        with patch.object(audit,'processes',return_value=[dict(ProcessId=os.getpid(),CommandLine='python audit.py --manifest C:/task/manifest.json')]):
            audit.inactive('C:/task/manifest.json','unused')
    def test_registered_worker_remains_forbidden(self):
        with patch.object(audit,'processes',return_value=[dict(ProcessId=os.getpid()+1,CommandLine='python driver.py --manifest C:/task/manifest.json --job 1')]):
            with self.assertRaises(ValueError):audit.inactive('C:/task/manifest.json','unused')
    def test_registered_coordinator_remains_forbidden(self):
        with patch.object(audit,'processes',return_value=[dict(ProcessId=os.getpid()+1,CommandLine='python driver.py --manifest C:/task/manifest.json --workers 4')]):
            with self.assertRaises(ValueError):audit.inactive('C:/task/manifest.json','unused')
    def test_other_study_keeps_running(self):
        with patch.object(audit,'processes',return_value=[dict(ProcessId=os.getpid()+1,CommandLine='python driver.py --manifest C:/other/manifest.json --job 1')]):
            audit.inactive('C:/task/manifest.json','unused')

if __name__=='__main__':unittest.main()
