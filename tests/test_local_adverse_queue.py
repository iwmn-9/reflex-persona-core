"""Local orchestration must not manufacture progress or duplicate workers."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.local_adverse_queue import encoded, exclusive, progress, sha


class LocalQueueContractTests(unittest.TestCase):
    def test_checkpoint_progress_is_visible_without_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = dict(trace=[{'event': 1}], world={'tick': 1}, owner={'last_tick': 0})
            (root/'checkpoint.json').write_bytes(encoded(dict(payload=payload, sha256=sha(encoded(payload)))))
            (root/'status.json').write_text(json.dumps(dict(state='running', actual_steps=0, complete=False)))
            result = progress(root)
            self.assertEqual(result['actual_steps'], 1)
            self.assertFalse(result['complete'])

    def test_corrupted_checkpoint_does_not_become_progress(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'checkpoint.json').write_bytes(encoded(dict(payload={}, sha256='wrong')))
            with self.assertRaises(AssertionError):
                progress(root)

    def test_mismatched_owner_timeline_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = dict(trace=[{}], world={'tick': 1}, owner={'last_tick': -1})
            (root/'checkpoint.json').write_bytes(encoded(dict(payload=payload, sha256=sha(encoded(payload)))))
            with self.assertRaises(AssertionError):
                progress(root)

    @unittest.skipUnless(sys.platform == 'win32', 'Windows byte-range lock')
    def test_live_worker_lock_blocks_another_process_and_releases(self):
        with tempfile.TemporaryDirectory() as temp:
            lock = Path(temp)/'worker.lock'
            code = 'from tools.local_adverse_queue import exclusive; import sys\nwith exclusive(sys.argv[1]): pass'
            with exclusive(lock):
                blocked = subprocess.run([sys.executable, '-c', code, str(lock)], capture_output=True)
                self.assertNotEqual(blocked.returncode, 0)
            available = subprocess.run([sys.executable, '-c', code, str(lock)], capture_output=True)
            self.assertEqual(available.returncode, 0, available.stderr)


if __name__ == '__main__':
    unittest.main()
