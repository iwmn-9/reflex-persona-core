"""Exercise real Windows handle sharing, not a mocked successful rename."""
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest

from tools.local_checkpoint_io import atomic_json, read_json


class SnapshotIOTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'nt', 'Windows sharing conflict')
    def test_reproduces_nonsharing_reader_and_recovers_after_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'checkpoint.json'
            atomic_json(path, {'tick': 1})
            blocked = path.open('rb')
            try:
                with self.assertRaises(PermissionError):
                    atomic_json(path, {'tick': 2}, sharing_timeout=0)
                release = threading.Thread(target=lambda: (time.sleep(.1), blocked.close()))
                release.start()
                try:
                    atomic_json(path, {'tick': 2})
                finally:
                    release.join()
                self.assertEqual(read_json(path), {'tick': 2})
            finally:
                blocked.close()

    @unittest.skipUnless(os.name == 'nt', 'Windows sharing conflict')
    def test_persistent_conflict_propagates_and_keeps_original_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'checkpoint.json'
            atomic_json(path, {'tick': 1})
            with path.open('rb'):
                with self.assertRaises(PermissionError):
                    atomic_json(path, {'tick': 2}, sharing_timeout=.03)
                self.assertEqual(read_json(path), {'tick': 1})
                self.assertEqual(read_json(path.with_suffix('.new')), {'tick': 2})

    def test_invalid_json_is_not_silently_recovered(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'checkpoint.json'
            path.write_bytes(b'{"tick":')
            with self.assertRaises(json.JSONDecodeError):
                read_json(path)


if __name__ == '__main__':
    unittest.main()
