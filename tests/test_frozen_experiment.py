import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.freeze_experiment import dispatch


class FrozenExperimentTests(unittest.TestCase):
    def test_child_runs_from_detached_bytes_and_cannot_overwrite_registration(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'study'
            with patch('tools.freeze_experiment.subprocess.run') as run:
                dispatch(root,'run_supported_comparison.py',['--start','7000','--seeds','4'])
            args=run.call_args.args[0];snapshot=root/'_source'
            # Windows runners may expose the temp directory via its 8.3 alias.
            self.assertEqual(Path(args[1]).resolve(),(snapshot/'tools/run_supported_comparison.py').resolve())
            self.assertIn('--frozen',args)
            records=json.loads((root/'source_snapshot.json').read_text(encoding='utf-8'))
            self.assertTrue(records)
            self.assertTrue(all(hashlib.sha256((snapshot/name).read_bytes()).hexdigest()==sha for name,sha in records.items()))
            with patch('tools.freeze_experiment.subprocess.run') as run:
                with self.assertRaises(FileExistsError):dispatch(root,'run_supported_comparison.py',[])
                run.assert_not_called()


if __name__=='__main__':unittest.main()
