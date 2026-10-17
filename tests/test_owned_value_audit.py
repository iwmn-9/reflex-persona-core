"""The public study counts must come from registration, not forecast rows."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from tools.audit_owned_value import audit
from tools.run_owned_value import run


class OwnedAuditTests(unittest.TestCase):
    def test_registered_run_count_is_numeric_with_forecasts(self):
        job=('auction',0,11,'outside',2,1,1,0)
        row=run(job,'endpoint')
        self.assertTrue(any('forecast_error' in t for t in row['trace']))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            source=Path(__file__).resolve().parents[1]/'tools/run_owned_value.py'
            copied=root/'_source/tools/run_owned_value.py'
            copied.parent.mkdir(parents=True)
            copied.write_bytes(source.read_bytes())
            (root/'source_snapshot.json').write_text(json.dumps({
                'tools/run_owned_value.py':hashlib.sha256(copied.read_bytes()).hexdigest()}))
            (root/'registration.json').write_text(json.dumps({'jobs':[job],'arms':['endpoint']}))
            (root/'trajectories.jsonl').write_text(json.dumps(row)+'\n')
            with contextlib.redirect_stdout(io.StringIO()):
                result=audit(root)
            self.assertIs(type(result['planned_runs']),int)
            self.assertEqual(result['planned_runs'],result['completed_runs'])
            self.assertTrue(result['complete'])
            self.assertEqual(json.loads((root/'audit.json').read_text())['planned_runs'],1)


if __name__=='__main__':unittest.main()
