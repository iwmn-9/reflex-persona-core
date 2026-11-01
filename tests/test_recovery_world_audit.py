import json
from pathlib import Path
import tempfile
import unittest

from tools.owned_checkpoint import encoded, sha
from tools.review_recovery_worlds import execution
from tools.run_adverse_recovery import run


class RecoveryWorldAuditTests(unittest.TestCase):
    def test_replay_and_independent_profit_recomputation(self):
        spec=dict(id='world-audit',mechanism='world',game='resources',profile=1,
                  seed=601,regime='outside',turns=2)
        reg=dict(plan_sha256='test',source_sha256='test')
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            result=run(folder,spec,'expected',registration=reg,continuation_scope='shared')
            envelope=json.loads((folder/'checkpoint.json').read_bytes());payload=envelope['payload']
            manifest=dict(sources=dict(payload['contract']['sources']))
            manifest['sources']['tools/run_adverse_recovery.py']=result['configuration']['runner_sha256']
            manifest['sources'].update({'tools/'+n:v for n,v in result['configuration']['adapters'].items()})
            audited=execution(folder,spec,'expected','shared',manifest,reg)
            self.assertEqual(audited['final_owner'],result['final_owner'])
            result['trace'][0]['conditional_increment']+=.1
            payload['trace']=result['trace']
            result['conditional_increment']=sum(t['conditional_increment'] for t in result['trace'])
            envelope['sha256']=sha(encoded(payload))
            (folder/'checkpoint.json').write_bytes(encoded(envelope))
            (folder/'result.json').write_bytes(encoded(result))
            with self.assertRaises(AssertionError):execution(folder,spec,'expected','shared',manifest,reg)


if __name__=='__main__':unittest.main()
