import copy
import json
from pathlib import Path
import tempfile
import unittest
from tools.probe_frozen_sample_budget import inputs


class FrozenSampleProbeTests(unittest.TestCase):
    def test_requires_complete_original_world_and_predeclared_banks(self):
        result=dict(complete=True,configuration=dict(spec=dict(turns=2),arm='sample8'),
                    replayed_actual_steps=2,trace=[{},{}])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'result.json';path.write_text(json.dumps(result),encoding='utf8')
            self.assertEqual(inputs(path,1,9101,8)[1],tuple(range(9101,9109)))
            for tick,start,samples in ((2,701,3),(0,500,3),(0,701,4)):
                with self.assertRaises(ValueError):inputs(path,tick,start,samples)
            partial=copy.deepcopy(result);partial['trace'].pop()
            path.write_text(json.dumps(partial),encoding='utf8')
            with self.assertRaises(ValueError):inputs(path,0,701,3)


if __name__=='__main__':unittest.main()
