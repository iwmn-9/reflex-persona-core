import hashlib,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from tools.freeze_experiment import freeze
from tools.resume_horizon_pair import prepare,run_pair
from tools.run_owned_value import run


class PairContinuationTests(unittest.TestCase):
    def fixture(self,root,game='gambling',tails=(1,2)):
        jobs=[[game,1,52997,'shift-return',5,1,tail,0] for tail in tails]
        pair=dict(version='same-host-horizon-pair-v1',pair_id='original-host',jobs=jobs,
            profile='steady',stage='diagnostic',recovery_limit=16,complete=False)
        inputs=[]
        for index,job in enumerate(jobs):
            child=root/f'input{index}';freeze(child);inputs.append(str(child))
            (child/'pair_execution.json').write_text(json.dumps(pair))
            self.advance(child,job,index+1,register=True)
        return inputs,jobs,*self.hashes(inputs)

    def hashes(self,inputs):
        return [[hashlib.sha256((Path(p)/name).read_bytes()).hexdigest() for p in inputs]
                for name in ('source_snapshot.json','checkpoint.json')]

    def advance(self,child,job,count,register=False):
        program='''import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from tools.run_owned_value import run
from tools.run_horizon_study import __doc__ as semantics
child=Path(sys.argv[2]);job=json.loads(sys.argv[3])
if sys.argv[5]=='1':
 (child/'registration.json').write_text(json.dumps(dict(jobs=[job],arms=['receding_recovery'],world_seeds=[job[2]],stage='diagnostic',semantics=semantics)))
assert run(job,'receding_recovery',checkpoint=child/'checkpoint.json',max_decisions=int(sys.argv[4])) is None
'''
        subprocess.run([sys.executable,'-c',program,str(child/'_source'),str(child),json.dumps(job),str(count),str(int(register))],check=True,capture_output=True)

    def test_common_boundary_continuation_matches_uninterrupted_decisions(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inputs,jobs,manifests,checkpoints=self.fixture(root,tails=(15,31))
            first=Path(inputs[0])
            subprocess.run([sys.executable,str(first/'_source/tools/run_horizon_study.py'),
                '--root',str(first),'--job',json.dumps(jobs[0]),'--stage','diagnostic','--frozen'],check=True,capture_output=True)
            manifests,checkpoints=self.hashes(inputs)
            before=[(Path(p)/'checkpoint.json').read_bytes() for p in inputs]
            proof=run_pair(root/'continued',inputs,jobs,manifests,checkpoints)
            self.assertTrue(proof['complete']);self.assertEqual(proof['continuation']['boundary'],2)
            self.assertEqual([(s['start'],s['end']) for s in proof['host_segments']],[(0,2),(2,5)])
            for index,job in enumerate(jobs):
                child=root/'continued'/f'h{job[6]+1}'
                self.assertEqual((Path(inputs[index])/'checkpoint.json').read_bytes(),before[index])
                actual=json.loads((child/'trajectories.jsonl').read_text());expected=run(job,'receding_recovery')
                actual.pop('seconds');expected.pop('seconds');self.assertEqual(actual,expected)
                boundary=json.loads((child/'boundary_provenance.json').read_text())
                self.assertEqual(boundary['original_turns'],(5,2)[index])
                self.assertEqual(boundary['original_checkpoint_sha256'],checkpoints[index])
                self.assertEqual(json.loads((child/'pair_execution.json').read_text()),proof)
            with self.assertRaises(FileExistsError):run_pair(root/'continued',inputs,jobs,manifests,checkpoints)
            self.assertTrue(all('runtime' in s for s in proof['host_segments']))
            import shutil
            from tools.summarize_horizon_study import summarize
            specs=[]
            for job in jobs:
                key=f'h{job[6]+1}';specs.append(dict(key=key,job=job))
                shutil.copytree(root/'continued'/key,root/'collected'/('horizon-'+key))
            plan=dict(retained_baselines=[],new_jobs=specs,stage='diagnostic',execution_pairing=True)
            result=summarize(Path(__file__).resolve().parents[1],root/'collected',plan)
            self.assertTrue(result['complete']);self.assertEqual(len(result['comparisons']),1)
            self.assertEqual(result['comparisons'][0]['execution_host_segments'],proof['host_segments'])
            for index,job in enumerate(jobs):
                shutil.copytree(inputs[index],root/'original'/f'horizon-h{job[6]+1}')
            before=summarize(root,root/'original',plan,partial=True)
            self.assertFalse(before['complete']);self.assertEqual(before['collected_runs'],0)
            combined=summarize(root,root/'original',plan,additional_roots=(root/'collected',))
            self.assertEqual(combined,result)
            file=root/'collected/horizon-h16/boundary_provenance.json'
            boundary=json.loads(file.read_text());boundary['retained_prefix_sha256']='0'*64;file.write_text(json.dumps(boundary))
            with self.assertRaises(AssertionError):summarize(root,root/'collected',plan)

    def test_repeated_interruption_preserves_host_intervals_and_owner_history(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inputs,jobs,manifests,checkpoints=self.fixture(root,'resources')
            prepare(root/'first',inputs,jobs,manifests,checkpoints)
            intermediate=[str(root/'first'/f'h{j[6]+1}') for j in jobs]
            for index,job in enumerate(jobs):self.advance(Path(intermediate[index]),job,index+1)
            proof=run_pair(root/'second',intermediate,jobs,*self.hashes(intermediate))
            self.assertEqual([(s['start'],s['end']) for s in proof['host_segments']],[(0,1),(1,2),(2,5)])
            for job in jobs:
                actual=json.loads((root/'second'/f'h{job[6]+1}'/'trajectories.jsonl').read_text())
                expected=run(job,'receding_recovery');actual.pop('seconds');expected.pop('seconds')
                self.assertEqual(actual,expected)

    def test_rejects_unmatched_pair_and_changed_input_before_writing(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inputs,jobs,manifests,checkpoints=self.fixture(root)
            with self.assertRaisesRegex(AssertionError,'wrong original checkpoint'):
                prepare(root/'invalid',inputs,jobs,manifests,['0'*64,checkpoints[1]])
            self.assertFalse((root/'invalid').exists())
            with self.assertRaisesRegex(AssertionError,'outside original inputs'):
                prepare(Path(inputs[0])/'nested',inputs,jobs,manifests,checkpoints)
            self.assertFalse((Path(inputs[0])/'nested').exists())
            from unittest.mock import patch
            with patch('tools.resume_horizon_pair.platform.system',return_value='changed-platform'):
                with self.assertRaisesRegex(AssertionError,'continuation runtime changed'):
                    prepare(root/'invalid',inputs,jobs,manifests,checkpoints)
            self.assertFalse((root/'invalid').exists())
            file=Path(inputs[1])/'pair_execution.json';pair=json.loads(file.read_text());pair['pair_id']='unrelated-host';file.write_text(json.dumps(pair))
            with self.assertRaisesRegex(AssertionError,'different original pair'):
                prepare(root/'invalid',inputs,jobs,manifests,checkpoints)
            self.assertFalse((root/'invalid').exists())


if __name__=='__main__':unittest.main()
