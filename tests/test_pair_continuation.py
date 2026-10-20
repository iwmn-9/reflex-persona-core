import hashlib,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from tools.freeze_experiment import freeze
from tools.resume_horizon_pair import prepare,run_pair,verify_segmented_result
from tools.run_owned_value import run


class PairContinuationTests(unittest.TestCase):
    def fixture(self,root,game='gambling',tails=(1,2),*,value_contract=False):
        jobs=[[game,1,52997,'shift-return',5,1,tail,0] for tail in tails]
        pair=dict(version='same-host-horizon-pair-v1',pair_id='original-host',jobs=jobs,
            profile='steady',stage='diagnostic',recovery_limit=16,complete=False)
        if value_contract:
            pair=dict(version='same-host-value-contract-v1',pair_id='original-host',job=jobs[1],complete=False,
                configurations={'baseline':{},'candidate':{'cumulative_achievement':True}},plan_sha256='a'*64)
            if jobs[0]!=jobs[1]:pair['jobs']=dict(zip(('baseline','candidate'),jobs))
        inputs=[]
        for index,job in enumerate(jobs):
            child=root/f'input{index}';freeze(child);inputs.append(str(child))
            if value_contract:
                snapshot=json.loads((child/'source_snapshot.json').read_text())
                pair['sources']={k:v for k,v in snapshot.items() if k.endswith('.py')}
            (child/('contract_pair.json' if value_contract else 'pair_execution.json')).write_text(json.dumps(pair))
            self.advance(child,job,index+1,register=True,cumulative=value_contract and index==1)
        return inputs,jobs,*self.hashes(inputs)

    def hashes(self,inputs):
        return [[hashlib.sha256((Path(p)/name).read_bytes()).hexdigest() for p in inputs]
                for name in ('source_snapshot.json','checkpoint.json')]

    def advance(self,child,job,count,register=False,cumulative=False):
        program='''import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from tools.run_owned_value import run
from tools.run_horizon_study import __doc__ as semantics
child=Path(sys.argv[2]);job=json.loads(sys.argv[3])
if sys.argv[5]=='1':
 reg=dict(jobs=[job],arms=['receding_recovery'],world_seeds=[job[2]],stage='diagnostic',semantics=semantics)
 if sys.argv[6]=='1':reg['configuration']={'cumulative_achievement':True}
 (child/'registration.json').write_text(json.dumps(reg))
configuration=json.loads((child/'registration.json').read_text()).get('configuration',{})
assert run(job,'receding_recovery',checkpoint=child/'checkpoint.json',max_decisions=int(sys.argv[4]),**configuration) is None
'''
        subprocess.run([sys.executable,'-c',program,str(child/'_source'),str(child),json.dumps(job),str(count),str(int(register)),str(int(cumulative))],check=True,capture_output=True)

    def test_value_contract_pair_retains_distinct_owners_and_cumulative_configuration(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inputs,jobs,manifests,checkpoints=self.fixture(root,'resources',tails=(2,2),value_contract=True)
            with self.assertRaises(AssertionError):
                prepare(root/'swapped',inputs[::-1],jobs,manifests[::-1],checkpoints[::-1],value_contract=True)
            self.assertFalse((root/'swapped').exists())
            proof=run_pair(root/'continued',inputs,jobs,manifests,checkpoints,value_contract=True)
            self.assertTrue(proof['complete']);self.assertEqual(proof['continuation']['boundary'],1)
            for enabled,name in enumerate(('baseline','candidate')):
                child=root/'continued'/name;actual=json.loads((child/'trajectories.jsonl').read_text())
                expected=run(jobs[0],'receding_recovery',cumulative_achievement=bool(enabled))
                actual.pop('seconds');expected.pop('seconds');self.assertEqual(actual,expected)
                self.assertEqual(json.loads((child/'contract_pair.json').read_text()),proof)
                verify_segmented_result(child,actual,proof,enabled)
            # Both worlds can finish before a killed coordinator writes its
            # final pair marker. Re-audit them without adding real decisions.
            proof['complete']=False;proof['host_segments'][-1]['end']=None
            finished=[str(root/'continued'/name) for name in ('baseline','candidate')]
            for p in finished:(Path(p)/'contract_pair.json').write_text(json.dumps(proof))
            final=run_pair(root/'finalized',finished,jobs,*self.hashes(finished),value_contract=True)
            self.assertTrue(final['continuation']['no_new_decisions'])
            self.assertEqual(final['continuation']['boundary'],5)
            for name in ('baseline','candidate'):
                before=json.loads((root/'continued'/name/'trajectories.jsonl').read_text())
                after=json.loads((root/'finalized'/name/'trajectories.jsonl').read_text())
                before.pop('seconds');after.pop('seconds');self.assertEqual(before,after)
                verify_segmented_result(root/'finalized'/name,after,final,int(name=='candidate'))
            bad=root/'finalized/candidate/boundary_provenance.json'
            boundary=json.loads(bad.read_text());boundary['retained_prefix_sha256']='0'*64;bad.write_text(json.dumps(boundary))
            with self.assertRaises(AssertionError):verify_segmented_result(bad.parent,after,final,1)

    def test_value_contract_resume_preserves_each_registered_horizon(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inputs,jobs,manifests,checkpoints=self.fixture(root,tails=(1,2),value_contract=True)
            proof=run_pair(root/'continued',inputs,jobs,manifests,checkpoints,value_contract=True)
            self.assertEqual(proof['jobs'],dict(zip(('baseline','candidate'),jobs)))
            self.assertEqual(proof['job'],jobs[1])
            for index,name in enumerate(('baseline','candidate')):
                child=root/'continued'/name;actual=json.loads((child/'trajectories.jsonl').read_text())
                expected=run(jobs[index],'receding_recovery',cumulative_achievement=bool(index))
                actual.pop('seconds');expected.pop('seconds');self.assertEqual(actual,expected)
                verify_segmented_result(child,actual,proof,index)

    def test_guarded_continuation_retains_complete_faster_arm_history(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            inputs,jobs,_,_=self.fixture(root,'resources',tails=(2,2),value_contract=True)
            first=Path(inputs[0])
            subprocess.run([sys.executable,str(first/'_source/tools/run_horizon_study.py'),
                '--root',str(first),'--job',json.dumps(jobs[0]),'--stage','diagnostic','--frozen'],
                check=True,capture_output=True)
            originals=[(Path(p)/'checkpoint.json').read_bytes() for p in inputs]
            proof=run_pair(root/'continued',inputs,jobs,*self.hashes(inputs),
                           value_contract=True,guard_history=True,recorded_predictions=True)
            self.assertTrue(proof['complete'])
            self.assertTrue(proof['retained_history_required'])
            self.assertEqual(proof['continuation']['boundary'],2)
            for index,name in enumerate(('baseline','candidate')):
                child=root/'continued'/name
                actual=json.loads((child/'trajectories.jsonl').read_bytes())
                verify_segmented_result(child,actual,proof,index)
                self.assertEqual((Path(inputs[index])/'checkpoint.json').read_bytes(),originals[index])
                self.assertEqual((child/'retained_execution.json').read_bytes(),originals[index])
            from tools.guard_retained_history import check_step
            original=json.loads(originals[0])['payload']['trace'][2]
            for field in ('action','owner','before_owner','law','after'):
                changed=dict(original);changed[field]='changed'
                with self.assertRaisesRegex(AssertionError,'retained experience changed'):
                    check_step(changed,original)
            receipt=root/'continued/baseline/retained_history_replay.json'
            content=json.loads(receipt.read_bytes());content['checked_until']-=1
            receipt.write_text(json.dumps(content))
            with self.assertRaises(AssertionError):
                verify_segmented_result(root/'continued/baseline',
                    json.loads((root/'continued/baseline/trajectories.jsonl').read_bytes()),proof,0)

    def test_guard_survives_another_interruption_before_old_history_is_replayed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            inputs,jobs,_,_=self.fixture(root,'resources',tails=(2,2),value_contract=True)
            first=Path(inputs[0])
            subprocess.run([sys.executable,str(first/'_source/tools/run_horizon_study.py'),
                '--root',str(first),'--job',json.dumps(jobs[0]),'--stage','diagnostic','--frozen'],
                check=True,capture_output=True)
            original=(first/'checkpoint.json').read_bytes()
            prepare(root/'interrupted',inputs,jobs,*self.hashes(inputs),value_contract=True,guard_history=True)
            intermediate=[str(root/'interrupted'/name) for name in ('baseline','candidate')]
            for index,path in enumerate(intermediate):
                self.advance(Path(path),jobs[index],1,cumulative=bool(index))
            proof=run_pair(root/'continued',intermediate,jobs,*self.hashes(intermediate),
                           value_contract=True,guard_history=True)
            self.assertEqual(proof['continuation']['boundary'],3)
            self.assertEqual((root/'continued/baseline/retained_execution.json').read_bytes(),original)
            for index,name in enumerate(('baseline','candidate')):
                child=root/'continued'/name
                verify_segmented_result(child,json.loads((child/'trajectories.jsonl').read_bytes()),proof,index)

    def test_guard_normalizes_only_bounded_factual_forecasts_without_forcing_actions(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            inputs,jobs,_,_=self.fixture(root,'resources',tails=(2,2),value_contract=True)
            first=Path(inputs[0])
            subprocess.run([sys.executable,str(first/'_source/tools/run_horizon_study.py'),
                '--root',str(first),'--job',json.dumps(jobs[0]),'--stage','diagnostic','--frozen'],
                check=True,capture_output=True)
            proof=prepare(root/'continued',inputs,jobs,*self.hashes(inputs),value_contract=True,guard_history=True)
            child=root/'continued/baseline'
            script=Path(__file__).resolve().parents[1]/'tools/guard_retained_history.py'
            program='''import math,runpy,sys
from pathlib import Path
child=Path(sys.argv[1]);guard=sys.argv[2];job=sys.argv[3]
sys.path.insert(0,str(child/'_source'))
from reflex.event_reader import EventReader
original=EventReader.predict
def rounded(self,c):
 result=original(self,c);key=next(iter(result))
 result[key]=math.nextafter(result[key],math.inf)
 return result
EventReader.predict=rounded
sys.argv=[guard,str(child/'_source/tools/run_horizon_study.py'),'--root',str(child),'--job',job,'--stage','diagnostic','--frozen']
runpy.run_path(guard,run_name='__main__')
'''
            result=subprocess.run([sys.executable,'-c',program,str(child),str(script),json.dumps(jobs[0])],
                                  capture_output=True,text=True)
            # A strict final audit may reject the injected rounding; it must
            # not stop guarded actual execution or corrupt its complete trace.
            if result.returncode:
                self.assertIn('stored actual action/feedback replay drift',result.stderr)
            from tools.guard_retained_history import verify
            row=json.loads((child/'trajectories.jsonl').read_bytes())
            receipt=verify(child,row,json.loads((child/'boundary_provenance.json').read_bytes()),
                           proof['continuation']['inputs'][0]['retained_checkpoint_sha256'])
            self.assertTrue(receipt['normalizations'])
            self.assertTrue(all(2<=item['tick']<5 for item in receipt['normalizations']))

    def test_value_contract_cli_resumes_from_outside_repository_without_pythonpath(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inputs,jobs,manifests,checkpoints=self.fixture(root,value_contract=True)
            originals=[(Path(p)/'checkpoint.json').read_bytes() for p in inputs]
            target=root/'continued';script=Path(__file__).resolve().parents[1]/'tools/resume_horizon_pair.py'
            env=os.environ.copy();env.pop('PYTHONPATH',None)
            command=[sys.executable,str(script),'--root',str(target),'--value-contract']
            for name,value in (('inputs',inputs),('jobs',jobs),('manifests',manifests),('checkpoints',checkpoints)):
                command+=['--'+name,json.dumps(value)]
            result=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,encoding='utf8')
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            proof=json.loads((target/'contract_pair.json').read_text());self.assertTrue(proof['complete'])
            self.assertEqual([(Path(p)/'checkpoint.json').read_bytes() for p in inputs],originals)
            self.assertEqual(proof['jobs'],dict(zip(('baseline','candidate'),jobs)))
            for index,name in enumerate(('baseline','candidate')):
                child=target/name;actual=json.loads((child/'trajectories.jsonl').read_text())
                expected=run(jobs[index],'receding_recovery',cumulative_achievement=bool(index))
                actual.pop('seconds');expected.pop('seconds');self.assertEqual(actual,expected)
                verify_segmented_result(child,actual,proof,index)

    def test_common_boundary_continuation_matches_uninterrupted_decisions(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inputs,jobs,manifests,checkpoints=self.fixture(root,tails=(15,31))
            # A killed original coordinator never writes its final marker.
            for source in inputs:
                marker=Path(source)/'pair_execution.json';record=json.loads(marker.read_text())
                record.pop('complete');marker.write_text(json.dumps(record))
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
