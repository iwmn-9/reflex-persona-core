import copy
import json
import os
from pathlib import Path
import shutil
import runpy
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from tools.horizon_segment import (PAUSED,RECEIPT,SegmentPaused,bounded_command,
    handoff_facts,read,save,sha,verify_handoff,write_handoff)
from tools.resume_horizon_pair import prepare,run_pair,verify_segmented_result
from tests import test_pair_continuation as fixtures


class HorizonSegmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.base=Path(cls.temp.name)
        cls.fixture=fixtures.PairContinuationTests()
        inputs,cls.jobs,manifests,checkpoints=cls.fixture.fixture(cls.base,value_contract=True)
        cls.partial=cls.base/'partial'
        header=prepare(cls.partial,inputs,cls.jobs,manifests,checkpoints,value_contract=True,guard_history=True)
        for i,name in enumerate(('baseline','candidate')):
            child=cls.partial/name
            cls.fixture.advance(child,cls.jobs[i],3,cumulative=bool(i))
            retained=read(child/'retained_execution.json')
            save(child/'retained_history_replay.json',dict(version='retained-history-replay-v1',start=1,
                checked_until=len(retained['payload']['trace']),complete=True,normalizations=[],
                retained_checkpoint_sha256=sha(child/'retained_execution.json')))
            save(child/'execution_status.json',dict(pid=1,return_code=PAUSED,ended_at='fixture'))
            save(child/'segment_interruption.json',dict(version='horizon-interruption-v1',
                termination='budget_exhausted_before_launch',child_return_code=None))
        header.update(return_codes=[PAUSED,PAUSED])
        for d in [cls.partial,*[cls.partial/n for n in ('baseline','candidate')]]:
            save(d/'contract_pair.json',header)
        cls.originals=[(Path(p)/'checkpoint.json').read_bytes() for p in inputs]

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def fresh(self):
        context=tempfile.TemporaryDirectory();self.addCleanup(context.cleanup)
        target=Path(context.name)/'pair';shutil.copytree(self.partial,target)
        return target

    def test_handoff_retains_history_and_resume_reaches_same_final_decisions(self):
        root=self.fresh();dirs=[root/n for n in ('baseline','candidate')]
        with patch.dict(os.environ,{'GITHUB_RUN_ID':'123','GITHUB_RUN_ATTEMPT':'1'}):
            receipt=write_handoff(root,self.jobs)
        self.assertFalse(receipt['experiment_complete'])
        self.assertEqual((receipt['from_tick'],receipt['to_tick']),(1,4))
        self.assertEqual(verify_handoff(dirs,self.jobs,run=123,plan_sha='a'*64),receipt)
        with self.assertRaises(AssertionError):verify_handoff(dirs,self.jobs,run=124)
        continued=root.parent/'continued'
        proof=run_pair(continued,list(map(str,dirs)),self.jobs,*self.fixture.hashes(dirs),value_contract=True,guard_history=True)
        self.assertTrue(proof['complete']);self.assertEqual(proof['return_codes'],[0,0])
        from tools.run_owned_value import run
        for i,name in enumerate(('baseline','candidate')):
            row=read(continued/name/'trajectories.jsonl')
            expected=run(self.jobs[i],'receding_recovery',cumulative_achievement=bool(i))
            row.pop('seconds');expected.pop('seconds');self.assertEqual(row,expected)
            verify_segmented_result(continued/name,row,proof,i)
            self.assertFalse((continued/name/RECEIPT).exists())
        # A failed strict audit whose recovery hits the budget is still a
        # failure; no planned-pause exception may hide the original error.
        from tools.resume_horizon_study import resume
        child=continued/'candidate';audit=child/'audit.json';original=audit.read_bytes();audit.unlink()
        with patch('tools.horizon_segment.bounded_command',side_effect=[subprocess.CalledProcessError(1,[]),SegmentPaused()]):
            with self.assertRaisesRegex(RuntimeError,'original failure remains'):
                resume(child,sha(child/'source_snapshot.json'),self.jobs[1],recorded_predictions=True,deadline=1.)
        audit.write_bytes(original)

    def test_corruption_failure_and_no_progress_are_not_handoffs(self):
        for mutation in ('source','checksum','owner','rng','history','prefix','error','no_progress','stderr'):
            with self.subTest(mutation=mutation):
                root=self.fresh();child=root/'candidate'
                if mutation=='source':
                    path=child/'_source/reflex/core.py';path.write_bytes(path.read_bytes()+b'\n# changed')
                elif mutation in ('checksum','owner','rng','history'):
                    path=child/'checkpoint.json';data=read(path)
                    if mutation=='checksum':data['sha256']='0'*64
                    else:
                        p=data['payload']
                        if mutation=='owner':p['owner']['last_tick']-=1
                        if mutation=='rng':p['rng']['state']['state']+=1
                        if mutation=='history':p['trace'][0]['action']='changed'
                        from tools.resume_horizon_pair import encoded,sha as bytes_sha
                        data['sha256']=bytes_sha(encoded(p))
                    save(path,data)
                elif mutation=='prefix':
                    path=child/'boundary_provenance.json';v=read(path);v['retained_prefix_sha256']='0'*64;save(path,v)
                elif mutation in ('error','no_progress'):
                    v=read(child/'contract_pair.json')
                    if mutation=='error':v['return_codes'][1]=1
                    else:v['continuation']['boundary']=4
                    for d in [root,root/'baseline',child]:save(d/'contract_pair.json',v)
                else:
                    (child/'test.stderr.log').write_text('AssertionError: genuine failure\n')
                    (child/'test.stdout.log').write_text('')
                    save(child/'segment_interruption.json',dict(version='horizon-interruption-v1',
                        termination='requested_sigterm',child_return_code=-15,stage='test',
                        stderr_sha256=sha(child/'test.stderr.log'),stdout_sha256=sha(child/'test.stdout.log')))
                with self.assertRaises(AssertionError):write_handoff(root,self.jobs)
                self.assertFalse((root/RECEIPT).exists())

    def test_receipt_edit_and_forged_complete_do_not_enter_quality_results(self):
        root=self.fresh();dirs=[root/n for n in ('baseline','candidate')]
        receipt=write_handoff(root,self.jobs)
        for field,value in (('to_tick',128),('experiment_complete',True)):
            changed=dict(receipt);changed[field]=value
            for d in dirs:save(d/RECEIPT,changed)
            with self.assertRaises(AssertionError):verify_handoff(dirs,self.jobs)
        # The existing quality reader still requires complete audited worlds.
        from tools.summarize_value_contract import read_run
        with self.assertRaises((FileNotFoundError,AssertionError)):
            read_run(dirs[0],self.jobs[0],False,{'configurations':{'baseline':{},'candidate':{'cumulative_achievement':True}}},'a'*64)

    def test_failed_peer_stays_failed_even_with_a_valid_paused_peer(self):
        root=self.fresh();dirs=[root/n for n in ('baseline','candidate')]
        header=read(root/'contract_pair.json');header['return_codes']=[1,PAUSED]
        for d in [root,*dirs]:save(d/'contract_pair.json',header)
        status=read(dirs[0]/'execution_status.json');status['return_code']=1;save(dirs[0]/'execution_status.json',status)
        facts=handoff_facts(dirs,self.jobs,failed_peer=True)
        self.assertEqual(facts['arms'][0]['return_code'],1)
        with self.assertRaisesRegex(AssertionError,'real failure'):write_handoff(root,self.jobs)
        self.assertFalse((root/RECEIPT).exists())

    def test_audited_faster_arm_does_not_make_the_pair_complete(self):
        root=self.fresh();child=root/'baseline'
        subprocess.run([sys.executable,str(child/'_source/tools/run_horizon_study.py'),
            '--root',str(child),'--job',json.dumps(self.jobs[0]),'--stage','diagnostic','--frozen'],
            check=True,capture_output=True)
        save(child/'resume_provenance.json',dict(complete=True,original_prefix_unchanged=True))
        status=read(child/'execution_status.json');status['return_code']=0;save(child/'execution_status.json',status)
        header=read(root/'contract_pair.json');header['return_codes']=[0,PAUSED]
        for d in [root,child,root/'candidate']:save(d/'contract_pair.json',header)
        receipt=write_handoff(root,self.jobs)
        self.assertEqual([a['ticks'] for a in receipt['arms']],[5,4])
        self.assertFalse(receipt['experiment_complete'])
        audit=read(child/'audit.json');audit['checks']['physical_replay']-=1;save(child/'audit.json',audit)
        with self.assertRaises(AssertionError):verify_handoff([child,root/'candidate'],self.jobs)

    def test_successful_segment_collection_keeps_partial_audit_out_of_completed_results(self):
        root=self.fresh();header=read(root/'contract_pair.json')
        plan=dict(stage='diagnostic',cases=[dict(key='case',job=self.jobs[1])],sources=header['sources'],
            configurations=header['configurations'],horizons={'baseline':2,'candidate':3})
        plan_path=root.parent/'plan.json';save(plan_path,plan);header['plan_sha256']=sha(plan_path)
        for d in [root,root/'baseline',root/'candidate']:save(d/'contract_pair.json',header)
        (root/'candidate/audit.json').write_text('{incomplete final audit write')
        with patch.dict(os.environ,{'GITHUB_RUN_ID':'123'}):write_handoff(root,self.jobs)
        collected=root.parent/'collected';collected.mkdir()
        for name in ('baseline','candidate'):shutil.copytree(root/name,collected/f'value-contract-case-{name}')
        state=dict(id=123,head_sha='b'*40,status='completed',conclusion='success',html_url='https://example.invalid',created_at='t',updated_at='t')
        def api(command,**kwargs):
            return json.dumps({'artifacts':[]} if '/artifacts?' in command[-1] else state).encode()
        args=['collect','--run','123','--sha','b'*40,'--root',str(collected),'--plan',str(plan_path),'--once']
        with patch.object(sys,'argv',args),patch('subprocess.check_output',side_effect=api):
            runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/collect_value_contract.py'),run_name='__main__')
        checks=read(collected/'collection.json')
        self.assertTrue(all(v['complete'] is False and v['pair_complete'] is False for v in checks.values()))
        self.assertEqual(read(collected/'handoffs.json')[0]['state'],'handoff_pending')

    @unittest.skipUnless(os.name=='posix','Actions process-group control is Linux-only')
    def test_real_process_stop_errors_and_deadline_are_distinct(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaises(SegmentPaused):
                bounded_command([sys.executable,'-c','import time; time.sleep(30)'],root,time.monotonic()+.5,'sleep')
            stop=read(root/'segment_interruption.json')
            self.assertEqual(stop['termination'],'requested_sigterm')
            (root/'segment_interruption.json').unlink()
            with self.assertRaises(subprocess.CalledProcessError):
                bounded_command([sys.executable,'-c','raise RuntimeError("real bug")'],root,time.monotonic()+10,'error')
            self.assertFalse((root/'segment_interruption.json').exists())
            with self.assertRaisesRegex(RuntimeError,'stderr'):
                bounded_command([sys.executable,'-c','import sys,time; print("failure",file=sys.stderr,flush=True); time.sleep(30)'],root,time.monotonic()+.5,'stderr')
            self.assertFalse((root/'segment_interruption.json').exists())

    @unittest.skipUnless(os.name=='posix','Actions process-group control is Linux-only')
    def test_real_partial_checkpoint_pause_then_resume(self):
        root=self.fresh();child=root/'candidate'
        # Child has real frozen checkpoint history. Block without altering it;
        # the same process termination/receipt used by Actions must validate.
        with self.assertRaises(SegmentPaused):
            bounded_command([sys.executable,'-c','import time; time.sleep(30)'],child,time.monotonic()+.5,'probe')
        receipt=write_handoff(root,self.jobs)
        self.assertEqual(receipt['to_tick'],4)
        dirs=[root/n for n in ('baseline','candidate')]
        result=run_pair(root.parent/'finish',list(map(str,dirs)),self.jobs,*self.fixture.hashes(dirs),
                        value_contract=True,guard_history=True,segment_seconds=90)
        self.assertTrue(result['complete'])
        # A budget exhausted before any new checkpoint remains a real failure.
        with self.assertRaisesRegex(AssertionError,'no resolved progress'):
            run_pair(root.parent/'no-progress',list(map(str,dirs)),self.jobs,*self.fixture.hashes(dirs),
                     value_contract=True,guard_history=True,segment_seconds=.000001)

    def test_monitor_labels_handoff_without_promoting_it_to_completion(self):
        from tools.observe_horizon_pairs import CheckPublisher,segment_outcome
        root=self.fresh();write_handoff(root,self.jobs)
        parent=root.parent/'observer';parent.mkdir();shutil.copytree(root,parent/'pair0')
        self.assertEqual(segment_outcome(parent,[{'jobs':self.jobs}],[0]),'handoff_pending')
        self.assertEqual(segment_outcome(parent,[{'jobs':self.jobs}],[1]),'failure')
        env={'GITHUB_REPOSITORY':'o/r','GITHUB_RUN_ID':'1','GITHUB_RUN_ATTEMPT':'1','GITHUB_SHA':'a'*40}
        with patch.dict(os.environ,env),patch.object(CheckPublisher,'request',return_value={'id':1}) as request:
            publisher=CheckPublisher('probe')
            publisher.publish({'segment_outcome':'handoff_pending'},finished=True)
            self.assertEqual(request.call_args.args[2]['conclusion'],'neutral')
            self.assertIn('incomplete',request.call_args.args[2]['output']['title'])
            publisher.publish({'segment_outcome':'failure'},finished=True)
            self.assertEqual(request.call_args.args[2]['conclusion'],'failure')


if __name__=='__main__':unittest.main()
