import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.observe_horizon_pairs import Progress, checkpoint_summary, digest, encoded, descendants


class HorizonObserverTests(unittest.TestCase):
    def test_cpu_activity_is_not_decision_advance_and_idle_flags_without_killing(self):
        obs=Progress(); cp={'ticks':12}; status={'pid':20,'return_code':None}
        table={20:dict(parent=1,state='S',born='a',cpu=1),21:dict(parent=20,state='R',born='b',cpu=20)}
        first=obs.observe('candidate',cp,status,table,'t0',0)
        self.assertIsNone(first['last_observed_advance_at'])
        table[21]['cpu']=80
        second=obs.observe('candidate',cp,status,table,'t1',60)
        self.assertEqual(second['phase'],'computing_no_new_checkpoint')
        self.assertIsNone(second['last_observed_advance_at'])
        cp={'ticks':16}
        third=obs.observe('candidate',cp,status,table,'t2',120)
        self.assertEqual(third['phase'],'saved_advance')
        self.assertEqual(third['last_observed_advance_at'],'t2')
        idle=obs.observe('candidate',cp,status,table,'t3',1020)
        self.assertEqual(idle['phase'],'suspected_stall_no_cpu_or_checkpoint_15m')
        failed=obs.observe('candidate',cp,dict(status,return_code=1),table,'t4',1080)
        self.assertEqual(failed['phase'],'failed')
        with self.assertRaisesRegex(AssertionError,'regressed'):
            obs.observe('candidate',{'ticks':12},status,table,'t5',1140)

    def test_process_pid_reuse_and_exit_do_not_make_cpu_go_backwards(self):
        obs=Progress(); cp={'ticks':4}; status={'pid':20}
        table={20:dict(parent=1,born='a',cpu=2),21:dict(parent=20,born='b',cpu=30)}
        self.assertEqual(descendants(20,table),{20,21})
        obs.observe('x',cp,status,table,'t0',0)
        del table[21]
        result=obs.observe('x',cp,status,table,'t1',60)
        self.assertEqual(result['cpu_delta'],0)
        table[21]=dict(parent=20,born='c',cpu=3)
        result=obs.observe('x',cp,status,table,'t2',120)
        self.assertEqual(result['cpu_delta'],3)

    def test_checkpoint_read_is_non_mutating_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); job=['any',0,1,'outside',128,1,31,0]
            manifest=encoded({'reflex/example.py':'abc'})
            (root/'source_snapshot.json').write_bytes(manifest)
            owner={'last_tick':0};world={'tick':1}
            payload=dict(contract={'job':job,'sources':json.loads(manifest)},world=world,owner=owner,
                         trace=[{'tick':0,'after':world,'owner':digest(encoded(owner))}],elapsed=21)
            raw=encoded({'payload':payload,'sha256':digest(encoded(payload))})
            file=root/'checkpoint.json';file.write_bytes(raw)
            result=checkpoint_summary(root,job,digest(manifest))
            self.assertEqual(result['ticks'],1)
            self.assertEqual(file.read_bytes(),raw)
            bad=dict(payload,elapsed=22)
            file.write_bytes(encoded({'payload':bad,'sha256':digest(encoded(payload))}))
            with self.assertRaisesRegex(AssertionError,'checksum'):
                checkpoint_summary(root,job,digest(manifest))

    def test_publisher_updates_same_check_and_is_neutral_not_quality_pass(self):
        from tools.observe_horizon_pairs import CheckPublisher
        env={'GITHUB_REPOSITORY':'owner/repo','GITHUB_RUN_ID':'7','GITHUB_RUN_ATTEMPT':'2','GITHUB_SHA':'a'*40}
        with patch.dict('os.environ',env),patch.object(CheckPublisher,'request',return_value={'id':90}) as request:
            publisher=CheckPublisher('candidate')
            publisher.publish({'arms':[]})
            publisher.publish({'arms':[{'ticks':4}]})
            publisher.publish({'arms':[{'ticks':8}]},finished=True)
            self.assertEqual([c.args[:2] for c in request.call_args_list],
                             [('POST','check-runs'),('PATCH','check-runs/90'),('PATCH','check-runs/90')])
            self.assertEqual(request.call_args.args[2]['conclusion'],'neutral')

    def test_reader_separates_stale_telemetry_and_rejects_wrong_source(self):
        from tools.inspect_registered_progress import live_progress
        task={'run':7,'head':'a'*40,'key':'case'};run={'run_attempt':2}
        external='horizon-progress:7:2:case-baseline'
        value=dict(version='horizon-progress-v1',run=7,attempt=2,head_sha=task['head'],
                   external_id=external,observed_at='2020-01-01T00:00:00+00:00',
                   arms=[{'key':'case-baseline'},{'key':'case-candidate'}])
        check=dict(id=3,html_url='https://github.com/example',external_id=external,
                   head_sha=task['head'],app={'slug':'github-actions'},output={'text':json.dumps(value)})
        result=live_progress([check],task,run)
        self.assertTrue(result['stale'])
        self.assertIsNone(live_progress([check],task,{'run_attempt':3}))
        check['head_sha']='b'*40
        with self.assertRaises(AssertionError):live_progress([check],task,run)


if __name__=='__main__':unittest.main()
