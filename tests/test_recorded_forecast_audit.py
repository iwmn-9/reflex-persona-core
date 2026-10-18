import hashlib,json,math,os,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from tools.audit_recorded_forecasts import probability_differences,verify_receipt


R=Path(__file__).resolve().parents[1]


class RecordedForecastAuditTests(unittest.TestCase):
    def test_numerical_bound_does_not_accept_changed_or_invalid_probabilities(self):
        original={'a':.3,'b':.7};fresh=dict(original)
        for _ in range(8):fresh['a']=math.nextafter(fresh['a'],math.inf)
        self.assertEqual(probability_differences(original,fresh)[0]['ulps'],8)
        fresh['a']=math.nextafter(fresh['a'],math.inf)
        with self.assertRaisesRegex(ValueError,'bound'):probability_differences(original,fresh)
        for bad in ({'a':True,'b':0.},{'a':math.nan,'b':.7},{'a':-.1,'b':1.1},
                    {'a':.3,'b':.6},{'different':1.}):
            with self.assertRaises(ValueError):probability_differences(original,bad)
        self.assertEqual(original,{'a':.3,'b':.7})

    def fixture(self,root):
        job=['gambling',1,52996,'shift-return',3,1,2,0]
        subprocess.run([sys.executable,str(R/'tools/run_horizon_study.py'),'--root',str(root),
            '--job',json.dumps(job),'--stage','diagnostic'],check=True,capture_output=True)
        (root/'audit.json').unlink()
        return job

    def test_recorded_probabilities_replay_exact_owners_and_reject_other_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'base';self.fixture(base)
            program='''import importlib.util,math,sys
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root/'_source'))
from reflex.event_reader import EventReader
original=EventReader.predict
def rounded(self,c):
 p=original(self,c);key=next(iter(p))
 for _ in range(int(sys.argv[3])):p[key]=math.nextafter(p[key],math.inf)
 return p
EventReader.predict=rounded
spec=importlib.util.spec_from_file_location('audit_checker',sys.argv[2]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.audit(root)
'''
            for name,ulps,alter_owner in (('rounding',2,False),('changed_forecast',16,False),('changed_owner',2,True)):
                child=root/name;shutil.copytree(base,child)
                if alter_owner:
                    path=child/'trajectories.jsonl';data=json.loads(path.read_text());data['trace'][0]['owner']='0'*64
                    path.write_text(json.dumps(data))
                trajectory=(child/'trajectories.jsonl').read_bytes();checkpoint=(child/'checkpoint.json').read_bytes()
                result=subprocess.run([sys.executable,'-c',program,str(child),str(R/'tools/audit_recorded_forecasts.py'),str(ulps)],capture_output=True)
                self.assertEqual((child/'trajectories.jsonl').read_bytes(),trajectory)
                self.assertEqual((child/'checkpoint.json').read_bytes(),checkpoint)
                receipt=json.loads((child/'recorded_forecast_replay.json').read_text())
                if name=='rounding':
                    self.assertEqual(result.returncode,0,result.stderr)
                    audit=json.loads((child/'audit.json').read_text())
                    self.assertTrue(receipt['accepted']);self.assertEqual(receipt['exact_owner_hashes'],3)
                    self.assertGreater(receipt['max_observed_ulps'],0)
                    self.assertEqual(audit['trajectory_sha256'],hashlib.sha256(trajectory).hexdigest())
                    row=json.loads(trajectory)
                    self.assertEqual(verify_receipt(child,audit,row),audit['recorded_forecast_replay'])
                    hidden=dict(audit);hidden.pop('recorded_forecast_replay')
                    with self.assertRaisesRegex(AssertionError,'hidden'):verify_receipt(child,hidden,row)
                    altered=json.loads(json.dumps(audit));altered['recorded_forecast_replay']['exact_owner_hashes']=2
                    with self.assertRaises(AssertionError):verify_receipt(child,altered,row)
                    altered_receipt=json.loads(json.dumps(receipt));altered_receipt['differences'][0]['saved']+=.01
                    (child/'recorded_forecast_replay.json').write_text(json.dumps(altered_receipt))
                    with self.assertRaises(AssertionError):verify_receipt(child,audit,row)
                    (child/'recorded_forecast_replay.json').write_text(json.dumps(receipt))
                    again=subprocess.run([sys.executable,str(R/'tools/audit_recorded_forecasts.py'),str(child)],capture_output=True)
                    self.assertNotEqual(again.returncode,0);self.assertIn(b'existing audit',again.stderr)
                else:
                    self.assertNotEqual(result.returncode,0)
                    self.assertFalse(receipt['accepted']);self.assertFalse((child/'audit.json').exists())
            changed=root/'source_changed';shutil.copytree(base,changed)
            path=changed/'_source/reflex/core.py';path.write_bytes(path.read_bytes()+b'\n')
            result=subprocess.run([sys.executable,str(R/'tools/audit_recorded_forecasts.py'),str(changed)],capture_output=True)
            self.assertNotEqual(result.returncode,0);self.assertIn(b'frozen source changed',result.stderr)
            self.assertFalse((changed/'audit.json').exists())
            # A continuation host may have installed the old actor package.
            # The new supervisor must resolve its own receipt checker.
            legacy=root/'legacy/tools';legacy.mkdir(parents=True);(legacy/'old_tool.py').write_text('')
            job=json.loads((base/'registration.json').read_text())['jobs'][0]
            result=subprocess.run([sys.executable,str(R/'tools/resume_horizon_study.py'),'--root',str(base),
                '--job',json.dumps(job),'--manifest',hashlib.sha256((base/'source_snapshot.json').read_bytes()).hexdigest()],
                cwd=root,env={**os.environ,'PYTHONPATH':str(legacy.parent)},capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_explicit_audit_recovery_keeps_completed_world_bytes(self):
        from tools.resume_horizon_study import resume
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'world';job=self.fixture(root)
            manifest=hashlib.sha256((root/'source_snapshot.json').read_bytes()).hexdigest()
            before={name:(root/name).read_bytes() for name in ('trajectories.jsonl','checkpoint.json','source_snapshot.json')}
            real_run=subprocess.run
            def failed_strict_audit(args,**kwargs):
                if args[1].endswith('audit_owned_value.py'):raise subprocess.CalledProcessError(1,args)
                return real_run(args,**kwargs)
            with patch('tools.resume_horizon_study.subprocess.run',side_effect=failed_strict_audit):
                with self.assertRaises(subprocess.CalledProcessError):resume(root,manifest,job)
                self.assertFalse((root/'audit.json').exists())
                proof=resume(root,manifest,job,recorded_predictions=True)
            self.assertTrue(proof['audit_only']);self.assertTrue(proof['original_prefix_unchanged'])
            self.assertEqual(proof['audit_recipe']['exact_owner_hashes'],3)
            self.assertEqual({name:(root/name).read_bytes() for name in before},before)


if __name__=='__main__':unittest.main()
