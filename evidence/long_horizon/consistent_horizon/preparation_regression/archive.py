"""Archive the complete original twelve-case regression, never partial success."""
import argparse,gzip,hashlib,json,zipfile
from pathlib import Path
def read(p):return json.loads(Path(p).read_bytes())
def encoded(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
 args=parser.parse_args();r=Path(args.root);out=Path(args.output)
 summary=read(r/'summary.json');plan=read(r/'plan.json');sources=read(r/'source_snapshot.json')
 assert read(r/'status.json')['complete'] and summary['complete']
 assert len(plan['specs'])==len(summary['pairs'])==12
 assert summary['actual_steps']==3072 and plan['modes']==['advice','current_model']
 assert summary['source_snapshot_sha256']==sha((r/'source_snapshot.json').read_bytes())
 assert summary['plan_sha256']==sha((r/'plan.json').read_bytes())
 assert plan['original_plan_sha256']==sha((r/'original_plan.json').read_bytes())
 assert plan['specs']==read(r/'original_plan.json')['specs']
 if out.exists():raise FileExistsError('Do not overwrite complete archived evidence')
 runtime={n:(r/'_source'/n).read_bytes() for n in sources}
 assert all(sha(b)==sources[n] for n,b in runtime.items())
 files={};runs=[];checks=[];reported_runtime=None
 for spec,pair in zip(plan['specs'],summary['pairs']):
  assert pair['id']==spec['id'] and spec['turns']==128
  values=[]
  for mode,receipt in zip(plan['modes'],pair['arms']):
   folder=r/spec['id']/mode
   saved={n:(folder/n).read_bytes() for n in ('checkpoint.json','result.json','audit.json')}
   assert read(folder/'audit.json')==receipt and receipt['complete'] and receipt['owner_replayed']
   assert receipt['actual_steps']==128
   assert all(sha(saved[n])==h for n,h in receipt['files'].items())
   cp=json.loads(saved['checkpoint.json']);v=cp['payload'];result=json.loads(saved['result.json'])
   assert sha(encoded(v))==cp['sha256']
   assert result['complete'] and result['trace']==v['trace'] and len(v['trace'])==result['physical_replay']==128
   assert result['configuration']==v['contract']['configuration']
   assert result['configuration']['spec']==spec and result['configuration'].get('reader_selection','advice')==mode
   assert result['final']==v['world'] and result['score']==receipt['score']
   assert v['owner']['last_tick']+1==v['world']['tick']==128
   assert result['checks']==v['checks'] and result['checks']['fixed_axes']==result['checks']['owner_parity']==128
   full_runtime=v['contract']['runtime']
   assert receipt['source_runtime']==full_runtime
   assert all(full_runtime[k]==value for k,value in plan['runtime'].items())
   assert result['runtime']['python']==full_runtime['python'].split()[0]
   assert result['runtime']['numpy']==full_runtime['numpy']
   assert result['runtime']['platform'].split('-')[0]==full_runtime['system']
   if reported_runtime is None:reported_runtime=result['runtime']
   assert result['runtime']==reported_runtime
   checks.append(dict(id=spec['id'],mode=mode,checks=result['checks'],persona=receipt['persona']))
   values.append(result['score']);runs.append(dict(spec=spec,mode=mode,result=result))
   for n,b in saved.items():files[f"{spec['id']}/{mode}/{n}"]=b
  assert values[1]-values[0]==pair['delta']
  assert pair['arms'][0]['persona']==pair['arms'][1]['persona']
 assert summary['preparation_no_loss']==all(p['delta']>=-1e-12 for p in summary['pairs'])
 out.mkdir(parents=True)
 for n in ('plan.json','original_plan.json','summary.json','source_snapshot.json','status.json'):(out/n).write_bytes((r/n).read_bytes())
 for name,contents in (('source.zip',runtime),('checkpoints.zip',files)):
  with zipfile.ZipFile(out/name,'x',compression=zipfile.ZIP_DEFLATED) as z:
   for n,b in contents.items():z.writestr(n,b)
  with zipfile.ZipFile(out/name) as z:
   assert set(z.namelist())==set(contents)
   assert all(z.read(n)==b for n,b in contents.items())
 (out/'runs.jsonl.gz').write_bytes(gzip.compress(b''.join(encoded(x)+b'\n' for x in runs),mtime=0))
 (out/'archive.py').write_bytes(Path(__file__).read_bytes())
 (out/'verification.json').write_bytes(encoded(dict(complete=True,pairs=12,executions=24,actual_steps=3072,
  no_preparation_loss=summary['preparation_no_loss'],adoption=False,checks=checks,
  artifacts={p.name:sha(p.read_bytes()) for p in out.iterdir()},
  limits=['Original deterministic preparation cases are a regression check, not independent multi-world confirmation.',
   'Each actual owner trajectory was replayed by the frozen worker. Restore-event counts are the original scheduled counts recorded above, not one fresh restore per step.'])))
 print(json.dumps(dict(complete=True,pairs=12,actual_steps=3072,no_loss=summary['preparation_no_loss'],adoption=False)))
if __name__=='__main__':main()
