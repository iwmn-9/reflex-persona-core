import hashlib,json,subprocess,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r))
from tools.run_owned_value import run
p=r/'results/owner_checkpoint_equivalence'
def canonical(row):return json.dumps({k:v for k,v in row.items() if k!='seconds'},sort_keys=True,separators=(',',':'))
def verify(item):
 index,(job,arm)=item; original=json.loads((p/f'original-{index}.json').read_text()); fresh=run(job,arm)
 assert canonical(fresh)==canonical(original),('unsliced changed',index)
 folder=p/f'case-{index}';folder.mkdir(exist_ok=False)
 (folder/'job.json').write_text(json.dumps(job));row_file=folder/'row.json';chunks=[]
 for size in (3,2,job[4]):
  cmd=[sys.executable,str(r/'tools/run_owned_chunk.py'),'--job-file',str(folder/'job.json'),'--arm',arm,'--checkpoint',str(folder/'checkpoint.json'),'--decisions',str(size),'--output',str(row_file)]
  result=subprocess.run(cmd,cwd=r,text=True,capture_output=True)
  assert result.returncode==0,(cmd,result.stdout,result.stderr)
  chunks.append(json.loads(result.stdout))
  if chunks[-1]['complete']:break
 actual=json.loads(row_file.read_text());assert canonical(actual)==canonical(original),('resumed changed',index)
 return dict(index=index,job=job,arm=arm,chunks=chunks,unsliced_exact=True,resumed_exact=True,non_timing_sha256=hashlib.sha256(canonical(actual).encode()).hexdigest())
if __name__=='__main__':
 manifest=json.loads((p/'original_manifest.json').read_text())
 for name,expected in manifest.items():
  if name!='tools/run_owned_value.py':assert hashlib.sha256((r/name).read_bytes()).hexdigest()==expected,name
 jobs=json.loads((p/'registration.json').read_text());out=[]
 with ProcessPoolExecutor(max_workers=2) as pool:
  fs=[pool.submit(verify,item) for item in enumerate(jobs)]
  for future in as_completed(fs):
   row=future.result();out.append(row);print(json.dumps(dict(index=row['index'],exact=True)),flush=True)
 result=dict(complete=True,executions=len(out),reference_runner_sha256=manifest['tools/run_owned_value.py'],checkpoint_runner_sha256=hashlib.sha256((r/'tools/run_owned_value.py').read_bytes()).hexdigest(),cases=sorted(out,key=lambda x:x['index']),note='Every non-timing field equals the original runner captured before the checkpoint change. Each resumed chunk ran in a fresh Python process.')
 (p/'verification.json').write_text(json.dumps(result,indent=2));print('All original, unsliced, and separately resumed executions match.',flush=True)
