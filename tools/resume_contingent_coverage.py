"""Resume interrupted CPU jobs from their frozen source, retaining every row.

Only infrastructure interruptions are resumable. Rule/contract errors require
a corrected new cohort; this tool cannot silently replace their source.
"""
import argparse,hashlib,importlib,json,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def resume(root,workers):
    root=Path(root).resolve();snapshot=root/'source';pre=read(root/'preregister.json');old=read(root/'evaluation.json')
    if not 1<=workers<=8:raise ValueError('bounded independent CPU worker count required')
    if not old['errors']:raise ValueError('study is already complete')
    if any(not e['error'].startswith('BrokenProcessPool(') for e in old['errors']):
        raise ValueError('numeric/rule errors need a corrected whole cohort, not resumption')
    sys.path[:0]=[str(snapshot/'tools'),str(snapshot)]
    frozen=importlib.import_module('run_contingent_coverage')
    assert Path(frozen.__file__).resolve()==snapshot/'tools/run_contingent_coverage.py'
    assert frozen.ROOT==snapshot and frozen.source_hashes()==pre['source_hashes']==old['source_hashes']
    assert frozen.digest(pre)==old['preregister_digest']
    runs=[];seen=set()
    with (root/'trajectories.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line);index=r['job_index'];assert index not in seen;seen.add(index)
            case,profile,seed,method,*_=pre['jobs'][index]
            assert (r['case'],r['profile'],r['seed'],r['method'])==(case,profile['id'],seed,method)
            runs.append({k:v for k,v in r.items() if k!='trace'})
    assert sorted(runs,key=lambda r:r['job_index'])==old['runs']
    assert len(seen)+len(old['errors'])==old['planned']==len(pre['jobs'])
    pending={e['index'] for e in old['errors']};assert pending==set(range(old['planned']))-seen
    # Long conditional jobs first; execution order is not an input to decisions.
    order=sorted(pending,key=lambda i:(pre['jobs'][i][3]!='conditional',i))
    serial=1
    while (root/f'resume_registration-{serial}.json').exists():serial+=1
    registration=dict(format='frozen-coverage-resumption-v1',workers=workers,indices=order,
        prior_evaluation_sha256=sha(root/'evaluation.json'),prior_raw_sha256=sha(root/'trajectories.jsonl'),
        source_hashes=pre['source_hashes'],worker_sha256=sha(snapshot/'tools/run_contingent_coverage.py'),
        resumer_sha256=sha(Path(__file__)),semantics='same frozen jobs; completed raw prefix retained; unfinished whole episodes restarted; no additional quality samples')
    (root/f'resume_registration-{serial}.json').write_text(json.dumps(registration,indent=2)+'\n',encoding='utf-8')
    (root/f'evaluation_before_resume-{serial}.json').write_bytes((root/'evaluation.json').read_bytes())
    errors=[];start=time.perf_counter()
    with (root/'trajectories.jsonl').open('a',encoding='utf-8',newline='\n') as stream,ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(frozen.worker,pre['jobs'][i]):i for i in order}
        for future in as_completed(futures):
            i=futures[future]
            try:r=future.result()
            except Exception as exc:errors.append(dict(index=i,task=pre['jobs'][i],error=repr(exc)));continue
            r['job_index']=i;stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush()
            runs.append({k:v for k,v in r.items() if k!='trace'})
            print(f"{len(runs)}/{old['planned']} {r['profile']} {r['method']} {r.get('goal',r.get('point_delta'))}",flush=True)
    assert frozen.source_hashes()==pre['source_hashes']
    result=dict(old,episodes=len(runs),errors=errors,runs=sorted(runs,key=lambda r:r['job_index']),
        summary=frozen.summarize(runs),elapsed_seconds=old['elapsed_seconds']+time.perf_counter()-start,
        resumptions=old.get('resumptions',[])+[registration])
    (root/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return 1 if errors else 0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    raise SystemExit(resume(a.root,a.workers))
