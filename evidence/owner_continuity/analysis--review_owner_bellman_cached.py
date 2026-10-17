"""Run the unchanged auction evaluator per frozen source, with reusable results.

This only schedules post-hoc analysis. It never changes a playing policy, counts
cached analyses as new trials, or declares an incomplete registration complete.
"""
import argparse, hashlib, json, os, platform, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import numpy as np
from tools import diagnose_auction_continuity as evaluator


def sha(data):return hashlib.sha256(data).hexdigest()


def one(job):
    source,cache,code=job;path=Path(source);raw=path.read_bytes()
    identity=dict(source=str(path),source_sha256=sha(raw),analysis_sources=code,
        python=platform.python_version(),numpy=np.__version__)
    key=sha(json.dumps(identity,sort_keys=True).encode())
    dest=Path(cache)/(key+'.json')
    if dest.exists():
        saved=json.loads(dest.read_text());assert saved['identity']==identity
        result=saved['result'];assert saved['result_sha256']==sha(json.dumps(result,sort_keys=True).encode())
        return result,True
    rows=[evaluator.diagnose(r) for r in map(json.loads,raw.decode().splitlines()) if r['game']=='auction']
    assert path.read_bytes()==raw,'source changed during analysis'
    result=dict(source=dict(path=str(path),sha256=sha(raw)),runs=rows)
    payload=dict(identity=identity,result=result,result_sha256=sha(json.dumps(result,sort_keys=True).encode()))
    temporary=dest.with_suffix('.'+str(os.getpid())+'.new')
    temporary.write_text(json.dumps(payload)+'\n');temporary.replace(dest)
    return result,False


def main():
    p=argparse.ArgumentParser();p.add_argument('--sources',nargs='+',required=True)
    p.add_argument('--output',required=True);p.add_argument('--cache',required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    assert 1<=a.workers<=8
    sources=[str(Path(s)) for s in a.sources];assert len(sources)==len(set(sources))
    cache=Path(a.cache);cache.mkdir(parents=True,exist_ok=True)
    code={name:sha((R/name).read_bytes()) for name in
          ('tools/diagnose_auction_continuity.py','tools/run_owned_value.py','results/review_owner_bellman_cached.py')}
    jobs=[(source,str(cache),code) for source in sources]
    rows=[];provenance=[];reused=0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for result,cached in pool.map(one,jobs):
            provenance.append(result['source']);rows.extend(result['runs']);reused+=int(cached)
            print(json.dumps(dict(analyzed_sources=len(provenance),requested_sources=len(jobs),cached=cached)),flush=True)
    result=dict(semantics=evaluator.__doc__,sources=provenance,runs=rows,
        execution=dict(source_files=len(sources),cached_files=reused,analysis_sources=code,
            note='Scheduling/cache only; per-row numerical evaluator is unchanged. No completeness or adoption claim.'))
    out=Path(a.output);temporary=out.with_suffix('.new')
    temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(out)
    print(json.dumps(dict(runs=len(rows),sources=len(sources),cached_sources=reused)),flush=True)


if __name__=='__main__':main()
