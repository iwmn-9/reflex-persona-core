"""Preregister cutoff-vs-settlement comparisons without fitting game weights."""
import argparse,hashlib,json,shutil,subprocess,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import digest
from reflex.laboratory import profiles
from reflex.integrated_trials import play
from tools.run_contingent_coverage import cases

def hashes():
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for d in ('reflex','tools','tests') for p in sorted((ROOT/d).glob('*.py'))}

def worker(job):
    case,profile,seed,horizon,stage=job;t=time.perf_counter();r=play(case,profile,seed,'returns',horizon=horizon)
    r.update(stage=stage,elapsed_seconds=time.perf_counter()-t)
    traces=[m for row in r['trace'] for m in row['modeled']]
    r['selected_terminal_branches']=sum(m['terminal'] for m in traces);r['selected_model_branches']=len(traces)
    return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=2);p.add_argument('--frozen',action='store_true');a=p.parse_args();out=Path(a.output).resolve()
    if not 1<=a.workers<=8:raise ValueError('bounded independent workers required')
    if not a.frozen:
        out.mkdir(parents=True,exist_ok=False);source=out/'source';source.mkdir()
        for d in ('reflex','tools','tests'):shutil.copytree(ROOT/d,source/d,ignore=shutil.ignore_patterns('__pycache__'))
        return subprocess.call([sys.executable,str(source/'tools'/Path(__file__).name),'--output',str(out),'--workers',str(a.workers),'--frozen'],cwd=source)
    assert ROOT==out/'source';fixed=hashes()
    jobs=[(c,p,s,h,stage) for stage in ('development','confirmation') for c in cases(stage) if c['game'] in ('auction_contest','hagetaka','gambling') for p in profiles() for s in range(1700,1704) for h in (2,4,8)]
    pre=dict(format='integrated-horizon-v1',jobs=jobs,source_hashes=fixed,
        hypothesis='settlement-based evaluation may resolve failures hidden by short cutoff proxies; longer models may also regress',
        controls='same return units, personal rigidity .5, two model seeds, actual feedback off, existing rules; only horizon changes',
        motivation='a complete H2 paired auction loss traced to a lower-cost choice within the proxy purpose corridor',
        limitations=['exploratory follow-up selected after observing main-cohort loss, not independent confirmatory evidence',
            'forecast models still use immediate persona continuation and two hypotheses; terminal samples are not calibrated probabilities'])
    (out/'preregister.json').write_text(json.dumps(pre,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');runs=[];errors=[];start=time.perf_counter()
    with (out/'trajectories.jsonl').open('x',encoding='utf-8') as stream,ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs={pool.submit(worker,j):i for i,j in enumerate(jobs)}
        for f in as_completed(fs):
            i=fs[f]
            try:r=f.result()
            except Exception as e:errors.append(dict(index=i,job=jobs[i],error=repr(e)));print('ERROR',i,repr(e),flush=True);continue
            r['index']=i;stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush();runs.append({k:v for k,v in r.items() if k!='trace'})
            print(len(runs)+len(errors),'/',len(jobs),r['case']['game'],r['horizon'],r['goal'],round(r['elapsed_seconds'],1),flush=True)
    assert hashes()==fixed;groups={}
    for r in runs:groups.setdefault(digest([r['case'],r['profile'],r['seed'],r['stage']]),{})[r['horizon']]=r
    paired=[]
    for g in groups.values():
        if set(g)!={2,4,8}:continue
        r=g[2];paired.append(dict(game=r['case']['game'],case=r['case'],profile=r['profile'],seed=r['seed'],stage=r['stage'],
            values={str(h):g[h]['goal']['value'] for h in (2,4,8)},delta4=g[4]['goal']['value']-r['goal']['value'],delta8=g[8]['goal']['value']-r['goal']['value'],
            actions_changed4=g[4]['actions']!=r['actions'],actions_changed8=g[8]['actions']!=r['actions']))
    result=dict(format=pre['format'],planned=len(jobs),episodes=len(runs),errors=errors,runs=runs,paired=paired,
        source_hashes=fixed,preregister_sha256=digest(pre),elapsed_seconds=time.perf_counter()-start)
    (out/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return int(bool(errors))

if __name__=='__main__':raise SystemExit(main())
