"""Vary only coherent model-bank size after a long-horizon regression."""
import argparse,hashlib,json,shutil,subprocess,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import digest
from reflex.laboratory import profiles
from tools.run_contingent_coverage import cases

def hashes():
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for d in ('reflex','tools','tests') for p in sorted((ROOT/d).glob('*.py'))}

def worker(job):
    import reflex.integrated_trials as trials
    from reflex.contingent_games import make_bridge
    case,profile,seed,count,stage=job
    def factory(p,c):
        bridge=make_bridge(p,c);bridge.seeds=tuple(range(401,401+count));return bridge
    # Explicit per-instance study configuration, no rule/scorer modification.
    trials.make_bridge=factory;t=time.perf_counter();r=trials.play(case,profile,seed,'returns',horizon=8)
    r.update(stage=stage,bank=count,elapsed_seconds=time.perf_counter()-t);return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--frozen',action='store_true');a=p.parse_args();out=Path(a.output).resolve()
    if not 1<=a.workers<=8:raise ValueError('bounded independent workers required')
    if not a.frozen:
        out.mkdir(parents=True,exist_ok=False);source=out/'source';source.mkdir()
        for d in ('reflex','tools','tests'):shutil.copytree(ROOT/d,source/d,ignore=shutil.ignore_patterns('__pycache__'))
        return subprocess.call([sys.executable,str(source/'tools'/Path(__file__).name),'--output',str(out),'--workers',str(a.workers),'--frozen'],cwd=source)
    assert ROOT==out/'source';fixed=hashes();jobs=[(c,p,s,n,stage) for stage in ('development','confirmation') for c in cases(stage) if c['game'] in ('auction_contest','hagetaka','gambling') for p in profiles() for s in range(1700,1704) for n in (2,8)]
    pre=dict(format='integrated-bank-v1',jobs=jobs,source_hashes=fixed,horizon=8,banks={'2':[401,402],'8':list(range(401,409))},
        hypothesis='more coherent trajectories reduce sampling error; they cannot correct a wrong executor or opponent model',
        controls='same rules, persona, return frame and horizon; only bank size changes',
        limitation='exploratory follow-up after observed long-horizon losses; not independent confirmation or calibrated probabilities')
    (out/'preregister.json').write_text(json.dumps(pre,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');runs=[];errors=[];start=time.perf_counter()
    with (out/'trajectories.jsonl').open('x',encoding='utf-8') as stream,ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs={pool.submit(worker,j):i for i,j in enumerate(jobs)}
        for f in as_completed(fs):
            i=fs[f]
            try:r=f.result()
            except Exception as e:errors.append(dict(index=i,job=jobs[i],error=repr(e)));print('ERROR',i,repr(e),flush=True);continue
            r['index']=i;stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush();runs.append({k:v for k,v in r.items() if k!='trace'})
            print(len(runs)+len(errors),'/',len(jobs),r['case']['game'],r['bank'],r['goal'],round(r['elapsed_seconds'],1),flush=True)
    assert hashes()==fixed;groups={}
    for r in runs:groups.setdefault(digest([r['case'],r['profile'],r['seed'],r['stage']]),{})[r['bank']]=r
    pairs=[]
    for g in groups.values():
        if set(g)!={2,8}:continue
        x,y=g[2],g[8];pairs.append(dict(case=x['case'],game=x['case']['game'],profile=x['profile'],seed=x['seed'],stage=x['stage'],
            baseline=x['goal']['value'],expanded=y['goal']['value'],delta=y['goal']['value']-x['goal']['value'],actions_changed=x['actions']!=y['actions']))
    result=dict(format=pre['format'],planned=len(jobs),episodes=len(runs),errors=errors,runs=runs,paired=pairs,source_hashes=fixed,preregister_sha256=digest(pre),elapsed_seconds=time.perf_counter()-start)
    (out/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');return int(bool(errors))

if __name__=='__main__':raise SystemExit(main())
