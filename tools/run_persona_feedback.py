"""Freeze and run matched known-rule decision-feedback comparisons."""
import argparse,hashlib,json,shutil,subprocess,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.decision_feedback_trials import play,GAMES,REGIMES
from reflex.laboratory import profiles


def worker(job):
    game,regime,p,seed,metric,turns,priority=job
    return play(game,regime,p,seed,metric,turns=turns,priority=priority)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--seeds',type=int,default=4)
    ap.add_argument('--seed-start',type=int,default=9100);ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--turns',type=int,default=64);ap.add_argument('--games',nargs='+',default=GAMES)
    ap.add_argument('--regimes',nargs='+',default=REGIMES);ap.add_argument('--priority',default='finite',choices=['finite','lexicographic'])
    ap.add_argument('--frozen',action='store_true');a=ap.parse_args();out=Path(a.output).resolve()
    if not 1<=a.seeds<=16 or not 1<=a.workers<=8 or not 16<=a.turns<=128:raise ValueError('bounded declared run required')
    if not a.frozen:
        out.mkdir(parents=True,exist_ok=False);source=out/'source';source.mkdir()
        for d in ('reflex','tools','tests'):shutil.copytree(ROOT/d,source/d,ignore=shutil.ignore_patterns('__pycache__'))
        return subprocess.call([sys.executable,str(source/'tools'/Path(__file__).name),'--output',str(out),'--seeds',str(a.seeds),
            '--seed-start',str(a.seed_start),'--workers',str(a.workers),'--turns',str(a.turns),'--games',*a.games,'--regimes',*a.regimes,
            '--priority',a.priority,'--frozen'],cwd=source)
    assert ROOT==out/'source'
    hashes=lambda:{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for d in ('reflex','tests','tools') for p in sorted((ROOT/d).glob('*.py'))}
    fixed=hashes();jobs=[(g,r,p,s,m,a.turns,a.priority) for g in a.games for r in a.regimes for p in profiles() for s in range(a.seed_start,a.seed_start+a.seeds) for m in ('purpose','persona')]
    pre=dict(format='persona-feedback-v1',source_hashes=fixed,jobs=jobs,
        hypothesis='judge conditional-rule advice by the SAME decision-time personal appraisal instead of purpose alone',
        limits=['immediate full conditional rules and exogenous publicly revealed responses only','unselected rule effects are not empirical experience',
            'auction is independent funded lots, not finite-budget campaign','exchange omits social memory and hidden luck','paired worlds are not all independent',
            'appraisal totals on divergent private trajectories are diagnostic, not a common fixed reward'],
        acceptance=['retain every regression and error','report objective AND enacted values','no per-game psychology fit','source frozen before run'])
    (out/'preregister.json').write_text(json.dumps(pre,ensure_ascii=False,indent=2),encoding='utf-8')
    runs=[];errors=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool,(out/'trajectories.jsonl').open('x',encoding='utf-8') as stream:
        fs={pool.submit(worker,j):i for i,j in enumerate(jobs)}
        for f in as_completed(fs):
            i=fs[f]
            try:r=f.result()
            except Exception as e:errors.append(dict(index=i,job=jobs[i],error=repr(e)));print('ERROR',i,repr(e),flush=True);continue
            r['index']=i;stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush()
            runs.append({k:v for k,v in r.items() if k!='trace'})
            if len(runs)%16==0:print(len(runs),'/',len(jobs),round(time.perf_counter()-start,1),flush=True)
    assert hashes()==fixed
    (out/'evaluation.json').write_text(json.dumps(dict(planned=len(jobs),episodes=len(runs),errors=errors,runs=runs,
        source_hashes=fixed,elapsed_seconds=time.perf_counter()-start),ensure_ascii=False,indent=2),encoding='utf-8')
    return int(bool(errors))


if __name__=='__main__':raise SystemExit(main())
