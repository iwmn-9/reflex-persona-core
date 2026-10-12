"""Freeze a prospectively specified multi-game reliability experiment."""
import argparse,hashlib,json,shutil,subprocess,sys,time
from dataclasses import replace
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import digest
from reflex.laboratory import profiles
from reflex.reliability_trials import play,BUDGETS
from tools.run_contingent_coverage import cases

GAMES=('auction_contest','hagetaka','gambling','empire_race','team_combat','cyclic_arena','shared_commons','branching_delivery')


def hashes():
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for d in ('reflex','tools','tests') for p in sorted((ROOT/d).glob('*.py'))}


def worker(job):
    start=time.perf_counter();case,profile,seed,arm,stage,*horizon=job
    budgets=None if not horizon else {arm:replace(BUDGETS[arm],horizon=BUDGETS[arm].horizon if arm=='short' else horizon[0])}
    r=play(case,profile,seed,arm,budgets=budgets);r.update(stage=stage,elapsed_seconds=time.perf_counter()-start);return r


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--seeds',type=int,default=4);ap.add_argument('--stage',choices=('pilot','development','confirmation','transfer'),required=True)
    ap.add_argument('--horizon',type=int)
    ap.add_argument('--games',nargs='*');ap.add_argument('--arms',nargs='*',default=[k for k in BUDGETS if k not in ('adaptive','challenged')]);ap.add_argument('--frozen',action='store_true')
    args=ap.parse_args();out=Path(args.output).resolve()
    if not 1<=args.workers<=8 or not 1<=args.seeds<=16 or set(args.arms)-set(BUDGETS):raise ValueError('bounded explicit cohort required')
    if args.horizon is not None and not 1<=args.horizon<=16:raise ValueError('bounded optional follow-up horizon required')
    if not args.frozen:
        out.mkdir(parents=True,exist_ok=False);source=out/'source';source.mkdir()
        for d in ('reflex','tools','tests'):shutil.copytree(ROOT/d,source/d,ignore=shutil.ignore_patterns('__pycache__'))
        return subprocess.call([sys.executable,str(source/'tools'/Path(__file__).name),*sys.argv[1:],'--frozen'],cwd=source)
    assert ROOT==out/'source';fixed=hashes();changed=args.stage in ('confirmation','transfer')
    selected=[c for c in cases('confirmation' if changed else 'development') if c['game'] in (args.games or GAMES)]
    if args.games and set(args.games)!={c['game'] for c in selected}:raise ValueError('all specified adapters required')
    ps=profiles()[:1] if args.stage=='pilot' else profiles()
    start_seed=3400 if changed else 2400
    jobs=[(c,p,s,arm,args.stage) for c in selected for p in ps for s in range(start_seed,start_seed+args.seeds) for arm in args.arms]
    if args.horizon is not None:jobs=[(*j,args.horizon) for j in jobs]
    budgets={k:BUDGETS[k] if args.horizon is None or k=='short' else replace(BUDGETS[k],horizon=args.horizon) for k in args.arms}
    pre=dict(format='forecast-reliability-v1',source_hashes=fixed,budgets={k:vars(v) for k,v in budgets.items()},jobs=jobs,
        hypotheses=['finite future replanning improves decisions beyond identical depth with reflex continuation',
            'independent paired validation reduces lucky discovery choices without erasing personality',
            'delayed goal scoring can reveal persistent model error; no causal credit or adoption from prediction accuracy alone'],
        controls='unchanged rules, fixed profiles, personal rigidity .5, all immediate learning OFF; actual RNG separate',
        acceptance=['all scheduled games/errors/losses retained','actual ledger replays exactly','no hypothetical owner learning',
            'confirmation and transfer coefficients frozen before outcomes; no per-game repair'],
        limitations=['new actual seeds, existing authored adapters; not independent unseen game designs',
            'more_samples is scenario-count control, not equal compute with nested controllers',
            'future policy has a finite nesting cutoff; predictive scores include real subsequent replanning',
            'running game proxies are not calibrated victory probabilities'],cloud_runtime_used=False)
    (out/'preregister.json').write_text(json.dumps(pre,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    started=time.perf_counter();runs=[];errors=[]
    with (out/'trajectories.jsonl').open('x',encoding='utf-8') as stream,ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs={pool.submit(worker,j):i for i,j in enumerate(jobs)}
        for f in as_completed(fs):
            i=fs[f]
            try:r=f.result()
            except Exception as e:errors.append(dict(index=i,job=jobs[i],error=repr(e)));print('ERROR',i,repr(e),flush=True);continue
            r['index']=i;stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush()
            slim={k:v for k,v in r.items() if k!='trace'}
            scores=[x for t in r['trace'] for x in t['forecast_scores']]
            checks=[t['check'] for t in r['trace'] if t['check'] is not None]
            slim.update(forecast_loss=sum(s['forecast_loss'] for s in scores)/len(scores),
                persistence_loss=sum(s['persistence_loss'] for s in scores)/len(scores),
                exact_action_branch_matches=sum(s['matching_action_branches'] for s in scores),scored_branches=sum(s['branches'] for s in scores),
                validation_changes=sum(t['candidate']!=t['reference'] for t in checks),validation_accepted=sum(t['accepted'] and t['candidate']!=t['reference'] for t in checks))
            slim['adaptive_expansions']=sum(t.get('adaptive') is not None for t in r['trace'])
            runs.append(slim);print(len(runs)+len(errors),'/',len(jobs),r['case']['game'],r['profile'],r['arm'],r['goal'],round(r['elapsed_seconds'],1),flush=True)
    assert hashes()==fixed;groups={}
    for r in runs:groups.setdefault(digest([r['case'],r['profile'],r['seed']]),{})[r['arm']]=r
    pairs=[]
    for g in groups.values():
        if 'short' not in g:continue
        for arm in args.arms:
            if arm=='short' or arm not in g:continue
            a,b=g['short'],g[arm];pairs.append(dict(case=a['case'],profile=a['profile'],seed=a['seed'],arm=arm,
                baseline=a['goal']['value'],candidate=b['goal']['value'],delta=b['goal']['value']-a['goal']['value'],
                actions_changed=a['actions']!=b['actions'],steps_delta=b['steps']-a['steps'],forecast_loss_delta=b['forecast_loss']-a['forecast_loss']))
    result=dict(format=pre['format'],stage=args.stage,planned=len(jobs),episodes=len(runs),errors=errors,runs=runs,paired=pairs,
        source_hashes=fixed,preregister_sha256=digest(pre),elapsed_seconds=time.perf_counter()-started)
    (out/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return int(bool(errors))


if __name__=='__main__':raise SystemExit(main())
