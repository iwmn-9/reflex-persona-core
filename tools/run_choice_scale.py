"""Freeze factorial goal-level/gain and absolute/opportunity-band comparisons."""
import argparse,hashlib,json,shutil,subprocess,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.laboratory import profiles
from reflex.reliability_trials import play
from reflex.forecast_control import ForecastBudget
from reflex.core import digest
from tools.run_contingent_coverage import cases
GAMES=('auction_contest','hagetaka','gambling','empire_race','team_combat','cyclic_arena','shared_commons','branching_delivery')
ARMS={'level':(False,False),'gain':(True,False),'opportunity':(False,True),'gain_opportunity':(True,True)}
def hashes():return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for d in ('reflex','tools','tests') for p in sorted((ROOT/d).glob('*.py'))}
def worker(job):
    case,profile,seed,arm,horizon=job;started=time.perf_counter();gain,band=ARMS[arm]
    budget=ForecastBudget(horizon=horizon,objective_gain=gain,opportunity_regret=band,direct_continuation=True)
    r=play(case,profile,seed,arm,budgets={arm:budget});r['elapsed_seconds']=time.perf_counter()-started;return r
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--stage',choices=('development','confirmation','transfer'),required=True)
    ap.add_argument('--seeds',type=int,default=2);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--horizon',type=int,default=4)
    ap.add_argument('--games',nargs='*');ap.add_argument('--arms',nargs='+',default=list(ARMS));ap.add_argument('--frozen',action='store_true');a=ap.parse_args()
    if not 1<=a.seeds<=8 or not 1<=a.workers<=8 or not 1<=a.horizon<=8 or set(a.arms)-set(ARMS):raise ValueError('finite declared comparison required')
    out=Path(a.output).resolve()
    if not a.frozen:
        out.mkdir(parents=True,exist_ok=False);source=out/'source';source.mkdir()
        for d in ('reflex','tools','tests'):shutil.copytree(ROOT/d,source/d,ignore=shutil.ignore_patterns('__pycache__'))
        command=[sys.executable,str(source/'tools'/Path(__file__).name),'--output',str(out),'--stage',a.stage,
            '--seeds',str(a.seeds),'--workers',str(a.workers),'--horizon',str(a.horizon),'--arms',*a.arms,'--frozen']
        if a.games:command.extend(['--games',*a.games])
        return subprocess.call(command,cwd=source)
    assert ROOT==out/'source';fixed=hashes();stage='development' if a.stage=='development' else 'confirmation'
    wanted=a.games or (GAMES if a.stage!='transfer' else ('no_thanks','goofspiel','team_projects'))
    selected=[c for c in cases(stage) if c['game'] in wanted]
    if set(wanted)!={c['game'] for c in selected}:raise ValueError('all declared game families required')
    first={'development':5200,'confirmation':6200,'transfer':8200}[a.stage]
    jobs=[(c,p,s,arm,a.horizon) for c in selected for p in profiles() for s in range(first,first+a.seeds) for arm in a.arms]
    pre=dict(format='choice-scale-v1',stage=a.stage,source_hashes=fixed,jobs=jobs,arms=ARMS,
        hypothesis='centering subjective endpoint on factual current progress reduces arbitrary origin sensitivity and improves actual decisions without erasing value tradeoffs',
        controls='equal H4/N4 unless explicit horizon override; unchanged physical rules, fixed personalities/rigidity .5, all immediate learning OFF',
        limitations=['opportunity fraction is an experimental negative control: two unequal deterministic options collapse to the goal-best for every fraction below one',
            'reference gain changes loss aversion; not potential-based shaping or an optimal-policy guarantee',
            'confidence adjustments are not calibrated; units conversion invariance demonstrated only under certain outcomes',
            'game endpoint and actual preference flows are distinct outcomes'],acceptance=['retain every error/regression','whole actual rule replay','purpose and realized values reported together','no game-specific coefficients'])
    (out/'preregister.json').write_text(json.dumps(pre,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');runs=[];errors=[];started=time.perf_counter()
    with (out/'trajectories.jsonl').open('x',encoding='utf-8') as stream,ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs={pool.submit(worker,j):i for i,j in enumerate(jobs)}
        for f in as_completed(fs):
            i=fs[f]
            try:r=f.result()
            except Exception as e:errors.append(dict(index=i,job=jobs[i],error=repr(e)));print('ERROR',i,repr(e),flush=True);continue
            r['index']=i;stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush()
            slim={k:v for k,v in r.items() if k!='trace'};scores=[x for t in r['trace'] for x in t['forecast_scores']]
            slim.update(forecast_loss=sum(s['forecast_loss'] for s in scores)/len(scores),persistence_loss=sum(s['persistence_loss'] for s in scores)/len(scores),
                exact_action_branch_matches=sum(s['matching_action_branches'] for s in scores),scored_branches=sum(s['branches'] for s in scores))
            runs.append(slim);print(len(runs)+len(errors),'/',len(jobs),r['case']['game'],r['profile'],r['arm'],r['goal'],round(r['elapsed_seconds'],1),flush=True)
    assert hashes()==fixed
    (out/'evaluation.json').write_text(json.dumps(dict(format=pre['format'],stage=a.stage,planned=len(jobs),episodes=len(runs),errors=errors,runs=runs,
        source_hashes=fixed,preregister_sha256=digest(pre),elapsed_seconds=time.perf_counter()-started),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return int(bool(errors))
if __name__=='__main__':raise SystemExit(main())
