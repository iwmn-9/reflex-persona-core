"""Fresh uncertainty comparison after stronger-rival mismatch was exposed."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from reflex.board_planning import SHORT
from reflex.laboratory import PROFILES
from tools.run_connect_policy_rollout import worker as pair


def worker(job):
    seed,profile,samples,rival=job
    a=pair((seed,profile,samples,'incumbent',rival,'minimax2'));b=pair((seed,profile,samples,'incumbent',rival,'mixture'))
    assert a[0]==b[0]
    a[0].update(condition='baseline',rival=rival)
    a[1].update(condition='minimax2',rival=rival)
    b[1].update(condition='mixture',rival=rival)
    return [a[0],a[1],b[1]]


def run(root,start,seeds,samples,workers,conditions=('baseline','minimax2','mixture'),job_worker=worker,registration=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);base=Path(__file__).resolve().parents[1]
    plan=dict(version='public-rival-uncertainty-v1',seeds=list(range(start,start+seeds)),samples=samples,workers=workers,
        profiles=[p['id'] for p in PROFILES],rivals=['minimax2','minimax4'],conditions=list(conditions),owner_budget=asdict(SHORT),
        primary='mixture minus baseline credit separately for two actual rival families; per-profile losses must be disclosed',
        secondary='mixture minus fixed minimax2 forecast, with actual owner continuation held constant',
        model='one latent minimax2/minimax4 class uniformly sampled per trial and shared across roots; not actual rival label; same 8 terminal trials and real owner controller',
        accounting='duplicate baseline execution verifies complete equality; not an extra performance observation',
        origin='9340..9355 matched minimax2 improved, stronger minimax4 actual rival did not; add a plausible strength prior, without altering personality or fitting results',
        limits=['known two-model family, no online opponent-type learning','equally weighted prior matches this balanced experiment; no universal calibration',
                'single root intervention; no human-level or continuously upgraded-policy proof'],
        sources={str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((base/'reflex').glob('*.py'))})
    if registration is not None:plan.update(registration)
    path=root/'preregister.json'
    if path.exists():raise FileExistsError('fresh rival-prior study required')
    path.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8');rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(job_worker,(seed,p,samples,rival)) for seed in plan['seeds'] for p in range(4) for rival in plan['rivals']]
        with (root/'trajectories.jsonl').open('w',encoding='utf-8') as f:
            for i,future in enumerate(as_completed(jobs),1):
                batch=future.result();rows.extend(batch)
                for row in batch:f.write(json.dumps(row,separators=(',',':'))+'\n')
                f.flush()
                if i%8==0:print('rival-prior cases',i,'/',len(jobs),flush=True)
    contrasts=[];rates=[]
    for rival in plan['rivals']:
        for condition in plan['conditions']:
            rr=[r for r in rows if r['rival']==rival and r['condition']==condition]
            rates.append(dict(rival=rival,condition=condition,credit=float(np.mean([r['credit'] for r in rr]))))
        first,second=plan['conditions'][1:]
        for reference,candidate in [('baseline',second),('baseline',first),(first,second)]:
            for profile in ['each_npc',*plan['profiles']]:
                series=[];better=same=worse=0
                for seed in plan['seeds']:
                    ps=plan['profiles'] if profile=='each_npc' else [profile];delta=[]
                    for p in ps:
                        a=next(r for r in rows if r['seed']==seed and r['profile']==p and r['rival']==rival and r['condition']==reference)
                        b=next(r for r in rows if r['seed']==seed and r['profile']==p and r['rival']==rival and r['condition']==candidate)
                        v=b['credit']-a['credit'];delta.append(v);better+=v>0;same+=v==0;worse+=v<0
                    series.append(float(np.mean(delta)))
                values=np.array(series);rng=np.random.default_rng(936071);boot=values[rng.integers(len(values),size=(10000,len(values)))].mean(1)
                contrasts.append(dict(rival=rival,reference=reference,candidate=candidate,profile=profile,
                    mean_credit_difference=float(values.mean()),paired_board_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
                    independent_boards=len(values),better=int(better),same=int(same),worse=int(worse),series_differences=series))
    result=dict(plan=plan,primary_games=len(rows),executed_games=len(rows)//3*4,duplicate_baseline_replays=len(rows)//3,
        trajectory_sha256=hashlib.sha256((root/'trajectories.jsonl').read_bytes()).hexdigest(),rates=rates,contrasts=contrasts,
        runs=[{k:v for k,v in r.items() if k!='trace'} for r in rows])
    (root/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(primary_games=len(rows),rates=rates,primary=[c for c in contrasts if c['profile']=='each_npc']),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--start',type=int,required=True)
    p.add_argument('--seeds',type=int,default=16);p.add_argument('--samples',type=int,default=8);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--frozen',action='store_true');a=p.parse_args()
    if a.frozen:run(a.root,a.start,a.seeds,a.samples,a.workers)
    else:
        from tools.freeze_experiment import dispatch
        dispatch(a.root,Path(__file__).name,['--start',str(a.start),'--seeds',str(a.seeds),'--samples',str(a.samples),'--workers',str(a.workers)])
