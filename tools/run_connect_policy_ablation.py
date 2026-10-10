"""Fresh board transfer: future-self ablation and stronger misspecified rival."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    a=pair((seed,profile,samples,'incumbent',rival));b=pair((seed,profile,samples,'reflex',rival))
    assert a[0]==b[0],'duplicate baseline must be completely identical'
    a[0].update(condition='baseline',rival=rival)
    a[1].update(condition='incumbent',rival=rival)
    b[1].update(condition='reflex',rival=rival)
    return [a[0],a[1],b[1]]


def run(root,start,seeds,samples,workers):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);base=Path(__file__).resolve().parents[1]
    plan=dict(version='public-base-policy-ablation-v1',seeds=list(range(start,start+seeds)),samples=samples,workers=workers,
        profiles=[p['id'] for p in PROFILES],rivals=['minimax2','minimax4'],conditions=['baseline','incumbent','reflex'],
        owner_budget=asdict(SHORT),primary='incumbent minus baseline own terminal credit; separately for matched minimax2 model and stronger minimax4 actual rival',
        secondary='incumbent minus reflex continuation with same minimax2 forecast, terminal sample count and current personality; every per-profile loss disclosed',
        controls='same public six-drop setup/seat/RNG; same actual owner SHORT controller after one intervention; fixed traits/values, no private opponent identity',
        accounting='each case executes the baseline twice to verify full equality; duplicate baseline not an extra strength observation',
        limits=['known public boards, finite search adversaries not optimal CPUs','minimax4 actual rival deliberately exceeds fixed minimax2 hypothesis; no online type learning added',
                '8 sampled terminal continuations, not exact values','single intervention; no continuous-policy or human-level claim'],
        sources={str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((base/'reflex').glob('*.py'))})
    path=root/'preregister.json'
    if path.exists():raise FileExistsError('fresh ablation required')
    path.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8');rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(worker,(seed,profile,samples,rival)) for seed in plan['seeds'] for profile in range(4) for rival in plan['rivals']]
        with (root/'trajectories.jsonl').open('w',encoding='utf-8') as f:
            for i,future in enumerate(as_completed(jobs),1):
                batch=future.result();rows.extend(batch)
                for row in batch:f.write(json.dumps(row,separators=(',',':'))+'\n')
                f.flush()
                if i%8==0:print('public ablation cases',i,'/',len(jobs),flush=True)
    contrasts=[];rates=[]
    for rival in plan['rivals']:
        for condition in plan['conditions']:
            rr=[r for r in rows if r['rival']==rival and r['condition']==condition]
            rates.append(dict(rival=rival,condition=condition,credit=float(np.mean([r['credit'] for r in rr]))))
        for reference,candidate in [('baseline','incumbent'),('baseline','reflex'),('reflex','incumbent')]:
            for profile in ['each_npc',*plan['profiles']]:
                series=[];better=same=worse=0
                for seed in plan['seeds']:
                    ps=plan['profiles'] if profile=='each_npc' else [profile];delta=[]
                    for p in ps:
                        a=next(r for r in rows if r['seed']==seed and r['profile']==p and r['rival']==rival and r['condition']==reference)
                        b=next(r for r in rows if r['seed']==seed and r['profile']==p and r['rival']==rival and r['condition']==candidate)
                        v=b['credit']-a['credit'];delta.append(v);better+=v>0;same+=v==0;worse+=v<0
                    series.append(float(np.mean(delta)))
                values=np.array(series);rng=np.random.default_rng(934071);boot=values[rng.integers(len(values),size=(10000,len(values)))].mean(1)
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
