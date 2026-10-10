"""Fixed public boards; paired actual games with one root policy improvement."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from reflex.board_models import ConnectPosition, connect_referee
from reflex.connect_policy_rollout import incumbent, decide
from reflex.board_planning import SHORT
from reflex.laboratory import PROFILES
from reflex.monte_carlo_comparison import minimax2
from reflex.monte_carlo import RolloutBudget
from reflex.core import TRAITS,VALUES
from reflex.tabletop_trials import score


def worker(job):
    seed,profile,samples,*extra=job
    owner_policy=extra[0] if extra else 'incumbent';rival=extra[1] if len(extra)>1 else 'minimax2'
    rival_forecast=extra[2] if len(extra)>2 else 'minimax2'
    if rival not in ('minimax2','minimax4'):raise ValueError('registered actual rival required')
    from reflex.connect_objective import choose as minimax4
    rival_controller=minimax2 if rival=='minimax2' else minimax4
    p=PROFILES[profile];start=ConnectPosition()
    rng=np.random.default_rng(seed+331071)
    for _ in range(6):
        if start.winner() is not None:break
        names=start.legal();start=start.play(names[int(rng.integers(len(names)))])
    assert start.winner() is None
    seat=seed%2;rows=[]
    for candidate in (False,True):
        s=start;state=None;trace=[];used=False;episode=f'public-transfer-{seed}-{profile}'
        rival_rng=np.random.default_rng(seed+82133)
        while s.winner() is None and sum(s.heights)<42:
            tick=len(trace);actor=s.turn;stats=None
            if actor==seat:
                if candidate and not used:
                    c,d,stats=decide(s,p,seed,tick,episode,state,
                        rollout=RolloutBudget(samples=samples,min_samples=samples,max_nodes=100000,max_steps=42,rollout_policy='persona'),
                        owner_policy=owner_policy,rival_policy=rival_forecast)
                    used=True
                else:c,d,stats=incumbent(s,p,seed,tick,episode,state)
                assert c['personality']==dict(zip(TRAITS,p['traits'])) and c['values']=={k:float(p['values'].get(k,0)) for k in VALUES}
                replay,_=score(c);assert replay==d
                move=d['action_id'];state=d['next_state']
            else:move=rival_controller(s,rival_rng)
            w,legal,_=connect_referee(s);assert w==s.winner() and move in legal
            after=s.play(move);w,legal,_=connect_referee(after);assert w==after.winner() and legal==after.legal()
            trace.append(dict(tick=tick,actor=actor,before=asdict(s),after=asdict(after),action=move,stats=stats));s=after
        w=s.winner();credit=.5 if w is None else float(w==seat)
        rows.append(dict(seed=seed,profile=p['id'],seat=seat,candidate=candidate,credit=credit,winner=w,trace=trace))
    return rows


def run(root,start,seeds,samples,workers):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);base=Path(__file__).resolve().parents[1]
    plan=dict(version='public-spatial-base-policy-v1',seeds=list(range(start,start+seeds)),profiles=[p['id'] for p in PROFILES],
        samples=samples,owner_budget=asdict(SHORT),workers=workers,
        design='six public setup drops, all four fixed profiles and both seats; one root intervention then actual incumbent finite-Policy SHORT re-searches; independent minimax2 rival',
        primary='own terminal credit; paired board-seed cluster',
        limits=['known public opponent family, no inferred opponent identity','new finite-Policy SHORT baseline; not a comparison with historical lexicographic board studies','single root not continuously upgraded policy','no human-level or unseen-commercial-game claim'],
        sources={str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((base/'reflex').glob('*.py'))})
    path=root/'preregister.json'
    if path.exists():raise FileExistsError('fresh public study required')
    path.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8');rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(worker,(seed,profile,samples)) for seed in plan['seeds'] for profile in range(4)]
        with (root/'trajectories.jsonl').open('w',encoding='utf-8') as f:
            for i,future in enumerate(as_completed(jobs),1):
                pair=future.result();rows.extend(pair)
                for row in pair:f.write(json.dumps(row,separators=(',',':'))+'\n')
                f.flush();print('public paired cases',i,'/',len(jobs),flush=True)
    deltas=[];profiles={}
    for profile in plan['profiles']:
        values=[];changes=0
        for seed in plan['seeds']:
            a=next(r for r in rows if r['seed']==seed and r['profile']==profile and not r['candidate'])
            b=next(r for r in rows if r['seed']==seed and r['profile']==profile and r['candidate'])
            values.append(b['credit']-a['credit'])
            changes+=sum(t['stats'] is not None and t['stats'].get('policy_rollout',{}).get('changed',False) for t in b['trace'])
        profiles[profile]=dict(mean_credit_difference=float(np.mean(values)),changed=changes,differences=values)
    for seed in plan['seeds']:
        deltas.append(float(np.mean([profiles[p]['differences'][plan['seeds'].index(seed)] for p in plan['profiles']])))
    v=np.array(deltas);rng=np.random.default_rng(931071);boot=v[rng.integers(len(v),size=(10000,len(v)))].mean(1)
    summary=dict(games=len(rows),independent_boards=len(v),mean_credit_difference=float(v.mean()),
        paired_board_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),profiles=profiles,
        actual_owner_policy_replays=sum(t['actor']==r['seat'] for r in rows for t in r['trace']),
        hypothetical_owner_searches=sum(t['stats'].get('policy_rollout',{}).get('owner_searches',0) for r in rows for t in r['trace'] if t['stats']))
    result=dict(plan=plan,summary=summary,runs=[{k:v for k,v in r.items() if k!='trace'} for r in rows],
        trajectory_sha256=hashlib.sha256((root/'trajectories.jsonl').read_bytes()).hexdigest())
    (root/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--start',required=True,type=int)
    p.add_argument('--seeds',type=int,default=8);p.add_argument('--samples',type=int,default=8);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--frozen',action='store_true');a=p.parse_args()
    if a.frozen:run(a.root,a.start,a.seeds,a.samples,a.workers)
    else:
        from tools.freeze_experiment import dispatch
        dispatch(a.root,Path(__file__).name,['--start',str(a.start),'--seeds',str(a.seeds),'--samples',str(a.samples),'--workers',str(a.workers)])
