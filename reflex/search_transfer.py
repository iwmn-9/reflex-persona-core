"""Fixed shared-continuation candidate versus the verified reflex continuation."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
from .core import digest
from .laboratory import profiles
from .observed_transfer import run,aggregate
from .purpose_experiment import source_hashes


def jobs():
    diagnostic=[dict(genre='delivery',limit=8,capacity=2,supply=3),
                dict(genre='delivery',limit=10,capacity=3,supply=2)]
    holdout=[dict(genre='delivery',limit=11,capacity=3,supply=3),
             dict(genre='delivery',limit=10,capacity=2,supply=2),
             dict(genre='resources',route='mixed',limit=14),
             dict(genre='auction',prizes=[9,14,6],budget=11,rival_shift=True),
             dict(genre='combat',hp=7,limit=7,distance=4)]
    for split,specs in (('diagnostic',diagnostic),('holdout',holdout)):
        for s in specs:
            for p in profiles():
                for variant in ('verified','search'):yield split,s,p,variant,280,6


def worker(job):
    split,s,p,v,seed,h=job
    r=run(s,p,'verified',seed,h,search=v in ('search','continuity'),continuity=v=='continuity',width=2,depth=2)
    r.update(split=split,variant=v)
    return r


def experiment(output,progress=None,workers=4,*,tasks=None,candidate='search',reference='verified',registration=None):
    if type(workers) is not int or not 1<=workers<=8:raise ValueError('bounded workers required')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if any((output/n).exists() for n in ('preregister.json','evaluation.json','trajectories.jsonl')):
        raise FileExistsError('frozen evidence exists; use a fresh directory')
    fixed=source_hashes();tasks=list(jobs()) if tasks is None else list(tasks)
    pre=dict(format='search-transfer-v1',source_hashes=fixed,jobs=tasks,workers=workers,
        parameters=dict(horizon=6,width=2,depth=2,max_regret=.15,proof_margin=0.),
        status='delivery diagnostic conditions previously exposed; new holdout parameter combinations frozen before comparisons, same four known rule engines; not unseen game transfer',
        adoption='optional only if all holdout pairs preserve status, goal value and survival, do not increase shortages or reachable-route loss, and logistics success improves; reject broad adoption on any regression; no post-holdout tuning')
    if registration is not None:pre.update(registration)
    pre.update(candidate=candidate,reference=reference)
    (output/'preregister.json').write_text(json.dumps(pre,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    runs=[]
    with (output/'trajectories.jsonl').open('w',encoding='utf-8',newline='\n') as stream:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for r in pool.map(worker,tasks):
                stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush()
                runs.append({k:v for k,v in r.items() if k!='trace'})
                if progress:progress(f'{len(runs)}/{len(tasks)} fixed-source episodes',flush=True)
    assert source_hashes()==fixed
    pairs=[]
    for a in runs:
        if a['variant']!=candidate:continue
        b=next(b for b in runs if b['variant']==reference and b['split']==a['split'] and b['spec']==a['spec'] and b['profile']==a['profile'] and b['seed']==a['seed'])
        pairs.append(dict(split=a['split'],genre=a['spec']['genre'],spec=a['spec'],profile=a['profile'],
            baseline_status=b['status'],candidate_status=a['status'],goal_delta=a['goal_value']-b['goal_value'],
            survival_regression=b['surviving'] is True and a['surviving'] is False,
            shortages_delta=(a['shortages'] or 0)-(b['shortages'] or 0),
            lost_route_delta=(a['solvable_route_lost'] or 0)-(b['solvable_route_lost'] or 0)))
    hold=[p for p in pairs if p['split']=='holdout']
    regressions=[p for p in hold if p['goal_delta']<0 or p['survival_regression'] or p['shortages_delta']>0 or p['lost_route_delta']>0 or
        (p['baseline_status']=='success' and p['candidate_status']!='success')]
    gained=any((p['genre']=='delivery' or candidate=='continuity') and p['baseline_status']!='success' and p['candidate_status']=='success' for p in hold)
    diagnostic_recovered=candidate!='continuity' or any(r['split']=='diagnostic' and r['variant']==candidate and
        r['spec']==dict(genre='resources',route='mixed',limit=14) and r['profile']=='ego' and r['status']=='success' for r in runs)
    result=dict(format=pre['format'],episodes=len(runs),source_hashes=fixed,preregister_digest=digest(pre),
        summary=aggregate(runs),runs=runs,pairs=pairs,holdout_regressions=regressions,
        broad_adoption_passed=not regressions and gained and diagnostic_recovered,
        diagnostic_recovered=diagnostic_recovered,default_policy_changed=False,
        coefficients_changed=False,training_performed=False,cloud_runtime_used=False)
    (output/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    return result
