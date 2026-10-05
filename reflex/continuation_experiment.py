"""Predeclared bounded self-continuation comparison; no learned policy update.

32 held-out pairs cover every fixed profile, both seats, four goals, two maps,
and three opponent families. The cells are stratified, not a full factorial.
Known seed180 is reported separately. Nested planning is an offline reference,
not an oracle or a recursively expanding rollout inside the game matrix.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import copy
import hashlib
import json
import time
import numpy as np
from .combat_planning import TacticalControl
from .laboratory import profiles
from .validation_experiment import combat, replay, replay_progress
from .alignment_experiment import audit_combat_execution
from .wait_experiment import diagnose
from .recovery_experiment import path_audit

CELLS=(('open','secure','reference'),('choke','eliminate','switch'),
       ('open','either','raider'),('choke','both','reference'))
SEEDS=(281,282)
VARIANTS=('objective','persona-band')


def sources():
    root=Path(__file__).parent
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))}


def continuation_audit(run):
    audit=audit_combat_execution(run);counts=Counter();offsets={}
    for comparison in audit['comparisons']:
        for row in comparison['steps']:
            offset=row['offset']
            if offset==0:continue
            local=Counter();local['modeled_offsets']+=1
            local['same_state']+=row['state']=='same'
            local['state_diverged']+=row['state']=='different'
            for side in ('self','opponent'):
                local[side+'_compared']+=row[side] in ('same','different')
                local[side+'_differences']+=row[side]=='different'
            counts.update(local);offsets.setdefault(offset,Counter()).update(local)
    return dict(all_steps=audit['totals'],continuations=dict(counts),by_offset={str(k):dict(v) for k,v in offsets.items()})


def jobs():
    out=[]
    for terrain,goal,rival in CELLS:
        for p in profiles():
            for seed in SEEDS:
                for variant in VARIANTS:out.append((terrain,goal,rival,p['id'],seed,variant,True,'heldout'))
    for recovery in (False,True):
        for variant in VARIANTS:out.append(('choke','eliminate','switch','steady',180,variant,recovery,'known'))
    return out


def run_job(job, trace_execution=True, capture=True):
    from . import combat_planning as cp
    terrain,goal,rival,persona,seed,variant,recovery,partition=job
    p=next(x for x in profiles() if x['id']==persona);fixed=copy.deepcopy(p)
    control=TacticalControl(self_continuation=variant,recovery_options=recovery,trace_execution=trace_execution)
    original=cp.forecast;work=Counter();snapshots=[]
    def measured(w,actors,contexts,reflex,control=None,root_allowed=None):
        # Count EVERY planner/recovery call, not just the retained audit.
        t=time.perf_counter();f=original(w,actors,contexts,reflex,control,root_allowed)
        work['planner_seconds']+=time.perf_counter()-t;work['calls']+=1
        work['model_transitions']+=f.audit['nodes'];work['recovery_calls']+=root_allowed is not None
        for k,v in f.audit.get('self_model',{}).items():work['persona_'+k]+=v
        if capture and variant=='objective' and root_allowed is None and (
            partition=='heldout' and seed==281 and w.tick==1 or partition=='known' and recovery and w.tick==19):
            from .combat import battle_record
            snapshots.append(dict(before=battle_record(w),actors=actors,contexts=copy.deepcopy(contexts),reflex=copy.deepcopy(reflex)))
        return f
    cp.forecast=measured
    try:
        t=time.perf_counter()
        r=combat(p,seed,terrain,goal,'baseline',learn=True,survival_security=True,
            rival=rival,planner=control,progress_watch=True)
        wall=time.perf_counter()-t
    finally:cp.forecast=original
    assert p==fixed
    r.update(variant=variant,rival=rival,recovery_options=recovery,partition=partition,
        total_work=dict(work),game_seconds=wall,trace_execution=trace_execution)
    checks=replay(r);progress_checks=replay_progress(r)
    r.update(diagnose(r,p));r['forecast_audit']=path_audit(r)
    if trace_execution:r['alignment']=continuation_audit(r)
    return r,snapshots,checks,progress_checks


def aggregate(rows):
    out=[]
    for partition in ('heldout','known'):
        for variant in VARIANTS:
            group=[r for r in rows if r['partition']==partition and r['variant']==variant]
            totals=Counter();alignment=Counter();work=Counter()
            for r in group:
                totals.update({k:r.get(k,0) for k in ('won','lost','ticks','focal_failed_moves','own_proposal_collisions','expired_stop_selections','unresolved_stop_selections','unsupported_root_selections')})
                alignment.update(r['alignment']['continuations']);work.update(r['total_work'])
            out.append(dict(partition=partition,variant=variant,games=len(group),**dict(totals),alignment=dict(alignment),work=dict(work)))
    key=lambda r:(r['partition'],r['scenario'],r['rival'],r['profile'],r['seed'],r['recovery_options'])
    before={key(r):r for r in rows if r['variant']=='objective'};pairs=[]
    for r in rows:
        if r['variant']!='persona-band':continue
        a=before[key(r)]
        pairs.append(dict(condition=list(key(r)),outcome_delta=r['win_credit']-a['win_credit'],
            failed_move_delta=r['focal_failed_moves']-a['focal_failed_moves'],
            expired_stop_delta=r.get('expired_stop_selections',0)-a.get('expired_stop_selections',0),
            self_compared_before=a['alignment']['continuations'].get('self_compared',0),
            self_compared_after=r['alignment']['continuations'].get('self_compared',0),
            self_mismatch_before=a['alignment']['continuations'].get('self_differences',0),
            self_mismatch_after=r['alignment']['continuations'].get('self_differences',0)))
    return out,pairs


def experiment(output,workers=4,progress=print):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    frozen=sources();work=jobs();results={};snapshots=[];checks=updates=0;t=time.perf_counter()
    prereg=dict(format='self-continuation-preregister-v1',jobs=work,
        seeds_status='281/282 reserved before candidate matrix; 180 known development case',
        selection='No tuning after heldout runs; no win-rate-only promotion.',
        gates=['fixed axes/owner/legal/public-only contracts; original tiers unchanged',
            'reduce same-state depth>0 self mismatch with denominator reported',
            'no personality-level outcome regression or unresolved/failed means worsening hidden by totals',
            'trace-off runtime and nested reference reported separately; no latency target claimed'],
        configs={v:asdict(TacticalControl(self_continuation=v,recovery_options=True)) for v in VARIANTS},source_hashes=frozen)
    (output/'preregister.json').write_text(json.dumps(prereg,indent=2)+'\n')
    with (output/'trajectories.jsonl').open('w') as stream,ProcessPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(run_job,j):i for i,j in enumerate(work)}
        for future in as_completed(pending):
            r,s,check,update=future.result();checks+=check;updates+=update
            stream.write(json.dumps(r)+'\n');stream.flush()
            identity={k:r[k] for k in ('scenario','profile','seed','rival','partition','recovery_options')}
            snapshots.extend(dict(**identity,**x) for x in s)
            results[pending[future]]={k:v for k,v in r.items() if k!='trace'}
            if progress:progress(f'{len(results)}/{len(work)}: {r["scenario"]} {r["profile"]} {r["seed"]} {r["variant"]}; won={r["won"]}; {r["game_seconds"]:.1f}s',flush=True)
    assert frozen==sources(),'source changed during frozen experiment'
    # Preserve fixed snapshots even if a later report formatter fails.
    (output/'snapshots.json').write_text(json.dumps(snapshots)+'\n')
    rows=[results[i] for i in range(len(work))];summary,pairs=aggregate(rows)
    result=dict(format='persona-self-continuation-v1',games=len(rows),rule_checks=checks,purpose_checks=updates,
        elapsed_seconds=time.perf_counter()-t,summary=summary,pairs=pairs,runs=rows,
        source_hashes=frozen,defaults_changed=False,learning_changed=False)
    (output/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',default='reflex_artifacts/persona_continuation');p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();experiment(a.output,a.workers)
