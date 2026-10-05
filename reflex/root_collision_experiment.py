"""Frozen root-only collision ablation; no horizon continuation changes."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import copy
import hashlib
import json
import time
from .combat_planning import TacticalControl
from .continuation_experiment import sources
from .laboratory import profiles
from .validation_experiment import combat, replay, replay_progress
from .wait_experiment import diagnose

CELLS=(('open','secure','reference'),('choke','eliminate','switch'),
       ('open','either','raider'),('choke','both','reference'))
SEEDS=(391,392)
VARIANTS=('baseline','root-completion')


def jobs():
    return [(terrain,goal,rival,p['id'],seed,variant)
        for terrain,goal,rival in CELLS for p in profiles() for seed in SEEDS for variant in VARIANTS]


def movement_and_completion(run):
    from .combat import alive,battle_from_record,progress
    counts=Counter();max_regret=0.;changed_owners=Counter()
    for row in run['trace']:
        before=battle_from_record(row['before']);after=battle_from_record(row['after'])
        team=run['seed']%2;ours=alive(before,team)
        choices={int(i):k for i,k in row['choices'].items()}
        for i in ours:
            key=choices[i]
            if not key.startswith('move:'):continue
            counts['move_attempts']+=1
            failed=before.units[i].pos==after.units[i].pos
            if not failed:continue
            counts['failed_moves']+=1
            own=any(j!=i and choices.get(j)==key for j in ours)
            enemy=any(choices.get(j)==key for j in alive(before,1-team))
            counts['both_collision_moves' if own and enemy else 'friendly_collision_moves' if own else
                'opponent_collision_moves' if enemy else 'unclassified_failed_moves']+=1
        audit=row.get('planning',{}).get('root_completion')
        if audit is not None:
            counts['completion_calls']+=1
            adapter=audit.get('adapter',{})
            counts['completion_collision_ticks']+=bool(adapter.get('colliding'))
            counts['completion_feasible_ticks']+=bool(adapter.get('feasible'))
            counts['completion_search_combinations']+=adapter.get('combinations',0)
            counts['completion_adopted_ticks']+=audit['adopted']
            counts['completion_changed_actors']+=audit['changed']
            counts['completion_unresolved_ticks']+=bool(adapter.get('colliding')) and not audit['adopted']
            for i,(a,b) in zip(ours,zip(audit['before'],audit['after'])):
                if a!=b:changed_owners[str(i)]+=1
            max_regret=max(max_regret,max(audit.get('persona_regret',[]),default=0.))
            counts['persona_band_violations']+=sum(x>.025+1e-12 for x in audit.get('persona_regret',[]))
            for e,a,b in zip(row.get('purpose',()),audit['before'],audit['after']):
                # Read the exact before-observation ProgressWatch and labels.
                from .progress import ProgressWatch,Activity,PurposeRequest
                w=ProgressWatch.from_record(e['scope'],e['before']);p=e['request']
                request=PurposeRequest(p['level'],p['readiness'],{k:Activity(**v) for k,v in p['activities'].items()},p['maintained'])
                c=dict(scope=e['scope'],actions=[dict(id=k,legal=True,known_failure=False) for k in p['activities']])
                _,raw=w.mask(c,request)
                counts['introduced_blocked_roots']+=a!=b and b in raw['blocked']
    final=battle_from_record(run['trace'][-1]['after'])
    counts['unwon_unlost_games']=int(not run['won'] and not run['lost'])
    return dict(movement=dict(counts),completion_max_persona_regret=max_regret,
        completion_changed_owners=dict(changed_owners),
        final_eliminate_progress=progress(final,run['seed']%2,'eliminate'),
        final_secure_progress=progress(final,run['seed']%2,'secure'))


def run_job(job,trace_execution=True):
    from . import combat_planning as cp
    terrain,goal,rival,persona,seed,variant=job
    profile=next(p for p in profiles() if p['id']==persona);fixed=copy.deepcopy(profile)
    control=TacticalControl(root_completion=variant=='root-completion',recovery_options=True,trace_execution=trace_execution)
    original=cp.forecast;work=Counter()
    def measured(w,actors,contexts,reflex,control=None,root_allowed=None):
        start=time.perf_counter();result=original(w,actors,contexts,reflex,control,root_allowed)
        work['planner_seconds']+=time.perf_counter()-start
        work['calls']+=1;work['model_transitions']+=result.audit['nodes']
        work['recovery_calls']+=root_allowed is not None
        assert 'self_model' not in result.audit
        return result
    cp.forecast=measured
    try:
        start=time.perf_counter()
        r=combat(profile,seed,terrain,goal,'baseline',learn=True,survival_security=True,
            rival=rival,planner=control,progress_watch=True)
        wall=time.perf_counter()-start
    finally:cp.forecast=original
    assert profile==fixed
    r.update(variant=variant,rival=rival,game_seconds=wall,total_work=dict(work),trace_execution=trace_execution)
    r.update(diagnose(r,profile));r.update(movement_and_completion(r))
    checks=replay(r);updates=replay_progress(r)
    return r,checks,updates



def paired_divergence(a,b):
    """Locate the first real divergence before attributing later outcomes."""
    team=a['seed']%2
    from .combat import alive,battle_from_record
    for left,right in zip(a['trace'],b['trace']):
        assert left['before']==right['before'],'world diverged before its first changed choice'
        if left['choices']!=right['choices']:
            completion=right.get('planning',{}).get('root_completion',{})
            actors=alive(battle_from_record(left['before']),team)
            old=[left['choices'].get(i,left['choices'].get(str(i))) for i in actors]
            new=[right['choices'].get(i,right['choices'].get(str(i))) for i in actors]
            assert completion.get('adopted') and completion['before']==old and completion['after']==new
            return dict(tick=left['before']['tick'],same_public_before=True,completion_caused_first_difference=True,
                actors=list(actors),before=old,after=new)
        assert left['after']==right['after'],'same choices/seed produced different real transition'
    assert len(a['trace'])==len(b['trace'])
    return dict(tick=None,same_trajectory=True,completion_caused_first_difference=False)


def aggregate(rows):
    identity=lambda r:(r['scenario'],r['profile'],r['seed'],r['rival'])
    baseline={identity(r):r for r in rows if r['variant']=='baseline'}
    pairs=[]
    for r in rows:
        if r['variant']!='root-completion':continue
        a=baseline[identity(r)]
        pairs.append(dict(condition=list(identity(r)),win_delta=r['win_credit']-a['win_credit'],
            failed_move_delta=r['movement'].get('failed_moves',0)-a['movement'].get('failed_moves',0),
            friendly_collision_delta=r['movement'].get('friendly_collision_moves',0)-a['movement'].get('friendly_collision_moves',0),
            unresolved_stop_delta=r.get('unresolved_stop_selections',0)-a.get('unresolved_stop_selections',0),
            eliminate_delta=r['final_eliminate_progress']-a['final_eliminate_progress'],
            secure_delta=r['final_secure_progress']-a['final_secure_progress']))
    groups=[]
    for dimension in ('all','profile','scenario','rival'):
        for group in ('all',) if dimension=='all' else sorted({r[dimension] for r in rows}):
            for variant in VARIANTS:
                selected=[r for r in rows if r['variant']==variant and (dimension=='all' or r[dimension]==group)]
                metrics=Counter();movement=Counter();work=Counter()
                for r in selected:
                    metrics.update({k:r.get(k,0) for k in ('won','lost','ticks','decisions','planned_ticks','planning_declines',
                        'expired_stop_selections','unresolved_stop_selections','unsupported_root_selections',
                        'final_eliminate_progress','final_secure_progress')})
                    movement.update(r['movement']);work.update(r['total_work'])
                attempts=movement.get('move_attempts',0)
                groups.append(dict(dimension=dimension,group=group,variant=variant,games=len(selected),
                    metrics=dict(metrics),movement=dict(movement),work=dict(work),
                    failed_move_rate=movement.get('failed_moves',0)/attempts if attempts else None,
                    friendly_collision_rate=movement.get('friendly_collision_moves',0)/attempts if attempts else None,
                    max_persona_regret=max(r['completion_max_persona_regret'] for r in selected),
                    longest_stall=max(r['longest_observed_stall'] for r in selected)))
    return groups,pairs


def experiment(output,workers=4):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    frozen=sources();work=jobs()
    prereg=dict(format='root-collision-preregister-v1',base='3ca35a3f0eccc8665c7f39596b52abe8e3ff3149',
        jobs=work,seeds_status='391/392 reserved before implementation and heldout results; prior 180/281/282 development only',
        frozen_candidate='Declined fallback only; objective continuation unchanged; only existing friendly colliders may change; original immediate Policy tier and .025 band under final progress/waste masks; no newly introduced raw-progress-blocked root; minimum changed actors then highest sum Policy score then canonical roots; abstain if no complete feasible combination',
        selection='No thresholds or candidate tuning on heldout; default remains off pending review',
        serial_timing=[['choke','eliminate','switch','care',281,v] for v in VARIANTS]+[['choke','both','reference','care',282,v] for v in VARIANTS],
        design_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/'evidence/root_collision/design.json').read_bytes()).hexdigest(),
        source_hashes=frozen,configs={v:asdict(TacticalControl(root_completion=v=='root-completion',recovery_options=True,trace_execution=True)) for v in VARIANTS})
    (output/'preregister.json').write_text(json.dumps(prereg,indent=2)+'\n')
    results={};checks=updates=0;started=time.perf_counter()
    with (output/'trajectories.jsonl').open('w') as stream,ProcessPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(run_job,job):i for i,job in enumerate(work)}
        for future in as_completed(pending):
            r,check,update=future.result();checks+=check;updates+=update
            stream.write(json.dumps(r)+'\n');stream.flush();results[pending[future]]=r
            print(f'{len(results)}/{len(work)} {r["scenario"]} {r["profile"]} {r["seed"]} {r["variant"]} win={r["won"]} {r["game_seconds"]:.1f}s',flush=True)
    assert frozen==sources(),'source changed during heldout experiment'
    raw=[results[i] for i in range(len(work))]
    divergences=[dict(condition=list(work[i][:-1]),**paired_divergence(raw[i],raw[i+1])) for i in range(0,len(work),2)]
    rows=[{k:v for k,v in r.items() if k!='trace'} for r in raw];summary,pairs=aggregate(rows)
    result=dict(format='root-collision-v1',games=len(rows),source_hashes=frozen,
        elapsed_seconds=time.perf_counter()-started,rule_checks=checks,purpose_checks=updates,
        summary=summary,pairs=pairs,paired_divergences=divergences,runs=rows,defaults_changed=False,continuation_changed=False)
    (output/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',default='reflex_artifacts/root_collision');p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();experiment(a.output,a.workers)
