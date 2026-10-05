"""Predeclared congestion diagnostics, not a new movement or personality policy.

Authored starts supply legal opportunities; forced probes and naturally selected
full games stay separate. Enemy hypotheses accept public pre-states only. Actual
revealed choices are used solely for post-decision scoring and rule replay.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
import copy
import hashlib
import json
import time
import numpy as np
from .combat import (Battle, Unit, alive, legal, resolve, battle_record,
    battle_from_record, hit_probability, zone_distance, firing_distance)
from .combat_planning import TacticalControl, routes, tactical_joint
from .continuation_experiment import sources
from .core import Policy, compile_batch, digest, FEATURES
from .laboratory import profiles
from .root_collision_experiment import movement_and_completion, paired_divergence as prior_paired_divergence
from .validation_experiment import combat, replay, replay_progress
from .wait_experiment import diagnose

BASE='5af568ecd93cdb655fd1bceccc4c261e6c5f31ea'
DESIGN=Path(__file__).resolve().parents[1]/'evidence/controlled_congestion/design.json'
VARIANTS=('baseline','root-completion')
SCENES=('friendly_convergence','enemy_convergence','mixed_junction')
HELDOUT=(('open','secure','reference'),('choke','eliminate','switch'),
         ('open','either','raider'),('choke','both','reference'))


def authored_world(scene, goal, team, limit=24):
    """Public, legal initial state; mirror ownership and geometry for seat one."""
    d=json.loads(DESIGN.read_text())['controlled_geometry'][scene]
    if team not in (0,1):raise ValueError('two game-owned teams required')
    point=lambda p:(8-p[0],p[1]) if team else tuple(p)
    positions={team:d['ours'],1-team:d['theirs']}
    units=tuple(Unit(t,*point(p)) for t in (0,1) for p in positions[t])
    w=Battle(units,walls=tuple(sorted(point(p) for p in d['walls'])),
        covers=tuple(sorted(point(p) for p in d['covers'])),goal=goal,limit=limit)
    occupied=[u.pos for u in w.units]
    assert len(set(occupied))==6 and not set(occupied)&set(w.walls)
    assert all(legal(w,i) for i in range(6))
    return w


def jobs():
    controlled=[('controlled',scene,goal,'reference',p['id'],seed,v)
        for scene,goal,p,seed,v in product(SCENES,('secure','both'),profiles(),(591,592),VARIANTS)]
    heldout=[('heldout',terrain,goal,rival,p['id'],611+(i+cell)%2,v)
        for cell,(terrain,goal,rival) in enumerate(HELDOUT) for i,p in enumerate(profiles()) for v in VARIANTS]
    return controlled+heldout


def enemy_hypotheses(w, team):
    """Exactly the two existing first-step forecast hypotheses, no actual inputs."""
    enemies=alive(w,1-team);r=routes(w,1-team);route=r[0] if r else 'secure'
    out=[]
    for aggressive in (False,True):
        choices=tactical_joint(w,enemies,route,aggressive=aggressive,coordinate=True)
        if aggressive:
            for j in enemies:
                shots=[k for k in legal(w,j) if k.startswith('shoot:')]
                if shots:choices[j]=max(shots,key=lambda k:(w.units[int(k.split(':')[1])].hp<=3,
                    hit_probability(w,j,int(k.split(':')[1])),-w.units[int(k.split(':')[1])].hp,k))
        out.append(choices)
    return out


def opportunities(w,team):
    """Potential same-destination conflicts, not observed moves or failures."""
    own=Counter(k for i in alive(w,team) for k in legal(w,i) if k.startswith('move:'))
    enemy=Counter(k for i in alive(w,1-team) for k in legal(w,i) if k.startswith('move:'))
    return dict(friendly_only=sum(n>=2 and enemy[k]==0 for k,n in own.items()),
        opponent_only=sum(n==1 and enemy[k]>0 for k,n in own.items()),
        mixed=sum(n>=2 and enemy[k]>0 for k,n in own.items()))


def root_conflicts(roots):
    counts=Counter(k for k in roots if k.startswith('move:'))
    return sum(n for n in counts.values() if n>1)


def goal_distances(w,actor,pos):
    enemies=tuple(w.units[i].pos for i in alive(w,1-w.units[actor].team))
    return dict(secure=zone_distance(w,pos),eliminate=firing_distance(w.walls,pos,enemies))


def trace_diagnostics(run):
    """Offline measured exposure; no diagnostic feeds the decision or learning."""
    counts=Counter();route_counts=Counter();predictions=[];team=run['seed']%2;max_combinations=0
    for row in run['trace']:
        w=battle_from_record(row['before']);after=battle_from_record(row['after'])
        ours=alive(w,team);choices={int(i):k for i,k in row['choices'].items()}
        for kind,n in opportunities(w,team).items():
            counts['legal_'+kind+'_destinations']+=n;counts['legal_'+kind+'_ticks']+=n>0
        plan=row.get('planning',{});completion=plan.get('root_completion',{})
        max_combinations=max(max_combinations,completion.get('adapter',{}).get('combinations',0))
        roots=completion.get('before',[choices[i] for i in ours])
        conflict=root_conflicts(roots)
        declined=not plan.get('adopted',False)
        counts['selected_friendly_conflict_ticks']+=conflict>0
        counts['declined_friendly_conflict_ticks']+=declined and conflict>0
        counts['declined_friendly_conflict_moves']+=conflict if declined else 0
        counts['adopted_plan_friendly_conflict_ticks']+=not declined and conflict>0
        if conflict and completion:
            counts['completion_no_eligible_band_ticks']+='outside final immediate persona band' in completion['reason']
            counts['completion_no_feasible_ticks']+=completion.get('adapter',{}).get('feasible')==0
        # Construct public forecasts before reading revealed opposing choices.
        hypotheses=enemy_hypotheses(w,team)
        actual={i:choices[i] for i in alive(w,1-team)}
        counts['enemy_hypothesis_actor_trials']+=len(actual)
        counts['enemy_action_covered']+=sum(any(h[i]==k for h in hypotheses) for i,k in actual.items())
        for i in ours:
            key=choices[i];route=row.get('selected_routes',{}).get(i,row.get('selected_routes',{}).get(str(i)))
            route_counts[route or 'missing']+=1
            if not key.startswith('move:'):continue
            probability=sum(key in h.values() for h in hypotheses)/len(hypotheses)
            contested=key in actual.values()
            counts['occupancy_prediction_trials']+=1
            counts['occupancy_probability_sum']+=probability
            counts['occupancy_actual_contests']+=contested
            counts['occupancy_brier_sum']+=(probability-float(contested))**2
            counts['occupancy_zero_probability_misses']+=contested and probability==0
            counts['occupancy_positive_false_alarms']+=not contested and probability>0
            old=goal_distances(w,i,w.units[i].pos)
            target=goal_distances(w,i,tuple(map(int,key.split(':')[1:])))
            actual_pos=goal_distances(w,i,after.units[i].pos)
            for goal in ('secure','eliminate'):
                counts['intended_'+goal+'_distance_gain']+=old[goal]-target[goal]
                counts['realized_'+goal+'_distance_gain']+=old[goal]-actual_pos[goal]
            if route in old:
                counts['chosen_route_moves']+=1
                counts['chosen_route_improving_moves']+=target[route]<old[route]
                counts['chosen_route_nonimproving_moves']+=target[route]>=old[route]
                counts['other_route_only_improving_moves']+=target[route]>=old[route] and any(target[g]<old[g] for g in old if g!=route)
        execution=plan.get('execution')
        if execution:
            for n,sample in enumerate(execution['samples']):
                if sample:
                    assert sample[0]['opponent']=={str(i):k for i,k in hypotheses[n%2].items()}
                    counts['saved_first_step_hypothesis_checks']+=1
        predictions.append(dict(tick=w.tick,public_before=digest(row['before']),
            hypotheses=hypotheses,actual_revealed=actual))
    return dict(diagnostics=dict(counts),selected_routes=dict(route_counts),enemy_predictions=predictions,completion_max_combinations=max_combinations)


def forced_probes():
    """Declared interventions test rules and unchanged admissibility, not efficacy."""
    from .combat import make_context
    from .root_completion import constrained_completion,combat_completion
    rows=[]
    for scene,team,p in product(SCENES,(0,1),profiles()):
        w=authored_world(scene,'both',team)
        d=json.loads(DESIGN.read_text())['controlled_geometry'][scene]
        x,y=d['target'];x=8-x if team else x;key=f'move:{x}:{y}'
        choices={i:key if key in legal(w,i) else 'guard' for i in range(6)}
        after,audit=resolve(w,choices,0);ours=alive(w,team)
        cs=[make_context(w,i,p,700+team,'secure',survival_security=True) for i in ours]
        b=compile_batch(cs);decision=Policy().decide(b,False)
        forced=replace(decision,action=np.array([b.ids[n].index(choices[i]) for n,i in enumerate(ours)])).records(b)
        exact=np.zeros(b.legal.shape+(len(FEATURES),),dtype=bool);exact[:,:,FEATURES.index('cost')]=True
        records,completion=constrained_completion(cs,forced,[set(ids) for ids in b.ids],exact,Policy(),
            lambda c,r,o:combat_completion(w,ours,c,r,o))
        selected={**choices,**{i:r['action_id'] for i,r in zip(ours,records)}}
        changed,changed_audit=resolve(w,selected,0)
        rows.append(dict(scene=scene,team=team,profile=p['id'],scope='forced mechanism probe, no memory updates',
            opportunities=opportunities(w,team),before=battle_record(w),choices=choices,
            actual_collisions=audit['collisions'],own_failed_moves=sum(choices[i].startswith('move:') and w.units[i].pos==after.units[i].pos for i in ours),
            completion=completion,completed_choices=selected,completed_collisions=changed_audit['collisions'],
            completed_own_failed_moves=sum(selected[i].startswith('move:') and w.units[i].pos==changed.units[i].pos for i in ours)))
    return rows


def run_job(job,trace_execution=True):
    from . import combat_planning as cp
    partition,scene,goal,rival,persona,seed,variant=job
    profile=next(p for p in profiles() if p['id']==persona);fixed=copy.deepcopy(profile)
    control=TacticalControl(root_completion=variant=='root-completion',recovery_options=True,trace_execution=trace_execution)
    initial=authored_world(scene,goal,seed%2) if partition=='controlled' else None
    original=cp.forecast;work=Counter()
    def measured(w,actors,contexts,reflex,control=None,root_allowed=None):
        start=time.perf_counter();result=original(w,actors,contexts,reflex,control,root_allowed)
        work['planner_seconds']+=time.perf_counter()-start;work['calls']+=1
        work['model_transitions']+=result.audit['nodes'];work['recovery_calls']+=root_allowed is not None
        return result
    cp.forecast=measured
    try:
        start=time.perf_counter()
        r=combat(profile,seed,scene,goal,'baseline',learn=True,survival_security=True,
            rival=rival,planner=control,progress_watch=True,initial_world=initial,trace_roots=True)
        wall=time.perf_counter()-start
    finally:cp.forecast=original
    assert profile==fixed
    r.update(partition=partition,variant=variant,rival=rival,game_seconds=wall,total_work=dict(work),trace_execution=trace_execution)
    r.update(diagnose(r,profile));r.update(movement_and_completion(r));r.update(trace_diagnostics(r))
    return r,replay(r),replay_progress(r)


def paired_divergence(a,b):
    result=prior_paired_divergence(a,b)
    if result['tick'] is not None:
        left=next(r for r in a['trace'] if r['before']['tick']==result['tick'])
        right=next(r for r in b['trace'] if r['before']['tick']==result['tick'])
        enemies=alive(battle_from_record(left['before']),1-a['seed']%2)
        choices=lambda r:{int(i):k for i,k in r['choices'].items()}
        assert all(choices(left)[i]==choices(right)[i] for i in enemies),'opponent differed on shared first changed state'
        result['same_revealed_enemy_choices']=True
    return result


def totals(rows):
    movement=Counter();metrics=Counter();diagnostics=Counter();work=Counter();routes=Counter()
    for r in rows:
        movement.update(r['movement']);diagnostics.update(r['diagnostics']);work.update(r['total_work']);routes.update(r['selected_routes'])
        metrics.update({k:r.get(k,0) for k in ('won','lost','ticks','decisions','planning_declines','planned_ticks',
            'expired_stop_selections','unresolved_stop_selections','unsupported_root_selections',
            'final_eliminate_progress','final_secure_progress')})
    attempts=movement.get('move_attempts',0);decisions=metrics.get('decisions',0)
    rate=lambda n,d:n/d if d else None
    return dict(games=len(rows),movement=dict(movement),metrics=dict(metrics),diagnostics=dict(diagnostics),work=dict(work),
        selected_routes=dict(routes),completion_max_combinations=max((r.get('completion_max_combinations',0) for r in rows),default=0),longest_stall=max((r['longest_observed_stall'] for r in rows),default=0),
        target_failure_rate=rate(movement['friendly_collision_moves']+movement['both_collision_moves'],attempts),
        all_failure_rate=rate(movement['failed_moves'],attempts),
        unresolved_stop_rate=rate(metrics['unresolved_stop_selections'],decisions),
        enemy_occupancy_brier=rate(diagnostics['occupancy_brier_sum'],diagnostics['occupancy_prediction_trials']),
        exposure_games=sum(r['diagnostics'].get('declined_friendly_conflict_ticks',0)>0 for r in rows))


def summarize(rows):
    summary=[]
    for partition in ('controlled','heldout'):
        subset=[r for r in rows if r['partition']==partition]
        for dimension in ('all','profile','scenario'):
            groups=('all',) if dimension=='all' else sorted({r[dimension] for r in subset})
            for group in groups:
                for variant in VARIANTS:
                    chosen=[r for r in subset if r['variant']==variant and (dimension=='all' or r[dimension]==group)]
                    summary.append(dict(partition=partition,dimension=dimension,group=group,variant=variant,**totals(chosen)))
    def get(partition,variant):return next(r for r in summary if r['partition']==partition and r['dimension']=='all' and r['variant']==variant)
    a,b=get('controlled','baseline'),get('controlled','root-completion')
    exposure=a['diagnostics'].get('declined_friendly_conflict_ticks',0)>=12 and a['exposure_games']>=6
    failures=[]
    if not exposure:failures.append('insufficient natural baseline declined-conflict exposure')
    if not(a['target_failure_rate'] and b['target_failure_rate'] is not None and b['target_failure_rate']<=.8*a['target_failure_rate']):failures.append('target failure reduction below20percent or unexposed')
    if a['all_failure_rate'] is not None and (b['all_failure_rate'] is None or b['all_failure_rate']>a['all_failure_rate']+1e-12):failures.append('controlled total failure rate increased')
    comparisons=[]
    for left in summary:
        if left['variant']!='baseline' or left['dimension'] not in ('all','profile'):continue
        right=next(r for r in summary if all(r[k]==left[k] for k in ('partition','dimension','group')) and r['variant']=='root-completion')
        reg=[]
        for k in ('won','final_eliminate_progress','final_secure_progress'):
            if right['metrics'].get(k,0)<left['metrics'].get(k,0)-1e-12:reg.append(k+' decreased')
        for k in ('lost',):
            if right['metrics'].get(k,0)>left['metrics'].get(k,0):reg.append(k+' increased')
        if right['movement'].get('unwon_unlost_games',0)>left['movement'].get('unwon_unlost_games',0):reg.append('unresolved games increased')
        if right['unresolved_stop_rate']>left['unresolved_stop_rate']+1e-12:reg.append('unresolved stop rate increased')
        if right['longest_stall']>left['longest_stall']:reg.append('maximum stall increased')
        comparisons.append(dict(partition=left['partition'],dimension=left['dimension'],group=left['group'],regressions=reg))
        if reg:failures.append('/'.join((left['partition'],left['dimension'],left['group']))+': '+', '.join(reg))
    for r in rows:
        if r.get('completion_max_combinations',0)>1728:failures.append('completion search bound exceeded')
        if any(r['movement'].get(k,0) for k in ('persona_band_violations','introduced_blocked_roots')) or r.get('unsupported_root_selections',0):
            failures.append('contract violation')
    return dict(summary=summary,gate_comparisons=comparisons,natural_exposure_sufficient=exposure,
        gate_failures=failures,decision='inconclusive exposure; keep default off' if not exposure else 'reject promotion; keep default off' if failures else 'review pending including serial timing; keep default off')


def paired_outcomes(a,b):
    keys=('won','lost','win_credit','final_eliminate_progress','final_secure_progress',
        'unresolved_stop_selections','expired_stop_selections','longest_observed_stall','ticks')
    movement=('move_attempts','failed_moves','friendly_collision_moves','both_collision_moves','opponent_collision_moves')
    return dict(condition=[a[k] for k in ('partition','scenario','profile','seed','rival')],
        metrics={k:b.get(k,0)-a.get(k,0) for k in keys},
        movement={k:b['movement'].get(k,0)-a['movement'].get(k,0) for k in movement},
        baseline_exposure=a['diagnostics'].get('declined_friendly_conflict_ticks',0),
        adopted_completion_ticks=b['movement'].get('completion_adopted_ticks',0),
        baseline_work=a['total_work']['model_transitions'],candidate_work=b['total_work']['model_transitions'])


def experiment(output,workers=4):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'preregister.json').exists():raise FileExistsError('use a fresh output; frozen evidence is never overwritten')
    frozen=sources();work=jobs();started=time.perf_counter()
    prereg=dict(format='controlled-congestion-preregister-v1',base=BASE,jobs=work,source_hashes=frozen,
        design_sha256=hashlib.sha256(DESIGN.read_bytes()).hexdigest(),
        description='Frozen before any controlled or heldout game; existing candidate unchanged; forced probes never count as exposure',
        configs={v:asdict(TacticalControl(root_completion=v=='root-completion',recovery_options=True,trace_execution=True)) for v in VARIANTS})
    (output/'preregister.json').write_text(json.dumps(prereg,indent=2)+'\n')
    probes=forced_probes();(output/'probes.json').write_text(json.dumps(probes,indent=2)+'\n')
    results={};checks=updates=0
    with (output/'trajectories.jsonl').open('w') as stream,ProcessPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(run_job,j):i for i,j in enumerate(work)}
        for future in as_completed(pending):
            r,check,update=future.result();checks+=check;updates+=update
            stream.write(json.dumps(r)+'\n');stream.flush();results[pending[future]]=r
            print(f'{len(results)}/{len(work)} {r["partition"]} {r["scenario"]} {r["profile"]} {r["seed"]} {r["variant"]} win={r["won"]} collisions={r["movement"].get("failed_moves",0)} complete={r["movement"].get("completion_adopted_ticks",0)} {r["game_seconds"]:.1f}s',flush=True)
    assert frozen==sources() and prereg['design_sha256']==hashlib.sha256(DESIGN.read_bytes()).hexdigest(),'frozen source/design changed'
    raw=[results[i] for i in range(len(work))]
    divergences=[dict(condition=list(work[i][:-1]),**paired_divergence(raw[i],raw[i+1])) for i in range(0,len(work),2)]
    rows=[{k:v for k,v in r.items() if k not in ('trace','enemy_predictions')} for r in raw]
    result=dict(format='controlled-congestion-v1',games=len(rows),source_hashes=frozen,
        elapsed_seconds=time.perf_counter()-started,rule_checks=checks,purpose_checks=updates,
        paired_divergences=divergences,pairs=[paired_outcomes(raw[i],raw[i+1]) for i in range(0,len(work),2)],runs=rows,defaults_changed=False,candidate_changed=False,**summarize(rows))
    (output/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',default='reflex_artifacts/controlled_congestion');p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();experiment(args.output,args.workers)
