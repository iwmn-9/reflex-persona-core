"""Ablate added recovery proposals and temporal purpose evaluation.

160/161 development; 170/171 interrupted for an empty-actor trace bug.
180/181 reserved before the final fixed-candidate confirmation.
Old purposeful, extra-options and extra-options+temporal variants use the same
fixed personality, observations, root experience and independent rule RNG.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from pathlib import Path
import copy
import json
import time
from .laboratory import profiles
from .combat_planning import TacticalControl
from .resource_planning import EconomyControl
from .horizon import compare_paths
from .validation_experiment import combat,resources,auction,gambling,replay,replay_progress
from .wait_experiment import diagnose
from .purpose_experiment import source_hashes
from .cross_games import core_hashes

VARIANTS=('purposeful','options','timely')
WEIGHT=.3


def path_audit(r):
    observations=[];paths=[];attempts=Counter()
    if r['genre']=='combat':
        from .combat import battle_from_record,terminal
        from .combat_planning import mission
        team=r['seed']%2
        for row in r['trace']:
            observations.append(mission(battle_from_record(row['before']),team))
            p=row.get('planning',{});path=p.get('selected_path')
            if path:
                waiting=any(e['request']['activities'][e['root']]['kind'] in ('wait','idle','maintain') for e in row.get('purpose',()))
                paths.append((len(observations)-1,path,waiting))
            a=p.get('recovery_attempt')
            if a:
                attempts['expanded_ticks']+=1;attempts['compatible_options_found']+=int(a['adopted'])
                attempts['compatible' if a['adopted'] else a['reason'] or 'no-compatible-recovery']+=1
        final=battle_from_record(r['trace'][-1]['after']);observations.append(mission(final,team));ended=terminal(final)
    elif r['genre']=='resources':
        from .resource_world import world_from_record,terminal
        # Route-specific forecasts require matching route-specific observed
        # values throughout their horizon, even if the real player switches.
        from .resource_planning import task
        seat=r['seed']%4;own=[row for row in r['trace'] if world_from_record(row['before']).turn==seat]
        final=world_from_record(r['trace'][-1]['after']);total=Counter()
        for index,row in enumerate(own):
            path=row.get('planning',{}).get('selected_path')
            if not path:continue
            route=row['selected_route'];obs=[task(world_from_record(x['before']),seat,route) for x in own]+[task(final,seat,route)]
            e=row['purpose'][0];waiting=e['request']['activities'][e['root']]['kind'] in ('wait','idle','maintain')
            total.update(compare_paths(obs,[(index,path,waiting)],terminal(final)))
        return dict(total)
    else:return {}
    return dict(compare_paths(observations,paths,ended),**dict(attempts))


def run_job(job):
    genre,scene,persona,seed,variant,rival=job;p=next(x for x in profiles() if x['id']==persona);fixed=copy.deepcopy(p)
    if genre=='combat':
        terrain,goal=scene.split('/')
        r=combat(p,seed,terrain,goal,'baseline',learn=True,survival_security=True,rival=rival,
            planner=TacticalControl(recovery_options=variant!='purposeful',progress_weight=WEIGHT if variant=='timely' else 0.),progress_watch=True)
    elif genre=='resources':
        r=resources(p,seed,scene,'baseline',planner=EconomyControl(progress_weight=WEIGHT if variant=='timely' else 0.),progress=True)
    elif genre in ('auction','hagetaka'):r=auction(p,seed,scene,'baseline',game=genre)
    else:r=gambling(p,seed,scene,'baseline')
    assert p==fixed;r.update(variant=variant,rival=rival);r.update(diagnose(r,p));r['forecast_audit']=path_audit(r)
    return r,replay(r),replay_progress(r)


def jobs(seeds,development=False):
    from .combat import GOALS
    from .resource_experiment import SCENARIOS
    work=[]
    scenes=('open/eliminate','choke/both') if development else tuple(m+'/'+g for m in ('open','choke') for g in GOALS)
    rivals=('switch',) if development else ('reference','raider','switch')
    for p in profiles():
        for seed in seeds:
            for variant in VARIANTS:
                work.extend(('combat',scene,p['id'],seed,variant,rival) for scene in scenes for rival in rivals)
                work.extend(('resources',scene,p['id'],seed,variant,'game-owned') for scene in (('mixed',) if development else SCENARIOS))
    if not development:
        from .contests_experiment import SCENARIOS as CONTEST
        from .gambling import SCENARIOS as GAMBLING
        for genre,scenes in (('auction',CONTEST),('hagetaka',CONTEST),('gambling',GAMBLING)):
            work.extend((genre,scene,p['id'],s,'control','game-owned') for scene in scenes for p in profiles() for s in seeds)
    return work


def experiment(output,seeds=(180,181),workers=4,development=False,progress=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);frozen=source_hashes();core=core_hashes();work=jobs(seeds,development);runs={};transitions=0;updates=0;started=time.perf_counter()
    with (output/'trajectories.jsonl').open('w',encoding='utf-8') as stream,ProcessPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(run_job,j):i for i,j in enumerate(work)}
        for future in as_completed(pending):
            try:r,t,u=future.result()
            except Exception:
                for f in pending:f.cancel()
                if progress:progress('Harness failure; pending jobs cancelled. Completed trajectories preserved.',flush=True)
                raise
            transitions+=t;updates+=u;stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush();runs[pending[future]]={k:v for k,v in r.items() if k!='trace'}
            if progress and len(runs)%12==0:progress(f'{len(runs)}/{len(work)} games; {transitions} rule transitions replayed',flush=True)
    assert frozen==source_hashes() and core==core_hashes()
    rows=[runs[i] for i in range(len(work))];summary=[];paired=[]
    for genre in ('combat','resources','auction','hagetaka','gambling'):
        variants=VARIANTS if genre in ('combat','resources') else ('control',)
        for variant in variants:
            for p in profiles():
                rs=[r for r in rows if (r['genre'],r['variant'],r['profile'])==(genre,variant,p['id'])]
                if not rs:continue
                audit=Counter()
                for r in rs:audit.update(r['forecast_audit'])
                summary.append(dict(genre=genre,variant=variant,profile=p['id'],games=len(rs),wins=sum(r['won'] for r in rs),
                    expired_stops=sum(r.get('expired_stop_selections',0) for r in rs),unresolved_stops=sum(r.get('unresolved_stop_selections',0) for r in rs),
                    missing_gain=sum(r.get('future_gain_unproven_stops',0) for r in rs),longest_unresolved=max(r.get('longest_unresolved_stop_run',0) for r in rs),
                    score=sum(r['score'] for r in rs),movement_failures=sum(r.get('focal_failed_moves',0) for r in rs),forecast_audit=dict(audit)))
        if genre in ('combat','resources'):
            key=lambda r:(r['scenario'],r['profile'],r['seed'],r['rival'])
            old={key(r):r for r in rows if (r['genre'],r['variant'])==(genre,'purposeful')}
            for v in ('options','timely'):
                new=[r for r in rows if (r['genre'],r['variant'])==(genre,v)]
                paired.append(dict(genre=genre,variant=v,pairs=len(new),better=sum(r['win_credit']>old[key(r)]['win_credit'] for r in new),worse=sum(r['win_credit']<old[key(r)]['win_credit'] for r in new),same=sum(r['win_credit']==old[key(r)]['win_credit'] for r in new)))
    result=dict(format='recovery-options-v1',games=len(rows),seeds=list(seeds),development=development,source_hashes=frozen,core_hashes=core,
        config=dict(combat=asdict(TacticalControl(recovery_options=True,progress_weight=WEIGHT)),resources=asdict(EconomyControl(progress_weight=WEIGHT))),
        rule_transitions_replayed=transitions,purpose_updates_replayed=updates,elapsed_seconds=time.perf_counter()-started,summary=summary,comparisons=paired,runs=rows,
        defaults_changed=False,performance_gate=False,limitations=['Finite authored games; personality compatibility, not equal win rates.','Path errors include changed real continuation; diagnostic only, not causal learning or calibrated confidence.','Added options may still lack joint tier agreement; no tier relaxation or unsafe forced movement.','Temporal weight trades some endpoint value for earlier purpose; not a universal optimum.','Auction/Hagetaka/gambling retain old controls; new planner not connected there.'])
    (output/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# 打開候補と先送り評価','',f'{len(rows)} games; seeds {list(seeds)}; {transitions} rule transitions, {updates} purpose updates replayed.','',
        'purposeful: old path; options: extra supported proposals; timely: options + 0.3 temporal purpose weighting. Fixed personality and real rules; no forced tier relaxation.','',
        '|game|variant|persona|wins/games|expired stops|unresolved|longest|','|---|---|---|---|---|---|---|']
    for s in summary:lines.append(f"|{s['genre']}|{s['variant']}|{s['profile']}|{s['wins']}/{s['games']}|{s['expired_stops']}|{s['unresolved_stops']}|{s['longest_unresolved']}|")
    lines+=['','Diagnostics, not automatic causal learning. Temporal scoring explicitly trades endpoint value for earlier progress. Defaults unchanged.']+result['limitations']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
