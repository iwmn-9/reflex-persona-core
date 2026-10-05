"""Trace-off, single-process continuation-reference and full-game cost checks.

Run after the matrix: python tools/continuation_reference.py [artifact folder].
A nested reference uses the full existing horizon/samples/plans ONCE at each
predetermined public snapshot; all its leaves use objective tactics, so there
is no recursion. It is not the full executing DecisionLoop: future experience,
progress-watch recovery, and route changes are not simulated.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataclasses import asdict
from collections import Counter
import copy
import json
import time
import platform
import hashlib
import numpy as np
from reflex.combat import battle_from_record
from reflex.combat_planning import TacticalControl, tactical_joint, forecast
from reflex.persona_continuation import PersonaContinuation
from reflex.deliberation import select
from reflex.continuation_experiment import run_job


def main(output):
    output=Path(output);snapshots=json.loads((output/'snapshots.json').read_text());rows=[]
    traces={}
    for line in (output/'trajectories.jsonl').open():
        r=json.loads(line)
        if r['variant']=='objective':traces[(r['scenario'],r['profile'],r['seed'],r['rival'],r['recovery_options'])]=r
    for snapshot in snapshots:
        w=battle_from_record(snapshot['before']);actors=snapshot['actors'];cs=snapshot['contexts'];ds=snapshot['reflex']
        team=w.units[actors[0]].team;route={i:c['facts']['chosen_route'] for i,c in zip(actors,cs)}
        states={i:c['state'] for i,c in zip(actors,cs)}
        identity=tuple(snapshot[k] for k in ('scenario','profile','seed','rival','recovery_options'))
        actual_row=next(r for r in traces[identity]['trace'] if r['before']['tick']==w.tick)
        actual={i:actual_row['choices'][str(i)] for i in actors}
        stable=copy.deepcopy((w,cs,ds))
        result=dict(condition=list(identity),tick=w.tick,actual={str(i):k for i,k in actual.items()},variants={})
        for variant in ('objective','persona-band','nested-reference'):
            timings=[];metadata={}
            for repeat in range(3):
                t=time.perf_counter()
                if variant=='objective':choices=tactical_joint(w,actors,route)
                elif variant=='persona-band':
                    model=PersonaContinuation(actors,cs,ds);choices,_=model.choose(w,team,route,states)
                    metadata=dict(scorings=model.scorings,model_transitions=0,coordination_conflicts=model.conflicts)
                else:
                    # Full current planner budget, exactly one level. The
                    # selected snapshot inputs contain observed experience;
                    # this privileged fidelity explains some concordance and
                    # is not evidence that nested planning is an oracle.
                    f=forecast(w,actors,cs,ds,TacticalControl())
                    decisions,audit=select(cs,f)
                    choices={i:d['action_id'] for i,d in zip(actors,decisions or ds)}
                    metadata=dict(model_transitions=f.audit['nodes'],plans=len(f.roots),adopted=audit['adopted'])
                timings.append((time.perf_counter()-t)*1000)
            assert (w,cs,ds)==stable
            result['variants'][variant]=dict(ms=timings,median_ms=float(np.median(timings)),choices={str(i):k for i,k in choices.items()},
                same_actual_actors=sum(choices[i]==actual[i] for i in actors),actors=len(actors),same_actual_joint=choices==actual,**metadata)
        rows.append(result)
        print('snapshot',len(rows),'/',len(snapshots),snapshot['profile'],snapshot['scenario'],w.tick,flush=True)
    # Full trace-off games are sequential; do not compare matrix multiprocess
    # times as uncontended CPU latency. Off/on behavior must be identical.
    runtime=[]
    for recovery in (False,True):
        for variant in ('objective','persona-band'):
            job=('choke','eliminate','switch','steady',180,variant,recovery,'known')
            r,_,checks,updates=run_job(job,trace_execution=False,capture=False)
            traced=next(json.loads(line) for line in (output/'trajectories.jsonl').open()
                if (lambda a:a['seed']==180 and a['variant']==variant and a['recovery_options']==recovery)(json.loads(line)))
            assert len(r['trace'])==len(traced['trace'])
            for a,b in zip(r['trace'],traced['trace']):
                # JSON round-trip normalizes integer dict keys.
                for key in ('before','after','choices','audit','purpose'):
                    assert json.loads(json.dumps(a[key]))==b[key],key
                assert json.loads(json.dumps(a.get('planning',{})))=={k:v for k,v in b.get('planning',{}).items() if k!='execution'}
            for key in ('won','lost','ticks','actions','learned_uses','regime_resets','decisions','planning_nodes'):
                assert r[key]==traced[key],key
            runtime.append(dict(variant=variant,recovery_options=recovery,ticks=r['ticks'],won=r['won'],
                seconds=r['game_seconds'],per_tick_ms=1000*r['game_seconds']/r['ticks'],
                decision_feedback_p50_ms=r['decision_and_feedback_p50_ms'],work=r['total_work'],
                trace_invariance=True,rule_checks=checks,purpose_checks=updates))
            print('trace off',variant,recovery,r['game_seconds'],flush=True)
    summary={}
    for v in ('objective','persona-band','nested-reference'):
        group=[r['variants'][v] for r in rows]
        summary[v]=dict(snapshots=len(group),median_snapshot_ms=float(np.median([g['median_ms'] for g in group])),
            actor_matches=sum(g['same_actual_actors'] for g in group),actors=sum(g['actors'] for g in group),
            joint_matches=sum(g['same_actual_joint'] for g in group),
            model_transitions=sum(g.get('model_transitions',0) for g in group))
    from reflex.continuation_experiment import sources
    provenance=sources();provenance['tools/continuation_reference.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result=dict(format='continuation-reference-v1',source_hashes=provenance,platform=platform.platform(),python=platform.python_version(),
        repetitions=3,snapshots=rows,summary=summary,trace_off_games=runtime,
        limits=['Same-public-state conditional probes, not trajectory calibration or heldout policy wins.',
          'Nested reference sees current effective root contexts including real observed experience; cheap reconstructs priors.',
          'Nested reference is full 6-tick/4-sample/24-plan forecast plus persona selection, with objective leaves and no recursive calls.',
          'No progress-watch recovery or route replanning is simulated by the nested reference.',
          'Full-game runtime uses four sequential seed180 games; snapshot timings use three repeats, not a large-N NPC benchmark.',
          'Fixed method order and shared geometry caches make snapshot numbers warmed/cache-sensitive diagnostics, not robust latency estimates.'])
    (output/'reference.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'reflex_artifacts/persona_continuation')
