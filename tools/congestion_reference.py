"""Frozen serial trace-off replay; does not add efficacy games or tune policy."""
from pathlib import Path
import copy
import json
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.congestion_experiment import run_job,VARIANTS
from reflex.continuation_experiment import sources


def stable(run):
    r=copy.deepcopy(run)
    for k in ('decision_p50_ms','decision_and_feedback_p50_ms','game_seconds','trace_execution','execution_contract'):
        r.pop(k,None)
    r['total_work'].pop('planner_seconds',None)
    r['diagnostics'].pop('saved_first_step_hypothesis_checks',None)
    for row in r['trace']:row.get('planning',{}).pop('execution',None)
    return json.loads(json.dumps(r))


def main(output):
    output=Path(output);frozen=sources()
    prereg=json.loads((output/'preregister.json').read_text())
    assert frozen==prereg['source_hashes'],'primary frozen code required'
    primary={}
    with (output/'trajectories.jsonl').open() as stream:
        for line in stream:
            r=json.loads(line)
            if (r['partition'],r['scenario'],r['profile'],r['seed'])==('controlled','mixed_junction/both','care',592):primary[r['variant']]=r
    assert set(primary)==set(VARIANTS),'primary reference pairs must be complete first'
    rows=[]
    with (output/'reference_trajectories.jsonl').open('w') as stream:
        for variant in VARIANTS:
            times=[];costs=[];reference=None
            for rep in range(3):
                r,checks,updates=run_job(('controlled','mixed_junction','both','reference','care',592,variant),False)
                assert stable(r)==stable(primary[variant]),'trace changed choices, feedback, learning or diagnostics'
                if reference is None:reference=r
                assert stable(r)==stable(reference),'serial repetition changed results'
                times.append(r['game_seconds']);costs.append(r['decision_and_feedback_p50_ms'])
                print(variant,rep,r['game_seconds'],flush=True)
            stream.write(json.dumps(reference)+'\n')
            rows.append(dict(variant=variant,game_seconds=times,median_game_seconds=statistics.median(times),
                decision_and_feedback_p50_ms=costs,median_tick_p50_ms=statistics.median(costs),
                trace_noninterference=True,rule_checks=checks,purpose_checks=updates,
                model_transitions=reference['total_work']['model_transitions'],movement=reference['movement'],
                completion_max_combinations=reference['completion_max_combinations']))
    assert frozen==sources()
    ratio=rows[1]['median_game_seconds']/rows[0]['median_game_seconds']
    result=dict(format='controlled-congestion-reference-v1',source_hashes=frozen,
        fixed_case='controlled mixed_junction/both care592',runs=rows,
        median_game_ratio=ratio,preregistered_runtime_gate_passed=ratio<=1.25,
        limitation='One warm serial game case, not population reflex speed. Path-dependent model work is not completion function overhead.')
    (output/'reference.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',default='reflex_artifacts/controlled_congestion')
    main(p.parse_args().output)
