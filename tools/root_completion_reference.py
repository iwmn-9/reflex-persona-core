"""Serial trace-off timing and full trace noninterference for frozen old cases."""
from pathlib import Path
import copy
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.root_collision_experiment import run_job,VARIANTS
from reflex.continuation_experiment import sources


def stable(run):
    r=copy.deepcopy(run)
    for k in ('decision_p50_ms','decision_and_feedback_p50_ms','game_seconds','trace_execution','execution_contract'):
        r.pop(k,None)
    r['total_work'].pop('planner_seconds',None)
    for row in r['trace']:row.get('planning',{}).pop('execution',None)
    return r


def main(output,repetitions=3,prior=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);frozen=sources()
    jobs=[('choke','eliminate','switch','care',281,v) for v in VARIANTS]+[('choke','both','reference','care',282,v) for v in VARIANTS]
    old={}
    if prior:
        with Path(prior).open() as stream:
            for line in stream:
                r=json.loads(line)
                if r['variant']=='objective' and r['partition']=='heldout':old[(r['scenario'],r['profile'],r['seed'],r['rival'])]=r
    rows=[]
    with (output/'reference_trajectories.jsonl').open('w') as stream:
        for job in jobs:
            reference=None;times=[];costs=[]
            for rep in range(repetitions):
                r,checks,updates=run_job(job,False)
                if reference is None:reference=r
                assert stable(r)==stable(reference),'serial repetition changed non-timing outcomes'
                times.append(r['game_seconds']);costs.append(r['decision_and_feedback_p50_ms'])
                print(job,rep,r['game_seconds'],flush=True)
            traced,_,_=run_job(job,True)
            assert stable(traced)==stable(reference),'tracing changed decisions/feedback/learning/audits'
            baseline_match=None
            if prior and job[-1]=='baseline':
                a=copy.deepcopy(old[(r['scenario'],r['profile'],r['seed'],r['rival'])]['trace'])
                for row in a:row.get('planning',{}).pop('execution',None)
                # Persisted JSON uses string map keys/lists; normalize the
                # in-memory trace before comparing against old JSON evidence.
                b=json.loads(json.dumps(reference['trace']))
                for row in b:row.get('planning',{}).pop('execution',None)
                assert a==b,'default path differs from base-commit old evidence'
                baseline_match=True
            stream.write(json.dumps(reference)+'\n');stream.flush()
            rows.append(dict(job=job,ticks=reference['ticks'],won=reference['won'],
                game_seconds=times,decision_and_feedback_p50_ms=costs,trace_noninterference=True,
                old_base_trajectory_match=baseline_match,rule_checks=checks,purpose_checks=updates,
                movement=reference['movement']))
    assert frozen==sources(),'source changed during serial reference'
    result=dict(format='root-collision-reference-v1',source_hashes=frozen,repetitions=repetitions,runs=rows,
        limitation='Warm serial repeated timings for two known development conditions; not population runtime or independent game-level confidence interval')
    (output/'reference.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',default='reflex_artifacts/root_collision');p.add_argument('--repetitions',type=int,default=3);p.add_argument('--prior')
    a=p.parse_args();main(a.output,a.repetitions,a.prior)
