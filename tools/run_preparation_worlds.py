"""Finite Colab development cohort; no partial execution or automatic adoption."""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.adverse_segment import audit
from tools.completion_metrics import summarize_trace
from tools.owned_checkpoint import encoded, sha
from tools.run_adverse_recovery import run, verify_registration
from tools.run_owned_value import law
from tools.review_adverse_recovery import summarize


def save(path, value):
    temp=path.with_suffix('.new');temp.write_bytes(encoded(value));temp.replace(path)


def execute(plan_path, output, spec, arm):
    registration=verify_registration(plan_path)
    plan=json.loads(Path(plan_path).read_bytes())
    assert sha((ROOT/'docs/COMPLETION.md').read_bytes())==plan['completion_contract_sha256']
    folder=Path(output)/f"adverse-{spec['id']}-{arm}"
    folder.mkdir(parents=True,exist_ok=True)
    result=run(folder,spec,arm,registration=registration)
    proof=audit(folder,spec,arm,registration)
    assert proof['complete'] and result['complete']
    metrics=summarize_trace(result['trace'],scheduled_turns=spec['turns'],terminal=True,
        truth_schedule=[law(t,spec['turns'],spec['regime']) for t in range(spec['turns'])])
    save(folder/'metrics.json',metrics)
    save(folder/'status.json',dict(**proof,state='complete',case=spec['id'],arm=arm,
        result_sha256=sha((folder/'result.json').read_bytes()),
        metrics_sha256=sha((folder/'metrics.json').read_bytes())))
    verify_registration(plan_path)
    assert sha((ROOT/'docs/COMPLETION.md').read_bytes())==plan['completion_contract_sha256']
    return dict(case=spec['id'],arm=arm,steps=proof['actual_steps'])


def main(plan_path, output, workers):
    plan_path=Path(plan_path).resolve();output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    handle=(output/'coordinator.lock').open('a+b')
    if os.name=='nt':
        import msvcrt
        handle.write(b'0');handle.flush();handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_registration(plan_path)
    plan=json.loads(plan_path.read_bytes());assert plan['arms']==['stable','expected']
    assert sha((ROOT/'docs/COMPLETION.md').read_bytes())==plan['completion_contract_sha256']
    finished=[];failures=[]
    def status(complete=False):
        save(output/'status.json',dict(complete=complete,pid=os.getpid(),updated=time.time(),
            executions_complete=len(finished),expected=len(plan['specs'])*2,
            actual_steps=sum(r['steps'] for r in finished),failures=failures,adoption=False))
    status()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(execute,str(plan_path),str(output),s,arm):(s['id'],arm)
              for s in plan['specs'] for arm in plan['arms']}
        for future in as_completed(jobs):
            try:finished.append(future.result())
            except Exception as error:failures.append(dict(job=jobs[future],error=repr(error)))
            status()
    if failures:raise RuntimeError('registered execution failed; no adoption or omission')
    result=summarize(output,plan)
    assert result['complete'] and result['pairs_complete']==10
    result['completion_contract_sha256']=plan['completion_contract_sha256']
    result['objective_status']=dict(C1='execution integrity audited',C2='development gate only',
        C3='prediction metrics measured; learned/frozen benefit comparison unmeasured',
        C4='unmeasured',C5='unmeasured',C6='unmeasured')
    save(output/'summary.json',result)
    status(True)
    handle.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if not 1<=a.workers<=8:p.error('1..8 workers required')
    main(a.plan,a.output,a.workers)
