"""Finite partitioned native-owner parity run. No scheduler, remote compute, or retries."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,sort_keys=True,indent=2),encoding='utf8');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,choices=range(1,5),default=4)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    plan=dict(cases=[dict(index=i,game=('auction','resources','combat','gambling')[i//4],profile=i%4) for i in range(16)],
        steps=164,horizon_at=129,horizon_steps=1,workers=args.workers,
        criteria='exact action, eligibility and discrete learned state; numerical error <=2e-12; bit-exact native restoration',
        purpose='translation/integration fidelity; existing 40-pair quality acceptance is unchanged')
    write(out/'plan.json',plan);started=time.time();completed=[]
    def run(case):
        path=out/f"case-{case['index']:02d}"
        cmd=[sys.executable,str(ROOT/'tools/verify_rust_owned.py'),'--output',str(path),'--steps','164',
            '--profiles','4','--horizon-at','129','--horizon-steps','1','--case-index',str(case['index'])]
        with (out/f"case-{case['index']:02d}.log").open('w',encoding='utf8') as log:
            result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        return dict(case=case,returncode=result.returncode,summary=json.loads((path/'summary.json').read_bytes()) if (path/'summary.json').exists() else None)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run,case) for case in plan['cases']]):
            result=future.result();completed.append(result);write(out/'progress.json',dict(completed=completed,seconds=time.time()-started))
            print(json.dumps(dict(case=result['case'],returncode=result['returncode'],completed=len(completed))),flush=True)
    good=len(completed)==16 and all(r['returncode']==0 and r['summary']['complete'] for r in completed)
    result=dict(complete=good,cases=sorted(completed,key=lambda r:r['case']['index']),seconds=time.time()-started)
    if good:
        fields=('decisions','saves','horizon_decisions')
        result.update({k:sum(r['summary'][k] for r in completed) for k in fields})
        result.update({k:max(r['summary'][k] for r in completed) for k in ('max_prediction_error','max_state_error','max_score_error')})
        manifests=[json.loads((out/f'case-{i:02d}'/'inputs.json').read_bytes())['sources'] for i in range(16)]
        assert all(m==manifests[0] for m in manifests),'workers used different source/binary versions'
    write(out/'summary.json',result)
    if not good:raise SystemExit(1)


if __name__=='__main__':main()
