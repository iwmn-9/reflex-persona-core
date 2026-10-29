"""Post-hoc H32 mechanism probes; never a new development/adoption study."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.owned_checkpoint import encoded,sha
from tools.probe_adverse_forecasts import probe


def run(plan_path,variant,output):
    plan_path=Path(plan_path).resolve();plan=json.loads(plan_path.read_bytes())
    for name,h in plan['sources'][variant].items():assert sha((ROOT/name).read_bytes())==h,name
    assert sha(Path(__file__).read_bytes())==plan['driver_sha256']
    raw=(plan_path.parent/'inputs.json.gz').read_bytes();assert sha(raw)==plan['inputs_sha256']
    inputs=json.loads(gzip.decompress(raw));jobs=[j for j in plan['jobs'] if j['variant']==variant]
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        pending={pool.submit(probe,inputs[j['input_key']],j['tick'],'stable',tuple(plan['seeds'])):j for j in jobs}
        for future in as_completed(pending):
            job=pending[future];r=future.result()
            assert r['factual_owner']==job['owner']
            r.update(job=job,variant=variant,adoption=False,plan_sha256=sha(plan_path.read_bytes()))
            rows.append(r);(output/(job['key']+'.json')).write_bytes(encoded(r))
            print(json.dumps({k:r[k] for k in ('variant','tick','action','selected_purpose','seconds')}),flush=True)
    assert len(rows)==len(jobs)
    (output/'summary.json').write_bytes(encoded(dict(complete=True,evaluations=len(rows),rows=rows,
        adoption=False,plan_sha256=sha(plan_path.read_bytes()),
        scope='Known adverse roots, same H32 bank; no real continuation, independent check or promotion.')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--variant',required=True,choices=('legacy','coherent'))
    p.add_argument('--output',required=True);a=p.parse_args();run(a.plan,a.variant,a.output)
