"""Re-evaluate one saved factual root with predeclared nested model banks.

Run outside the frozen worker checkout. Reuse its unchanged probe and controller;
only the diagnostic model seed bank varies. No real-world continuation occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def inputs(path,tick,start,samples):
    raw=Path(path).read_bytes();result=json.loads(raw)
    if start not in (701,9101,9201) or samples not in (3,8):
        raise ValueError('predeclared nested diagnostic bank required')
    spec=result['configuration']['spec']
    if (not result['complete'] or len(result['trace'])!=spec['turns']
            or result['replayed_actual_steps']!=spec['turns']
            or result['configuration']['arm']!='sample8'):
        raise ValueError('complete original sample8 execution required')
    if type(tick) is not int or not 0<=tick<len(result['trace']):
        raise ValueError('resolved original root required')
    return dict(spec=spec,result=result),tuple(range(start,start+samples)),hashlib.sha256(raw).hexdigest()


def main(args):
    worker=Path(args.worker).resolve();output=Path(args.output).resolve()
    if output.exists():raise FileExistsError('do not replace a diagnostic result')
    data,seeds,checksum=inputs(args.input,args.tick,args.start,args.samples)
    sys.path.insert(0,str(worker))
    from tools.run_adverse_recovery import verify_registration
    from tools.probe_adverse_forecasts import probe
    registration=verify_registration(worker/args.plan)
    if data['result']['configuration']['registration']!=registration:
        raise ValueError('input does not belong to frozen computation')
    started=time.monotonic()
    row=probe(data,args.tick,'stable',seeds)
    saved=data['result']['trace'][args.tick]
    assert row['factual_owner']==saved['before_owner']
    if args.start==701 and args.samples==8:
        # The original eight-sample forecast is an exact diagnostic control.
        assert row['action']==saved['action']
        assert row['selected_purpose']==saved['selected_purpose']
    row.update(registration=registration,input_sha256=checksum,adoption=False,
        exact_registered_forecast_reproduced=args.start==701 and args.samples==8,
        diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        total_seconds=time.monotonic()-started)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf8') as stream:json.dump(row,stream,sort_keys=True)
    print(json.dumps({k:row[k] for k in ('id','tick','seeds','action','selected_purpose','total_seconds')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',required=True);p.add_argument('--input',required=True)
    p.add_argument('--plan',default='evidence/long_horizon/adverse_sample_budget/plan.json')
    p.add_argument('--tick',type=int,required=True);p.add_argument('--start',type=int,required=True)
    p.add_argument('--samples',type=int,required=True);p.add_argument('--output',required=True)
    main(p.parse_args())
