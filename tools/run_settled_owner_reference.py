"""Known finite root audit; the reference changes only the first action."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reflex.laboratory import PROFILES
from tests.test_continuation_persona import IntrinsicTradeoff
from tools.preparation_world import PreparationChain, DelayedYield
from tools.settled_owner_reference import evaluate
from tools.owned_checkpoint import encoded, sha
from tools.review_settled_owner_reference import aggregate


def game(spec):
    p=PROFILES[spec['profile']]
    if spec['kind']=='chain':return PreparationChain(3,payout=8.,turns=8,profile=p)
    if spec['kind']=='delay':return DelayedYield(3,payout=8.,turns=8,profile=p)
    return IntrinsicTradeoff(p,turns=4,mode=spec['kind'])


def execute(spec,arm,scope='root'):
    g=game(spec);w=g.start();a=g.actor(g.context(w),learning='segmented')
    result=evaluate(g,a,w,arm,maximum_steps=8,continuation_scope=scope if arm=='expected' else 'root')
    return dict(spec=spec,arm=arm,result=result)


def main(output,workers,scope='root'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'plan.json').exists():raise FileExistsError('fresh reference run required')
    specs=[dict(id=f'{kind}-{p}',kind=kind,profile=p) for kind in ('chain','delay','principle','need') for p in range(4)]
    sources={p.relative_to(ROOT).as_posix():sha(p.read_bytes()) for d in ('reflex','tools','tests') for p in (ROOT/d).rglob('*.py')}
    plan=dict(specs=specs,arms=['stable','expected'],sources=sources,expected_continuation_scope=scope,
        completion_contract_sha256=sha((ROOT/'docs/COMPLETION.md').read_bytes()),
        criteria='All32 roots settled, actual future learner/restoration preserved. Report every incompatible root separately; never drop it from the denominator. Numeric regret mean<=.05,p90<=.15,max<=.5 for each kind/persona. Nearest-rank p90. Known fixtures only; no C2 completion or controller adoption.',
        adoption=False,reference='all deterministic first-action deviations, same future controller; not optimal-policy search')
    (out/'plan.json').write_bytes(encoded(plan))
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(execute,s,a,scope) for s in specs for a in plan['arms']]
        for future in as_completed(futures):
            row=future.result();rows.append(row)
            (out/(row['spec']['id']+'-'+row['arm']+'.json')).write_bytes(encoded(row))
            (out/'progress.json').write_bytes(encoded(dict(complete=False,roots=len(rows),expected=32)))
    assert len(rows)==32 and all(r['result']['complete'] for r in rows)
    result=aggregate(rows,plan)
    assert all(sha((ROOT/n).read_bytes())==h for n,h in sources.items())
    result.update(source_unchanged=True,plan_sha256=sha((out/'plan.json').read_bytes()))
    (out/'summary.json').write_bytes(encoded(result))
    (out/'progress.json').write_bytes(encoded(dict(complete=True,roots=32,expected=32)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--scope',choices=('root','shared'),default='root');a=p.parse_args()
    if a.workers!=1:p.error('one worker while the long cohort owns the other work')
    main(a.output,a.workers,a.scope)
