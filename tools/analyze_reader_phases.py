"""Account for every completed action, separating early and late effects.

This is a descriptive audit of already-completed development runs. Phases are
correlated and never treated as independent replications or adoption evidence.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.compare_consistent_reading import load
from tools.owned_checkpoint import encoded,sha


def analyze(reference,current,output):
    reference,current,output=map(Path,(reference,current,output))
    old,new=load(reference),load(current)
    assert old.keys()==new.keys()
    rows=[]
    for key in sorted(old):
        spec=old[key]['spec'];assert spec==new[key]['spec'] and spec['turns']==128
        for predictor in ('stable','transfer'):
            a,b=old[key]['arms'][predictor],new[key]['arms'][predictor]
            phases=[]
            for begin in (0,32,64,96):
                left,right=a['trace'][begin:begin+32],b['trace'][begin:begin+32]
                assert len(left)==len(right)==32
                assert [(r['truth'],r['response'],r['response_prediction']) for r in left]==[(r['truth'],r['response'],r['response_prediction']) for r in right]
                phases.append(dict(begin=begin,end=begin+32,
                    conditional_delta=sum(r['conditional_increment'] for r in right)-sum(r['conditional_increment'] for r in left),
                    changed_choices=sum(x['action']!=y['action'] for x,y in zip(left,right))))
            assert abs(sum(r['conditional_delta'] for r in phases)-(b['conditional']-a['conditional']))<1e-12
            rows.append(dict(spec=spec,predictor=predictor,phases=phases))
    groups=defaultdict(list)
    for r in rows:groups[(r['predictor'],r['spec']['game'])].append(r)
    summary=[]
    for (predictor,game),part in sorted(groups.items()):
        summary.append(dict(predictor=predictor,game=game,conditions=len(part),
            phases=[dict(begin=begin,end=begin+32,
                conditional_delta=sum(r['phases'][i]['conditional_delta'] for r in part)/len(part),
                changed_choices=sum(r['phases'][i]['changed_choices'] for r in part))
                for i,begin in enumerate((0,32,64,96))]))
    result=dict(complete=True,comparisons=len(rows),summary=summary,rows=rows,adoption=False,
        interpretation='Paired conditional gain on each actual trajectory, including differences in previously acquired resources. Not causal learning-rate estimates, independent phases, or full H32 results.',
        inputs={str(p):sha((p/'verification.json').read_bytes()) for p in (reference,current)},
        analyzer_sha256=sha(Path(__file__).read_bytes()))
    if output.exists():raise FileExistsError('retain prior analysis')
    output.write_bytes(encoded(result))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('reference','current','output'):p.add_argument('--'+field,required=True)
    a=p.parse_args();analyze(a.reference,a.current,a.output)
