"""Existing responsive hypothesis tracker on the exact same public streams."""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import digest
from reflex.opponent_beliefs import HypothesisTracker


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--controls',required=True);p.add_argument('--frozen',action='store_true');a=p.parse_args();root=Path(a.root)
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        return dispatch(root,Path(__file__).name,['--controls',str(Path(a.controls).resolve())])
    supplied=[json.loads(l) for l in (Path(a.controls)/'trajectories.jsonl').read_text(encoding='utf-8').splitlines()]
    streams=sorted((r for r in supplied if r['arm']=='adaptive'),key=lambda r:(r['scenario'],r['seed']))
    (root/'preregister.json').write_text(json.dumps(dict(version='open-model-legacy-v1',input=digest(streams),
        comparison='unmodified HypothesisTracker default responsive inference on identical later public reveals; prediction comparison, not live decision comparison',
        streams=[(r['scenario'],r['seed']) for r in streams]),indent=2)+'\n',encoding='utf-8')
    models={'a':dict(a=1.,b=0.,c=0.),'b':dict(a=0.,b=1.,c=0.)};results=[]
    for r in streams:
        tracker=HypothesisTracker(tuple(models));trace=[]
        for t in r['trace']:
            prediction=tracker.snapshot().predict(models)
            trace.append(dict(step=t['step'],outcome=t['outcome'],prediction=prediction,
                log_loss=-math.log(prediction[t['outcome']]),expected_log_loss=-sum(q*math.log(prediction[k]) for k,q in t['actual'].items())))
            tracker.observe(models,t['outcome'],str(t['step']),forecast=prediction)
        results.append(dict(scenario=r['scenario'],seed=r['seed'],arm='existing-responsive',trace=trace,final=tracker.record()))
    with (root/'trajectories.jsonl').open('x',encoding='utf-8') as f:
        for r in results:f.write(json.dumps(r)+'\n')
    evaluation=dict(streams=len(results),observations=sum(len(r['trace']) for r in results))
    (root/'evaluation.json').write_text(json.dumps(evaluation,indent=2)+'\n');print(json.dumps(evaluation),flush=True)


if __name__=='__main__':main()
