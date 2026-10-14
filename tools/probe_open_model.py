"""Outcome-only controls: exact library, noise, novel law, change and recurrence."""
import argparse
import json
import math
import random
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.adaptive_dynamics import EventLearner
from reflex.core import digest


def run():
    legal=('a','b','c');models={'a':dict(a=1.,b=0.,c=0.),'b':dict(a=0.,b=1.,c=0.)};rows=[]
    for scenario in ('exact-library','noise','novel','shift','return','interaction','unobservable'):
        for seed in range(75000,75008):
            records=[];rng=random.Random(seed)
            for step in range(384):
                x,y=str(rng.randrange(2)),str(rng.randrange(2));target='a'
                if scenario=='novel':target='c'
                elif scenario=='shift':target='c' if step>=192 else 'a'
                elif scenario=='return':target='c' if 128<=step<256 else 'a'
                elif scenario=='interaction':target='c' if x!=y else 'a'
                elif scenario=='unobservable':target=rng.choice(('a','c'))
                p={k:.98*float(k==target)+.02/3 for k in legal} if scenario!='noise' else dict.fromkeys(legal,1/3)
                u=rng.random();mass=0.
                for token in legal:
                    mass+=p[token]
                    if u<mass:break
                records.append(((x,y),token,p))
            for arm in ('library','marginal','long','adaptive'):
                m=EventLearner(owner='control',model_id='control',factors=() if arm=='marginal' else ('x','y'),
                    names=tuple(models),learned=arm!='library',windows=(128,) if arm=='long' else (8,32,128))
                s=m.start();trace=[]
                for step,(context,token,actual) in enumerate(records):
                    context=() if arm=='marginal' else context;p=m.predict(s,context,legal,models)
                    trace.append(dict(step=step,outcome=token,prediction=p,actual=actual,
                        log_loss=-math.log(p[token]),expected_log_loss=-sum(q*math.log(p[k]) for k,q in actual.items()),
                        learned_weight=m.diagnostics(s)['learned_weight']))
                    s=m.observe(s,context,legal,models,token)
                rows.append(dict(scenario=scenario,seed=seed,arm=arm,trace=trace,final=m.record(s)))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--frozen',action='store_true');a=p.parse_args();root=Path(a.root)
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        return dispatch(root,Path(__file__).name,[])
    registration=dict(version='open-model-controls-v1',scenarios=['exact-library','noise','novel','shift','return','interaction','unobservable'],
        seeds=list(range(75000,75008)),steps=384,arms=['library','marginal','long','adaptive'],
        source=digest(json.loads((root/'source_snapshot.json').read_text())),
        purpose='predictive cost under a correctly specified library; false certainty in noise or absent observable cues; recovery without an oracle reset')
    (root/'preregister.json').write_text(json.dumps(registration,indent=2)+'\n')
    rows=run()
    with (root/'trajectories.jsonl').open('x',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    result=dict(streams=len(rows),observations=sum(len(r['trace']) for r in rows))
    (root/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
