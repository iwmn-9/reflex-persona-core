"""Evaluator-only auction Bellman reference; NEVER supplied to a playing NPC.

Exact finite-horizon purpose opportunity given an explicit response law. This
does not price personality, so its regret cannot be called a personality error.
It localizes losses at scheduled versus intervening choices and model mismatch.
"""
import argparse,json,sys
from functools import lru_cache
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.run_owned_value import law


def reference(turns,variant,probabilities):
    prizes=(2,7,3,8,1,6,4,5) if not variant else (8,1,5,2,7,3,6,4)
    @lru_cache(None)
    def value(t,cash):
        if t==turns:return .25*cash
        return max(options(t,cash).values())
    def options(t,cash):
        result={}
        for bid in range(min(4,cash)+1):
            q=0.
            for z,p in enumerate(probabilities(t)):
                other=(0,2,3)[z]
                gain=prizes[t%8]*(1 if bid>other else -1 if bid<other else 0)
                after=min(12,cash-(bid if bid>other else 0)+1)
                q+=p*(gain+value(t+1,after))
            result['bid:'+str(bid)]=q
        return result
    return lambda t,cash:{k:v/(8*turns) for k,v in options(t,cash).items()}


def diagnose(row):
    turns=row['turns'];truth=reference(turns,row['seed']%2,
        lambda t:tuple(law(t,turns,row['regime']).values()))
    rows=[]
    for t in row['trace']:
        learned=reference(turns,row['seed']%2,lambda _,p=tuple(t['law'].values()):p)
        q=truth(t['tick'],int(t['before']['cash']));p=learned(t['tick'],int(t['before']['cash']))
        best=max(q,key=q.get);believed=max(p,key=p.get)
        rows.append(dict(tick=t['tick'],action=t['action'],planned=t['planned'],
            cash=t['before']['cash'],truth_best=best,believed_best=believed,
            purpose_opportunity_loss=q[best]-q[t['action']],
            believed_opportunity_loss=p[believed]-p[t['action']],
            model_choice_loss=q[best]-q[believed]))
    return dict(**{k:row[k] for k in ('seed','profile','regime','arm','interval','score')},
        summary={label:dict(count=len(ts),mean_loss=float(np.mean([t['purpose_opportunity_loss'] for t in ts])),
            sum_loss=sum(t['purpose_opportunity_loss'] for t in ts),
            mean_believed_loss=float(np.mean([t['believed_opportunity_loss'] for t in ts])),
            mean_model_choice_loss=float(np.mean([t['model_choice_loss'] for t in ts])))
            for label,ts in [('all',rows),('planned',[t for t in rows if t['planned']]),('reflex',[t for t in rows if not t['planned']])]
            if ts},trace=rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--sources',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=[]
    for source in a.sources:
        for line in Path(source).read_text().splitlines():
            r=json.loads(line)
            if r['game']=='auction':result.append(diagnose(r))
    Path(a.output).write_text(json.dumps(dict(semantics=__doc__,runs=result),indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='trace'} for r in result]))


if __name__=='__main__':main()
