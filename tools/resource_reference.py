"""Evaluator-only exact resource value, with cash and delayed construction.

The reference knows the test world's future response law. NPCs never receive
these values. Purpose opportunity loss includes personality and unknown future
changes; it is not a measure of intelligence alone.
"""
from functools import lru_cache
import math
import numpy as np


@lru_cache(maxsize=4)
def _tables(turns, best_probabilities):
    # Every legal step changes cash by at least -4 and at most +6. These
    # time-dependent bounds cover all continuations from the declared cash=8
    # start. The terminal purpose clipping is unreachable beyond these bounds.
    tables=[None]*(turns+1)
    end_cash=np.arange(8-4*turns,8+6*turns+1,dtype=float)
    end=np.broadcast_to((end_cash-8)/(6*turns),(5,7,len(end_cash))).copy()
    end[4,1:]=np.nan;end.setflags(write=False);tables[turns]=end
    for tick in range(turns-1,-1,-1):
        cash=np.arange(8-4*tick,8+6*tick+1)
        # The next time slice's minimum cash is four units lower.
        index=np.arange(len(cash))+4
        values=np.full((5,7,len(cash)),np.nan);future=tables[tick+1]
        for plant in range(5):
            for construction in range(1 if plant==4 else 7):
                new_plant=plant+(construction==1)
                new_construction=max(0,construction-1)
                continuation=future[new_plant,new_construction]
                trade=continuation[index+1]
                # More cash never reduces future options or terminal purpose.
                # Hence the crop with highest hit probability also maximizes
                # continuation value, including future affordability effects.
                probability=best_probabilities[tick]
                crop=probability*continuation[index+2+plant]+(1-probability)*continuation[index-1]
                best=np.maximum(trade,crop)
                if plant<4 and not construction:
                    invest=future[plant,6,index-4]
                    best=np.where(cash>=4,np.maximum(best,invest),best)
                values[plant,construction]=best
        values.setflags(write=False);tables[tick]=values
    return tuple(tables)


def reference(turns,probabilities):
    if type(turns) is not int or not 1<=turns<=256:
        raise ValueError('one to 256 finite resource turns required')
    laws=[tuple(probabilities(t)) for t in range(turns)]
    if any(len(p)!=3 or any(not math.isfinite(x) or not 0<=x<=1 for x in p) or abs(sum(p)-1)>1e-12 for p in laws):
        raise ValueError('three normalized public response probabilities required')
    tables=_tables(turns,tuple(max(p) for p in laws))

    def state(w):
        t,c,p,d=w['tick'],w['cash'],w['plant'],w['investment']
        if (type(t) is not int or not 0<=t<=turns or isinstance(c,bool)
            or not isinstance(c,(int,float)) or not math.isfinite(c) or c!=int(c)
            or not 8-4*t<=c<=8+6*t or type(p) is not int or not 0<=p<=4
            or type(d) is not int or not 0<=d<=6 or p==4 and d!=0 or w['hp']!=12):
            raise ValueError('public state from the declared resource rules required')
        return t,int(c),p,d

    def value(w):
        t,c,p,d=state(w)
        return float(tables[t][p,d,c-(8-4*t)])

    def choices(w):
        t,c,p,d=state(w)
        if t==turns:return {}
        index=c-(8-4*(t+1));future=tables[t+1]
        continuation=future[p+(d==1),max(0,d-1)]
        hit,miss=continuation[index+2+p],continuation[index-1]
        q={'trade':float(continuation[index+1])}
        q.update((f'crop:{z}',float(prob*hit+(1-prob)*miss)) for z,prob in enumerate(laws[t]))
        if c>=4 and p<4 and d==0:q['invest']=float(future[p,6,index-4])
        return q

    return value,choices


def account(row):
    if row['game']!='resources':raise ValueError('resource reference is evaluator-only for that rule family')
    from tools.run_owned_value import law
    value,choices=reference(row['turns'],lambda t:tuple(law(t,row['turns'],row['regime']).values()))
    initial=value(row['trace'][0]['before']);loss=noise=0.;steps=[]
    for step in row['trace']:
        q=choices(step['before']);regret=max(q.values())-q[step['action']]
        residual=value(step['after'])-q[step['action']]
        loss+=regret;noise+=residual
        steps.append(dict(tick=step['tick'],purpose_opportunity_loss=regret,residual=residual))
    error=initial-loss+noise-row['score']
    assert abs(error)<1e-10,'resource Bellman account must reconstruct final score'
    return dict(initial_perfect_information=initial,purpose_opportunity_loss=loss,
        residual=noise,residual_removed=initial-loss,reconstruction_error=error,increments=steps)
