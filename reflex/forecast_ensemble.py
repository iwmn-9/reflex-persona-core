"""Retain multiple conditional futures instead of asserting one is the truth.

Purpose uses the worst model when requested; subjective effects retain equal
model mixtures, including unfavorable outcomes. Equal prior is a declared
ignorance convention, not learned predictive reliability.
"""
import copy
import numpy as np
from .deliberation import JointForecast
from .examples import action


def combine(forecasts,*,purpose_rule='worst'):
    if not 2<=len(forecasts)<=8 or purpose_rule not in ('worst','mean'):
        raise ValueError('two to eight models and declared purpose aggregation required')
    first=forecasts[0]
    if not all(isinstance(f,JointForecast) for f in forecasts):raise ValueError('declared joint forecasts required')
    for f in forecasts:
        if f.continuity is not None:raise ValueError('arrival continuity must be resolved explicitly before model aggregation')
        if (f.roots!=first.roots or f.target!=first.target or f.horizon!=first.horizon or
                f.max_regret!=first.max_regret or len(f.contexts)!=len(first.contexts)):
            raise ValueError('identical owners, real roots, purpose target and horizon required')
        if set(f.purpose)!=set(first.roots) or not all(np.isfinite(v) and abs(v)<=1 for v in f.purpose.values()):
            raise ValueError('finite complete bounded purpose required')
        for c,other in zip(f.contexts,first.contexts):
            if {k:v for k,v in c.items() if k!='actions'}!={k:v for k,v in other.items() if k!='actions'}:
                raise ValueError('identical owner observations and fixed axes required')
            if len(c['actions'])!=len(first.roots) or {a['id'] for a in c['actions']}!=set(first.roots):
                raise ValueError('one effect set per candidate required')
    contexts=copy.deepcopy(first.contexts)
    for j,c in enumerate(contexts):
        c['actions']=[]
        for key in first.roots:
            outcomes=[];confidences=[]
            for f in forecasts:
                a=next(a for a in f.contexts[j]['actions'] if a['id']==key)
                if not a['legal'] or a['known_failure']:raise ValueError('model root feasibility must agree')
                es=copy.deepcopy(a['outcomes'])
                if abs(sum(e['p'] for e in es)-1)>1e-8:raise ValueError('each model must contain complete outcome mass')
                for e in es:e['p']/=len(forecasts)
                outcomes.extend(es);confidences.append(a['confidence'])
            c['actions'].append(action(key,*outcomes,confidence=min(confidences)))
    rows={k:tuple(f.purpose[k] for f in forecasts) for k in first.roots}
    purpose={k:float(min(v) if purpose_rule=='worst' else np.clip(np.mean(v),-1,1)) for k,v in rows.items()}
    return JointForecast(tuple(contexts),copy.deepcopy(first.roots),purpose,first.horizon,first.max_regret,
        dict(nodes=sum((f.audit or {}).get('nodes',0) for f in forecasts),models=len(forecasts),
            purpose_rule=purpose_rule,model_purpose=rows,
            model_disagreement={k:max(v)-min(v) for k,v in rows.items()},
            model='declared equal mixture for personal effects; worst conditional purpose without claiming a true model'),
        target=first.target)
