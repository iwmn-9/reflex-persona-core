"""Escalate a bounded model horizon instead of mistaking cutoff proxies for goals.

All current viable roots remain covered at every level. Game callbacks still
own termination and goal meaning. Model settlement is not actual certainty.
"""
from dataclasses import replace
import copy
from .continuation_search import search_forecast


def completion_forecast(c,initial,*,horizon,max_horizon,**kwargs):
    if type(max_horizon) is not int or not horizon<=max_horizon<=16:
        raise ValueError('explicit bounded completion horizon required')
    levels=[];current=horizon
    while True:
        f=search_forecast(c,initial,horizon=current,**kwargs)
        endpoints=[g for rows in f.audit['endpoint'].values() for g in rows]
        running=sum(g['status']=='running' for g in endpoints)
        levels.append(dict(horizon=current,endpoints=len(endpoints),running=running))
        if not running or current==max_horizon:break
        current=min(max_horizon,current*2)
    metadata=copy.deepcopy(f.audit)
    metadata['completion_search']=dict(levels=levels,max_horizon=max_horizon,all_model_endpoints_settled=not running,
        semantics='complete all candidate endpoints or exhaust explicit horizon cap; sampled models, not real future certainty')
    return replace(f,audit=metadata)
