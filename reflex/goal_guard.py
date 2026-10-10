"""Optional common purpose boundary before fixed-personality selection.

The adapter supplies normalized [0,1] goal returns for shared trial scenarios.
Sampling uncertainty is distinguished from model correctness. The boundary does
not infer goals from rules, change personality, or equalize win rates.
"""
import copy
import numpy as np
from .core import Policy, compile_batch


def choose_with_goal(context,names,returns,*,max_regret=.12,error_multiplier=2.):
    names=tuple(names);returns=np.asarray(returns,dtype=float)
    if len(set(names))!=len(names) or set(names)!={a['id'] for a in context['actions']}:
        raise ValueError('goal rows must match every distinct context action')
    if returns.ndim!=2 or returns.shape[0]!=len(names) or returns.shape[1]<2:
        raise ValueError('two or more goal samples per candidate required')
    if not np.isfinite(returns).all() or np.any((returns<0)|(returns>1)):
        raise ValueError('finite normalized goal returns required')
    if not np.isfinite(max_regret) or not 0<=max_regret<=1 or not np.isfinite(error_multiplier) or error_multiplier<0:
        raise ValueError('bounded regret and finite nonnegative sampling multiplier required')
    viable={a['id'] for a in context['actions'] if a['legal'] and not a['known_failure']}
    if not viable:raise ValueError('no viable action; adapter must supply recovery')
    means=returns.mean(1);best=max((i for i,n in enumerate(names) if n in viable),key=lambda i:means[i])
    allowed=[];rejected=[];bounds={}
    for i,name in enumerate(names):
        diff=returns[best]-returns[i];error=float(diff.std(ddof=1)/np.sqrt(len(diff)))
        lower=float(diff.mean())-error_multiplier*error
        bounds[name]=dict(estimated_regret=float(diff.mean()),sampling_error=error,lower_bound=lower)
        if name in viable and lower<=max_regret:allowed.append(name)
        else:rejected.append(name)
    policy=Policy(principle_priority='finite');original_batch=compile_batch([context])
    original=policy.decide(original_batch,False).records(original_batch)[0]
    filtered=copy.deepcopy(context);filtered['actions']=[a for a in filtered['actions'] if a['id'] in allowed]
    batch=compile_batch([filtered]);result=policy.decide(batch,False).records(batch)[0]
    return result,dict(allowed=allowed,rejected=rejected,max_model_regret=max_regret,
        personality_action=original['action_id'],guard_changed=original['action_id']!=result['action_id'],
        bounds=bounds,uncertainty='paired Monte Carlo sampling only; wrong continuation models are not covered')
