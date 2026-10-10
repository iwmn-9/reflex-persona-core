"""Keep avoidable purpose regret small across declared credible model families.

Response prediction gains do not prove action gains. The incumbent personality
context remains the preference reference; challengers constrain purpose risk.
Models are hypotheses, not opponent identities or independent observations.
"""
import copy
import numpy as np
from .core import Policy,compile_batch
from .goal_progress import progress_context


def select(context,names,forecasts,*,weights=None,max_regret=.12,error_multiplier=2.):
    """forecasts: model -> (success samples, progress samples), incumbent first.

    Complete finite weights have zero sampling error. Otherwise paired standard
    errors are sampled-model diagnostics, not coverage of opponent model error.
    A conflict can make some regret unavoidable: minimize the worst supported
    regret, then retain the existing personality within the same tolerance.
    """
    names=tuple(names)
    if len(set(names))!=len(names) or set(names)!={a['id'] for a in context['actions']}:
        raise ValueError('model rows must match every distinct context action')
    if not isinstance(forecasts,dict) or not 1<=len(forecasts)<=16 or any(not isinstance(k,str) or not 0<len(k)<=128 for k in forecasts):
        raise ValueError('1..16 declared bounded model families required')
    if weights is not None and set(weights)!=set(forecasts):raise ValueError('complete masses for every model required')
    if not np.isfinite(max_regret) or not 0<=max_regret<=1 or not np.isfinite(error_multiplier) or error_multiplier<0:
        raise ValueError('bounded purpose tolerance and nonnegative uncertainty multiplier required')
    viable={a['id'] for a in context['actions'] if a['legal'] and not a['known_failure']}
    if not viable:raise ValueError('adapter must supply viable recovery')
    evidence={};all_zero=True
    for key,(success,progress) in forecasts.items():
        success=np.asarray(success,dtype=float);progress=np.asarray(progress,dtype=float)
        if success.ndim!=2 or success.shape[0]!=len(names) or success.shape[1]<(1 if weights is not None else 2) or progress.shape!=success.shape:
            raise ValueError('matched nonempty purpose/progress rows required')
        if not np.isfinite(success).all() or not np.isfinite(progress).all() or np.any((success<0)|(success>1)) or np.any((progress<0)|(progress>1)):
            raise ValueError('finite bounded purpose/progress outcomes required')
        mass=None if weights is None else np.asarray(weights[key],dtype=float)
        if mass is not None and (mass.shape!=success.shape or not np.isfinite(mass).all() or np.any(mass<0) or not np.allclose(mass.sum(1),1,rtol=0,atol=1e-8)):
            raise ValueError('normalized nonnegative complete model masses required')
        all_zero=all_zero and not np.any(success if mass is None else success[mass>0])
        evidence[key]=(success,progress,mass)
    first=next(iter(evidence));adjusted=context
    if all_zero:
        adjusted=progress_context(context,names,evidence[first][1],weights=evidence[first][2])
    means={};errors={};regrets={};lower={}
    for key,(success,progress,mass) in evidence.items():
        signal=progress if all_zero else success
        mean=signal.mean(1) if mass is None else (signal*mass).sum(1)
        best=max((i for i,name in enumerate(names) if name in viable),key=lambda i:mean[i])
        error=np.zeros(len(names)) if mass is not None else (signal[best]-signal).std(1,ddof=1)/np.sqrt(signal.shape[1])
        regret=mean[best]-mean
        means[key]=list(map(float,mean));errors[key]=list(map(float,error));regrets[key]=list(map(float,regret))
        lower[key]=regret-error_multiplier*error
    worst=np.maximum(0,np.maximum.reduce(list(lower.values())))
    unavoidable=min(float(worst[i]) for i,name in enumerate(names) if name in viable)
    limit=unavoidable+max_regret
    allowed=[name for i,name in enumerate(names) if name in viable and worst[i]<=limit+1e-12]
    policy=Policy(principle_priority='finite');original=compile_batch([adjusted])
    prior=policy.decide(original,False).records(original)[0]
    filtered=copy.deepcopy(adjusted);filtered['actions']=[a for a in filtered['actions'] if a['id'] in allowed]
    batch=compile_batch([filtered]);decision=policy.decide(batch,False).records(batch)[0]
    guard=dict(allowed=allowed,rejected=[n for n in names if n not in allowed],max_model_regret=max_regret,
        unavoidable_model_regret=unavoidable,allowed_model_regret=limit,model_robust=True,
        personality_action=prior['action_id'],guard_changed=decision['action_id']!=prior['action_id'],
        bounds={n:dict(lower_bound=float(worst[i])) for i,n in enumerate(names)},
        model_means=means,model_regrets=regrets,model_sampling_errors=errors,error_multiplier=error_multiplier,
        signal='goal_progress_on_constant_success' if all_zero else 'terminal_success',
        uncertainty='declared model disagreement plus paired sampling diagnostics; not true-game guarantees',
        personality_reference_model=first)
    if all_zero:guard['primary_constant']=0.
    adjusted=copy.deepcopy(adjusted);adjusted['facts']['model_selection']='共有モデルを人格評価の基準とし、宣言した相手仮説すべての避けられる目的損を比較'
    return adjusted,decision,guard
