"""Optional learned purpose arbitration before the real decision commits.

Adapters declare public features, their stable scope and purpose meaning. A
teacher residual is an empirical diagnostic, never a guarantee about the game.
No fitted model is enabled by default; immediate experience stays immediate.
"""
from dataclasses import dataclass,replace
import copy
import numpy as np
from .core import compile_batch,digest,identifier,FEATURES
from .judgment import avoid_waste
from .paired_value import predict,arbitrate


@dataclass(frozen=True)
class ValueProposal:
    model: dict
    errors: object
    features: dict
    scope: str
    target: str
    max_regret: float=.12


def constrain(context,incumbent,proposal,policy,allowed,exact):
    if not isinstance(proposal,ValueProposal):raise ValueError('explicit ValueProposal required')
    identifier(proposal.scope);identifier(proposal.target)
    viable=[a['id'] for a in context['actions'] if a['legal'] and not a['known_failure']]
    if set(proposal.features)!=set(viable):raise ValueError('public features for every viable root required')
    names=list(viable);x=np.asarray([proposal.features[k] for k in names],float)
    # Validate even a forced action; duplicate only for shape validation, never
    # to manufacture a comparison or an adoption opportunity.
    prediction,_=predict(proposal.model,x if len(x)>1 else np.repeat(x,2,axis=0))
    errors=np.asarray(proposal.errors,float)
    if errors.shape!=(prediction.shape[1],) or not np.isfinite(errors).all() or (errors<0).any() or not np.isfinite(proposal.max_regret) or not 0<=proposal.max_regret<=1:
        raise ValueError('finite empirical errors and bounded purpose tolerance required')
    batch=compile_batch([context])
    mask=np.array([[[f in exact[k] for f in FEATURES] for k in batch.ids[0]]])
    batch,_=avoid_waste(replace(batch,legal=batch.legal&np.array([[k in allowed for k in batch.ids[0]]])),mask)
    scored=policy.decide(batch,False)
    permitted=[k for j,k in enumerate(batch.ids[0]) if scored.eligible[0,j]]
    audit=dict(scope=proposal.scope,target=proposal.target,model_sha256=digest(proposal.model),changed=False,
        uncertainty='held-out paired residual is empirical; no true-game safety guarantee',
        excluded=[k for k in names if k not in permitted])
    old=incumbent['action_id']
    if old not in permitted or len(permitted)<2:
        audit['reason']='retain incumbent; no comparable alternatives in its immediate personality tier'
        return incumbent,audit
    features=[proposal.features[k] for k in permitted]
    chosen,gate=arbitrate(proposal.model,features,permitted.index(old),proposal.errors,max_regret=proposal.max_regret)
    audit.update(gate);audit.update(incumbent=old,candidate=permitted[gate['candidate']])
    if not gate['changed']:
        audit['reason']='retain incumbent; unsupported or insufficient learned purpose gain'
        return incumbent,audit
    key=permitted[chosen];index=batch.ids[0].index(key)
    decision=replace(scored,action=np.array([index])).records(batch)[0]
    decision['reading_used']=True
    audit.update(reason='supported empirical purpose gain within existing personality tier',selected=key)
    return copy.deepcopy(decision),audit
