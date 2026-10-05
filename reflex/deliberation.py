"""Joint horizon choice, with immediate experience kept on a separate target.

Games supply legal root combinations, modeled horizon effects and a purpose
score. This module chooses among them using each owner's existing Policy.
Purpose regret is an explicit competence constraint, not a new personality.
"""
from dataclasses import dataclass,replace
import copy
import numpy as np
from .core import Policy,compile_batch,digest,identifier


@dataclass(frozen=True)
class JointForecast:
    contexts: tuple
    roots: dict
    purpose: dict
    horizon: int
    max_regret: float=.15
    audit: dict=None
    target: str='game-purpose'


def select(base,forecast,policy=None,root_allowed=None):
    if not isinstance(forecast,JointForecast):raise ValueError('JointForecast required')
    identifier(forecast.target)
    if type(forecast.horizon) is not int or not 1<=forecast.horizon<=16:raise ValueError('bounded horizon required')
    if isinstance(forecast.max_regret,bool) or not np.isfinite(forecast.max_regret) or not 0<=forecast.max_regret<=2:
        raise ValueError('bounded purpose regret required')
    if len(forecast.contexts)!=len(base) or not base:raise ValueError('one forecast per actor required')
    if root_allowed is not None and (len(root_allowed)!=len(base) or any(not isinstance(a,set) for a in root_allowed)):
        raise ValueError('one allowed real-root set per actor required')
    ids=set(forecast.roots)
    if not ids or set(forecast.purpose)!=ids:raise ValueError('purpose and roots must cover the same plans')
    for key,roots in forecast.roots.items():
        if len(roots)!=len(base):raise ValueError('one root per actor per joint plan')
        for c,root in zip(base,roots):
            valid={a['id'] for a in c['actions'] if a['legal'] and not a['known_failure']}
            if root not in valid:raise ValueError('joint plan contains nonviable real root')
    for c,fc in zip(base,forecast.contexts):
        for key in c:
            if key not in ('actions','facts') and fc.get(key)!=c[key]:raise ValueError('forecast changed actor contract: '+key)
        if any(fc['facts'].get(k)!=v for k,v in c['facts'].items()):raise ValueError('forecast changed perceived facts')
        if {a['id'] for a in fc['actions']}!=ids:raise ValueError('same joint plans for every actor required')
        if not all(a['legal'] and not a['known_failure'] for a in fc['actions']):raise ValueError('plan feasibility must be explicit at generation')
    purpose=np.array([forecast.purpose[k] for k in sorted(ids)],float)
    if not np.isfinite(purpose).all() or (abs(purpose)>1).any():raise ValueError('bounded model purpose scores required')
    # Root action legality is unchanged. This mask is explicitly the modeled
    # purpose corridor, not a declaration that excluded actions are illegal.
    names=tuple(sorted(ids))
    supported=np.array([root_allowed is None or all(root in permitted for root,permitted in zip(forecast.roots[k],root_allowed)) for k in names])
    coverage=dict(supported_plans=int(supported.sum()),progress_excluded=int((~supported).sum()))
    if not supported.any():
        return None,dict(target='modeled-horizon',adopted=False,horizon=forecast.horizon,
            reason='no joint plan with supported progress roots',plans=len(names),metadata=copy.deepcopy(forecast.audit or {}),**coverage)
    best=float(np.max(purpose[supported]));allowed=supported&(purpose>=best-forecast.max_regret-1e-12)
    contexts=[]
    for fc in forecast.contexts:
        c=copy.deepcopy(fc);by={a['id']:a for a in c['actions']};c['actions']=[by[k] for k in names];contexts.append(c)
    b=compile_batch(contexts);p=policy or Policy();d=p.decide(replace(b,legal=b.legal&allowed[None,:]),False)
    # Cooperation cannot silently overwrite an individual's strongest value.
    # Accept a joint plan only when every actor's current tier permits it.
    votes=d.eligible.sum(0);eligible=allowed&d.eligible.all(0)
    if not eligible.any():
        return None,dict(target='modeled-horizon',adopted=False,horizon=forecast.horizon,
            reason='no joint personality-compatible plan in purpose corridor',
            plans=len(names),metadata=copy.deepcopy(forecast.audit or {}),
            tier_options_per_actor=d.eligible.sum(1).tolist(),**coverage)
    maxima=np.max(np.where(d.eligible,d.scores,-np.inf),axis=1)
    regret=(d.scores-maxima[:,None]).mean(0)
    j=int(np.argmax(np.where(eligible,regret,-np.inf)));key=names[j]
    records=replace(d,action=np.full(len(base),j,dtype=int)).records(b)
    decisions=[]
    for i,(c,root,record) in enumerate(zip(base,forecast.roots[key],records)):
        a=next(a for a in c['actions'] if a['id']==root)
        old=c['state'];age=min(old['age']+1,1000000) if old['intent_action']==root else 0
        record.update(context_hash=digest(c),action_id=root,target=a['target'],reading_used=True)
        record['next_state'].update(intent_action=root,age=age)
        decisions.append(record)
    audit=dict(target='modeled-horizon',adopted=True,horizon=forecast.horizon,selected_plan=key,
        selected_purpose=float(purpose[j]),best_purpose=best,max_regret=forecast.max_regret,
        progress_excluded=int((~supported).sum()),
        plans=len(names),purpose_excluded=int((~allowed).sum()),tier_agreement=int(votes[j]),actors=len(base),
        subjective_regret=float(regret[j]),metadata=copy.deepcopy(forecast.audit or {}))
    return decisions,audit
