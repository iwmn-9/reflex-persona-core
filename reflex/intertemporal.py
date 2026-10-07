"""Optional discounted FLOW forecasts, selected by the unchanged persona Policy.

Adapters provide complete, equally long conditional trajectories. Each effect
is earned/spent once; stocks and terminal prizes must not be repeated as income.
This is finite foresight, not a learned terminal value or a promise of payback.
"""
from dataclasses import dataclass
import copy
import numpy as np
from .core import compile_batch, identifier, number, MAX_OUTCOMES
from .examples import action
from .planning import vector, from_vector
from .deliberation import JointForecast


@dataclass(frozen=True)
class Branch:
    probability: float
    effects: tuple
    confidence: tuple


def patience(c):
    """Explicit provisional game-design mapping, not psychological measurement.

    Conscientiousness extends the time weight; neuroticism and current unmet
    physiology/safety shorten it. Growth/value preferences remain in Policy.
    More patience alone is not a claim of greater intelligence.
    """
    compile_batch([c])
    urgent=max((c['needs'][k]['deficit'] for k in ('physiology','safety')
                if c['needs'][k]['enabled']),default=0.)
    t=c['personality']
    return float(np.clip(.45+.45*t['conscientiousness']+.1*(1-t['neuroticism'])-.35*urgent,.25,1.))


def forecast(c,branches,*,horizon,unit,target,max_regret=.15,discount=None,plan_roots=None):
    """One actor, all viable roots, <=8 coherent conditional branches per root.

    Discounted flows use ONE common denominator sum(gamma**t), keeping effects
    bounded without clipping away costs. A zero flow pads an absorbing terminal;
    an unknown/truncated future must NOT be padded by the adapter. Probabilities
    describe modeled branches, confidence discounts their positive estimates
    (never their losses/costs). Root confidence is still applied by Policy.
    Future effects are not returned as immediate empirical observations.
    """
    compile_batch([c]);identifier(unit);identifier(target)
    if type(horizon) is not int or not 1<=horizon<=16:raise ValueError('bounded common flow horizon required')
    gamma=patience(c) if discount is None else number(discount,0,1)
    number(max_regret,0,2)
    roots={a['id']:a for a in c['actions'] if a['legal'] and not a['known_failure']}
    if plan_roots is None:plan_roots={k:k for k in roots}
    if not isinstance(plan_roots,dict) or not isinstance(branches,dict) or set(branches)!=set(plan_roots) or set(plan_roots.values())!=set(roots):
        raise ValueError('complete viable-root coverage required')
    if not 1<=len(plan_roots)<=256:raise ValueError('bounded proposal count required')
    for key in plan_roots:identifier(key)
    weights=gamma**np.arange(horizon);normalizer=float(weights.sum())
    future=copy.deepcopy(c);future['actions']=[]
    future['facts']['planning_target']='discounted incremental flows; conditional continuation; separate from immediate feedback'
    plans={};purpose={};audit={}
    for index,proposal in enumerate(sorted(plan_roots)):
        root=plan_roots[proposal];rows=branches[proposal]
        if not isinstance(rows,(tuple,list)) or not 1<=len(rows)<=MAX_OUTCOMES:raise ValueError('bounded coherent branches required')
        probability=[];outcomes=[];path_values=[]
        for branch in rows:
            if not isinstance(branch,Branch):raise ValueError('Branch required')
            p=number(branch.probability,0,1);probability.append(p)
            if len(branch.effects)!=horizon or len(branch.confidence)!=horizon:raise ValueError('equal complete horizons and confidence paths required')
            vectors=[]
            for row,q in zip(branch.effects,branch.confidence):
                q=number(q,0,1)
                # Reuse the public numeric contract for every flow, including
                # keys, bounds, probability=1 and disabled-need semantics.
                check=copy.deepcopy(c);check['actions']=[action('flow',copy.deepcopy(row))]
                compile_batch([check])
                if row['p']!=1:raise ValueError('probability belongs to the branch, not an individual flow')
                v=vector(row);v[:-1]*=np.where(v[:-1]>0,q,1.)
                vectors.append(v)
            matrix=np.array(vectors);total=np.einsum('t,td->d',weights,matrix)/normalizer
            outcomes.append(from_vector(total,p))
            path_values.append(dict(probability=p,objective=matrix[:,0].tolist(),
                cost=matrix[:,-1].tolist(),confidence=list(branch.confidence)))
        if abs(sum(probability)-1)>1e-8:raise ValueError('normalized branch probabilities required')
        key=f'flow-plan-{index:03d}';a=copy.deepcopy(roots[root]);a.update(id=key,outcomes=outcomes)
        # Familiarity and switching are once-per-root terms outside the effect
        # vector. Normalize them by the SAME denominator as all earned flows;
        # otherwise a longer window would invent extra inertia/familiarity.
        a['familiarity']/=normalizer;a['switch_cost']/=normalizer
        # Plan IDs differ from real intent IDs. Preserve the zero switch fee
        # for continuing the current root without rewriting actor state.
        if c['state']['intent_action']==root:a['switch_cost']=0.
        future['actions'].append(a);plans[key]=(root,)
        purpose[key]=float(sum(r['p']*r['objective']*(a['confidence'] if r['objective']>0 else 1.) for r in outcomes))
        audit[key]=dict(root=root,paths=path_values,discounted_objective=purpose[key])
        if proposal!=root:audit[key]['proposal']=proposal
    compile_batch([future])
    return JointForecast((future,),plans,purpose,horizon,max_regret,
        dict(discount=gamma,normalizer=normalizer,unit=unit,plans=audit,
             contract='incremental flows; no unseen tail value; confidence is not a calibrated success probability'),
        target=target)
