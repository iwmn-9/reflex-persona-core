"""Optional budgeted game-independent planning; shared reflex Policy is unchanged.

The game supplies public/model states, legal moves, transitions, chance hypotheses
and a personality-aware continuation selector. No world/hidden-roster access here.
"""
from dataclasses import dataclass, asdict
import copy
import numpy as np
from .core import FEATURES, NEEDS, VALUES, TRAITS, MAX_OUTCOMES
from .examples import effect


@dataclass(frozen=True)
class PlanningBudget:
    enabled: bool = True
    depth: int = 3
    max_nodes: int = 256
    width: int = 3
    chance_samples: int = 3

    def __post_init__(self):
        if type(self.enabled) is not bool: raise ValueError('enabled must be boolean')
        for key,low,high in (('depth',1,16),('max_nodes',0,8192),('width',1,256),('chance_samples',1,8)):
            value=getattr(self,key)
            if type(value) is not int or not low<=value<=high: raise ValueError(f'{key} outside [{low},{high}]')


def vector(row):
    return np.array([row['objective']]+[row['needs'].get(k,0) for k in NEEDS]+
                    [row['values'].get(k,0) for k in VALUES]+[row['style'].get(k,0) for k in TRAITS]+[row['cost']],dtype=float)


def from_vector(v,p):
    # A convex mean of bounded effects can exceed an endpoint by one ULP.
    # Correct only roundoff; materially invalid inputs still fail at the core.
    v=np.asarray(v,dtype=float)
    v=np.where((v>1)&(v<=1+1e-12),1.,np.where((v< -1)&(v>=-1-1e-12),-1.,v))
    if 1<p<=1+1e-12: p=1.
    return effect(float(v[0]),dict(zip(NEEDS,map(float,v[1:6]))),dict(zip(VALUES,map(float,v[6:16]))),
                  float(v[-1]),float(p),dict(zip(TRAITS,map(float,v[16:21]))))


def compress_outcomes(outcomes, max_outcomes=MAX_OUTCOMES):
    """Keep probability/first moments and worst objective branch; approximate risk.

    Intra-bucket downside and correlations with a persona's value axes can be lost.
    No claim that the retained worst objective is worst for every personality.
    A one-slot budget retains only the mean, including for the objective.
    """
    if type(max_outcomes) is not int or not 1<=max_outcomes<=MAX_OUTCOMES:
        raise ValueError('bounded outcome capacity required')
    if len(outcomes)<=max_outcomes: return copy.deepcopy(outcomes)
    ordered=sorted(outcomes,key=lambda r:r['objective'])
    result=[] if max_outcomes==1 else [copy.deepcopy(ordered[0])]
    start=0 if max_outcomes==1 else 1
    for indices in np.array_split(np.arange(start,len(ordered)),max(1,max_outcomes-1)):
        p=sum(ordered[int(i)]['p'] for i in indices)
        if not p: continue
        average=sum((ordered[int(i)]['p']*vector(ordered[int(i)]) for i in indices),np.zeros(len(FEATURES)))/p
        result.append(from_vector(average,p))
    return result


class Exhausted(Exception): pass


def refine(root_states,adapter,budget):
    """Iterative deepening, complete horizons only; all root candidates retained.

    Depth includes the already-computed immediate own action (depth=1 reflex).
    max_nodes bounds ADDITIONAL successor calls, including sampled chance draws;
    baseline/root construction and game callback work are recorded separately by
    the caller. This is not a wall-clock limit. Inner actions may be beam-pruned.

    Adapter: terminal(s), chance(s), draws(s,k)->[(p,s)], legal(s), order(s,moves),
    step(s,move), select(s,{move:[(p,leaf)]})->move. select must only use the
    modeled observer's information; it must not peek at real hidden future state.
    """
    if not root_states or len(set(root_states))!=len(root_states): raise ValueError('distinct roots required')
    result={name:[(1.,state)] for name,state in root_states.items()}
    stats=dict(config=asdict(budget),root_candidates=len(root_states),additional_nodes=0,
               reached_depth=1,requested_depth=budget.depth,continuation_choices=0,
               beam_pruned_actions=0,chance_draws=0,budget_exhausted=False,
               abandoned_partial_horizon=False,used=False)
    if not budget.enabled or budget.depth==1 or budget.max_nodes==0: return result,stats

    def debit():
        if stats['additional_nodes']>=budget.max_nodes: raise Exhausted()
        stats['additional_nodes']+=1

    def visit(state,remaining):
        if not remaining or adapter.terminal(state): return [(1.,state)]
        if adapter.chance(state):
            # Reserve the complete hypothesis set before constructing successors.
            # Charge all generated children even if a later subtree exhausts budget.
            if stats['additional_nodes']+budget.chance_samples>budget.max_nodes: raise Exhausted()
            draws=adapter.draws(state,budget.chance_samples)
            if not draws or len(draws)>budget.chance_samples or not all(np.isfinite(p) and p>0 for p,_ in draws) or abs(sum(p for p,_ in draws)-1)>1e-8:
                raise ValueError('chance hypotheses must have positive normalized mass')
            for _ in draws: debit()
            stats['chance_draws']+=len(draws)
            leaves=[]
            for p,child in draws:
                leaves.extend((p*q,s) for q,s in visit(child,remaining))
            return leaves
        moves=tuple(adapter.legal(state)); ordered=tuple(adapter.order(state,moves))
        if not moves or len(ordered)!=len(moves) or set(ordered)!=set(moves): raise ValueError('ordering must retain legal moves')
        chosen=ordered[:budget.width]; stats['beam_pruned_actions']+=len(moves)-len(chosen)
        options={}
        for move in chosen:
            debit(); options[move]=visit(adapter.step(state,move),remaining-1)
        selected=adapter.select(state,options); stats['continuation_choices']+=1
        if selected not in options: raise ValueError('selector returned an unsearched action')
        return options[selected]

    for depth in range(2,budget.depth+1):
        try:
            candidate={name:visit(state,depth-1) for name,state in root_states.items()}
        except Exhausted:
            stats.update(budget_exhausted=True,abandoned_partial_horizon=True)
            break
        # Do not mix deep forecasts for early roots with shallow ones for later roots.
        result=candidate; stats.update(reached_depth=depth,used=True)
    return result,stats
