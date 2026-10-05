"""Optional bounded probabilistic lookahead, outside the reflex scoring path.

Game supplies perceived transitions including opponent tendencies, not hidden facts.
This function does not infer opponent personalities from scratch.
"""
import numpy as np
from .core import FEATURES, NEEDS, VALUES, TRAITS, number, MAX_OUTCOMES


def forecast(branches, depth=2, discount=.8, budget=64):
    """Evaluate a finite outcome tree; return expected normalized effects and nodes.

Each branch: p, effects[22], children[]. Children are perceived conditional outcomes
under a fixed proposed continuation, not an omniscient minimax opponent. Personality
scores these discounted features in Policy. Adapter may compare a few continuations.
Reject an over-budget tree instead of silently clipping probability mass.
"""
    if type(depth) is not int or not 1<=depth<=3 or type(budget) is not int or not 1<=budget<=128:
        raise ValueError("depth 1..3 / budget 1..128 required")
    number(discount,0,1); used=0; leaves=[]
    scale=sum(discount**i for i in range(depth))
    def visit(items, remaining, probability, prefix):
        nonlocal used
        if not items or abs(sum(number(b["p"],0,1) for b in items)-1)>1e-8: raise ValueError("branch probabilities must sum to one")
        for b in items:
            used+=1
            if used>budget: raise ValueError("forecast budget exceeded")
            if set(b)!={"p","effects","children"}: raise ValueError("invalid forecast branch")
            if not isinstance(b["children"],list): raise ValueError("children must be a list")
            f=np.asarray(b["effects"],dtype=float)
            if f.shape!=(len(FEATURES),) or not np.isfinite(f).all() or (abs(f)>1).any() or f[-1]<0: raise ValueError("invalid forecast effects")
            accumulated=prefix+discount**(depth-remaining)*f
            if remaining>1 and b["children"]: visit(b["children"],remaining-1,probability*b["p"],accumulated)
            else:
                leaves.append(dict(p=probability*b["p"],effects=accumulated/scale))
                if len(leaves)>MAX_OUTCOMES: raise ValueError("forecast exceeds eight retained outcome paths")
    visit(branches,depth,1,np.zeros(len(FEATURES)))
    result=sum((r["p"]*r["effects"] for r in leaves),np.zeros(len(FEATURES)))
    return dict(effects=result,outcomes=leaves,nodes=used,depth=depth)


def as_outcomes(prediction):
    """Preserve predicted outcome paths for the reflex risk/benefit scorer."""
    result=[]
    for row in prediction["outcomes"]:
        f=row["effects"]
        result.append(dict(p=row["p"],objective=float(f[0]),needs=dict(zip(NEEDS,map(float,f[1:6]))),
                           values=dict(zip(VALUES,map(float,f[6:16]))),style=dict(zip(TRAITS,map(float,f[16:21]))),cost=float(f[-1])))
    return result
