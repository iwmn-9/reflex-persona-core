"""Opt-in completion of declined friendly roots, never a future self model.

Only an explicitly coordinated game adapter can request completion. The common
boundary retains final progress/waste masks, Policy tier and its .025 near-tie
band. The combat search freezes noncolliding intentions; if compatible roots
cannot all avoid a known friendly collision, it retains the original choices.
No opponent intent, actual hit RNG, learned update or persistent cache is used.
"""
from collections import Counter
from dataclasses import asdict, replace
from itertools import product
import copy
import numpy as np
from .core import compile_batch
from .judgment import avoid_waste


def constrained_completion(contexts, records, root_allowed, exact_masks, policy, complete, blocked_changes=None):
    """Validate an adapter completion before DecisionLoop's atomic commit."""
    if not contexts or len(records)!=len(contexts) or len(root_allowed)!=len(contexts):
        raise ValueError('one current record and allowed-root set per owner required')
    if len({(c['scope']['game'],c['scope']['episode'],c['scope']['npc']) for c in contexts})!=len(contexts):
        raise ValueError('distinct completion owners required')
    scopes={(c['scope']['game'],c['scope']['episode'],c['tick']) for c in contexts}
    if len(scopes)!=1:
        raise ValueError('root completion cannot coordinate independent episodes or ticks')
    b=compile_batch(contexts)
    if any(r['context_hash']!=h for r,h in zip(records,b.hashes)):
        raise ValueError('completion record/context owner mismatch')
    mask=np.array([[k in allowed for k in list(ids)+[None]*(b.legal.shape[1]-len(ids))]
        for ids,allowed in zip(b.ids,root_allowed)])
    guarded,_=avoid_waste(replace(b,legal=b.legal&mask),exact_masks)
    d=policy.decide(guarded,False)
    best=np.max(np.where(d.eligible,d.scores,-np.inf),axis=1)
    options=[{k:float(d.scores[i,j]) for j,k in enumerate(ids)
        if d.eligible[i,j] and d.scores[i,j]>=best[i]-.025} for i,ids in enumerate(b.ids)]
    roots=tuple(r['action_id'] for r in records)
    audit=dict(adopted=False,before=list(roots),after=list(roots),changed=0,
        reason='original root outside final immediate persona band',persona_regret=[])
    # Reader or unusual caller decisions are not silently reinterpreted.
    if any(k not in pool for k,pool in zip(roots,options)):return records,audit
    if blocked_changes is not None:
        if len(blocked_changes)!=len(roots):raise ValueError('one blocked-alternative set per owner required')
        # Proof reconciliation may restore a blocked baseline when no better
        # plan was demonstrated. That is not support for introducing a NEW
        # expired stop or repeated failed root. Filter AFTER tier/band scoring.
        options=[{k:v for k,v in pool.items() if k==root or k not in blocked}
            for pool,root,blocked in zip(options,roots,blocked_changes)]
    proposal,metadata=complete(copy.deepcopy(contexts),roots,copy.deepcopy(options))
    audit['adapter']=copy.deepcopy(metadata)
    if proposal is None:
        audit['reason']=metadata['reason'];return records,audit
    if not isinstance(proposal,tuple) or len(proposal)!=len(roots) or any(
            k not in pool for k,pool in zip(proposal,options)):
        raise ValueError('completion must retain every owner persona tier, band and progress/waste mask')
    selected=np.array([ids.index(k) for ids,k in zip(b.ids,proposal)],dtype=int)
    candidates=replace(d,action=selected).records(guarded)
    completed=[old if a==z else new for old,new,a,z in zip(records,candidates,roots,proposal)]
    audit.update(adopted=proposal!=roots,after=list(proposal),changed=sum(a!=z for a,z in zip(roots,proposal)),
        reason='compatible friendly root completion' if proposal!=roots else 'unchanged',
        persona_regret=[float(best[i]-options[i][k]) for i,k in enumerate(proposal)])
    return completed,audit


def combat_completion(w, actors, contexts, roots, options):
    """Bounded search on the already explicit, same-world friendly team.

    No actor outside an existing collision may change. Minimize changed roots,
    then maximize unchanged Policy scores, then use canonical root ordering.
    The caller owns all masks; this adapter cannot widen any actor's options.
    """
    from .combat import alive, legal
    actors=tuple(actors)
    if not actors or len(actors)>3 or len(contexts)!=len(actors) or len(roots)!=len(actors) or len(options)!=len(actors):
        raise ValueError('one context per explicit bounded friendly team required')
    team=w.units[actors[0]].team
    if actors!=alive(w,team):raise ValueError('one explicit same-world friendly team required')
    public=dict(goal=w.goal,public_tick=str(w.tick),public_hold=str(w.hold),public_secured=str(w.secured),
        walls=str(w.walls),cover=str(w.covers))
    public.update({f'public_unit_{i}':str(asdict(u)) for i,u in enumerate(w.units)})
    for i,c,root,pool in zip(actors,contexts,roots,options):
        if (c['scope']!=dict(game='combat-action-v1',episode=f'{w.goal}-{c["seed"]}',npc=f'unit-{i}')
                or c['tick']!=w.tick or any(c['facts'].get(k)!=v for k,v in public.items())):
            raise ValueError('completion owner/public world mismatch')
        if root not in legal(w,i) or not set(pool)<=set(legal(w,i)):
            raise ValueError('completion requires only current legal roots')
    if len({c['seed'] for c in contexts})!=1:raise ValueError('one explicit combat episode required')
    counts=Counter(k for k in roots if k.startswith('move:'))
    colliding={i for i,k in enumerate(roots) if counts[k]>1}
    metadata=dict(reason='no friendly collision',colliding=sorted(colliding),combinations=0,feasible=0)
    if not colliding:return None,metadata
    pools=[tuple(sorted(pool)) if i in colliding else (roots[i],) for i,pool in enumerate(options)]
    winner=None;quality=None
    for candidate in product(*pools):
        metadata['combinations']+=1
        moves=[k for k in candidate if k.startswith('move:')]
        if len(moves)!=len(set(moves)):continue
        metadata['feasible']+=1
        value=(-sum(a!=b for a,b in zip(roots,candidate)),sum(pool[k] for pool,k in zip(options,candidate)),candidate)
        if quality is None or value>quality:winner,quality=candidate,value
    metadata['reason']='no compatible collision-free completion' if winner is None else 'compatible completion found'
    return winner,metadata
