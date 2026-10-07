"""Optional finite public-state flow rollouts with persona-owned continuation.

The game supplies observations, exact modeled transitions and terminal tests.
The shared helper never invents a game score, hidden information or tail value.
Replanning still uses a reflex continuation model, not recursive optimal play.
"""
from dataclasses import replace
import copy
import numpy as np
from .core import Policy, compile_batch, MAX_OUTCOMES
from .examples import effect
from .intertemporal import Branch


def rollout(c, initial, *, observe, advance, terminal, horizon, seeds=(0,), continuation=None, assess=None, observer=None, schedule=(), roots=None, record_choices=False):
    """Return complete root Branches and a diagnostic continuation audit.

    observe(state, actor_state) returns an immediate Policy context.
    advance(state, action, model_seed) returns (successor, realized FLOW effect).
    terminal includes owner death/known episode end where appropriate.
    Each model seed is one coherent trajectory, with equal probability. Caller
    keeps actual RNG separate. An optional scripted continuation is a control.
    No hypothetical experience is learned; hypothetical intent/mode is carried.
    Optional assess(final_state) records a game-owned endpoint assessment once.
    Optional ModelObserver forks actual progress/pressure for each branch. It
    updates only modeled state; continuation remains reflex, not recursive search.
    """
    compile_batch([c])
    if type(horizon) is not int or not 1 <= horizon <= 16:
        raise ValueError('bounded horizon required')
    if not isinstance(seeds, (tuple, list)) or not 1 <= len(seeds) <= MAX_OUTCOMES or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('distinct bounded model seeds required')
    if terminal(initial):
        raise ValueError('live initial state required')
    if observer is not None:
        from .model_observer import ModelObserver
        if not isinstance(observer,ModelObserver):raise ValueError('ModelObserver required')
    viable_roots={a['id'] for a in c['actions'] if a['legal'] and not a['known_failure']}
    if roots is None:roots=sorted(viable_roots)
    elif not isinstance(roots,(tuple,list)) or not roots or len(set(roots))!=len(roots) or not set(roots)<=viable_roots:
        raise ValueError('explicit viable root subset required')
    if not isinstance(schedule,(tuple,list)) or len(schedule)>=horizon or any(not isinstance(k,str) for k in schedule):
        raise ValueError('bounded future action schedule required')
    if schedule and continuation is not None:raise ValueError('one continuation mechanism required')
    paths={};audit={};policy=Policy()
    for root in roots:
        branches=[];traces=[]
        for seed in seeds:
            state=copy.deepcopy(initial);memory=copy.deepcopy(c['state']);flows=[];actions=[];ended=None;choices=[];fallbacks=[]
            modeled=None if observer is None else observer.fork();observations=[]
            for tick in range(horizon):
                if terminal(state):
                    if ended is None:ended=tick
                    flows.append(effect());continue
                cc=copy.deepcopy(c) if tick == 0 else observe(copy.deepcopy(state),copy.deepcopy(memory))
                req=None;allowed=None;progress_audit=None
                if modeled is not None:
                    cc,req,allowed,progress_audit=modeled.prepare(state,cc,prepared=tick==0)
                for key in ('scope','seed','personality','values','objective'):
                    if cc[key] != c[key]:
                        raise ValueError('continuation changed actor identity or preferences: '+key)
                b=compile_batch([cc])
                legal=b.legal if allowed is None or tick==0 else b.legal&np.array([[k in allowed for k in b.ids[0]]])
                d=policy.decide(replace(b,legal=legal),False)
                key=root if tick == 0 else (d.records(b)[0]['action_id'] if continuation is None else continuation(copy.deepcopy(state),copy.deepcopy(cc)))
                viable=[a['id'] for a in cc['actions'] if a['legal'] and not a['known_failure']]
                if schedule or record_choices:
                    permitted=viable if allowed is None or tick==0 else [k for k in viable if k in allowed]
                    choices.append(sorted(permitted))
                    if 0<tick<=len(schedule):
                        wanted=schedule[tick-1]
                        if wanted in permitted:key=wanted
                        else:fallbacks.append(tick)
                if key not in viable:raise ValueError('continuation must choose a viable root')
                if tick>0 and allowed is not None and key not in allowed:raise ValueError('continuation ignored modeled progress mask')
                # Forced first roots still carry the Policy's current mode and
                # need, with the actual selected root's intent/age transition.
                memory=replace(d,action=np.array([b.ids[0].index(key)])).records(b)[0]['next_state']
                before=copy.deepcopy(state);state,row=advance(state,key,seed)
                if row.get('p') != 1:raise ValueError('branch probability must not be counted twice')
                flows.append(copy.deepcopy(row));actions.append(key)
                if modeled is not None:
                    modeled.advance(before,state,key,cc,req)
                    observations.append(dict(tick=cc['tick'],allowed=sorted(allowed),progress=progress_audit,
                        checkpoint=modeled.record()))
            branches.append(Branch(1/len(seeds),tuple(flows),(1.,)*horizon))
            trace=dict(seed=seed,actions=actions,absorbed_at=ended,terminal=bool(terminal(state)))
            if schedule or record_choices:trace.update(choices=choices,schedule_fallbacks=fallbacks)
            if assess is not None:trace['assessment']=copy.deepcopy(assess(copy.deepcopy(state)))
            if modeled is not None:trace['modeled_observations']=observations
            traces.append(trace)
        paths[root]=tuple(branches);audit[root]=traces
    return paths,audit
