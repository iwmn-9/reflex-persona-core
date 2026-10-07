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


def rollout(c, initial, *, observe, advance, terminal, horizon, seeds=(0,), continuation=None, assess=None):
    """Return complete root Branches and a diagnostic continuation audit.

    observe(state, actor_state) returns an immediate Policy context.
    advance(state, action, model_seed) returns (successor, realized FLOW effect).
    terminal includes owner death/known episode end where appropriate.
    Each model seed is one coherent trajectory, with equal probability. Caller
    keeps actual RNG separate. An optional scripted continuation is a control.
    No hypothetical experience is learned; hypothetical intent/mode is carried.
    Optional assess(final_state) records a game-owned endpoint assessment once.
    """
    compile_batch([c])
    if type(horizon) is not int or not 1 <= horizon <= 16:
        raise ValueError('bounded horizon required')
    if not isinstance(seeds, (tuple, list)) or not 1 <= len(seeds) <= MAX_OUTCOMES or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('distinct bounded model seeds required')
    if terminal(initial):
        raise ValueError('live initial state required')
    roots=sorted(a['id'] for a in c['actions'] if a['legal'] and not a['known_failure'])
    paths={};audit={};policy=Policy()
    for root in roots:
        branches=[];traces=[]
        for seed in seeds:
            state=copy.deepcopy(initial);memory=copy.deepcopy(c['state']);flows=[];actions=[];ended=None
            for tick in range(horizon):
                if terminal(state):
                    if ended is None:ended=tick
                    flows.append(effect());continue
                cc=copy.deepcopy(c) if tick == 0 else observe(copy.deepcopy(state),copy.deepcopy(memory))
                for key in ('scope','seed','personality','values','objective'):
                    if cc[key] != c[key]:
                        raise ValueError('continuation changed actor identity or preferences: '+key)
                b=compile_batch([cc]);d=policy.decide(b,False)
                key=root if tick == 0 else (d.records(b)[0]['action_id'] if continuation is None else continuation(copy.deepcopy(state),copy.deepcopy(cc)))
                viable=[a['id'] for a in cc['actions'] if a['legal'] and not a['known_failure']]
                if key not in viable:raise ValueError('continuation must choose a viable root')
                # Forced first roots still carry the Policy's current mode and
                # need, with the actual selected root's intent/age transition.
                memory=replace(d,action=np.array([b.ids[0].index(key)])).records(b)[0]['next_state']
                state,row=advance(state,key,seed)
                if row.get('p') != 1:raise ValueError('branch probability must not be counted twice')
                flows.append(copy.deepcopy(row));actions.append(key)
            branches.append(Branch(1/len(seeds),tuple(flows),(1.,)*horizon))
            trace=dict(seed=seed,actions=actions,absorbed_at=ended,terminal=bool(terminal(state)))
            if assess is not None:trace['assessment']=copy.deepcopy(assess(copy.deepcopy(state)))
            traces.append(trace)
        paths[root]=tuple(branches);audit[root]=traces
    return paths,audit
