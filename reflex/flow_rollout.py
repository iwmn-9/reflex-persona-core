"""Optional finite public-state flow rollouts with persona-owned continuation.

The game supplies observations, exact modeled transitions and terminal tests.
The shared helper never invents a game score, hidden information or tail value.
Replanning still uses a reflex continuation model, not recursive optimal play.
"""
from dataclasses import replace
import copy
import numpy as np
from .core import Policy, compile_batch, MAX_OUTCOMES,digest
from .examples import effect
from .intertemporal import Branch


def rollout(c, initial, *, observe, advance, terminal, horizon, seeds=(0,), continuation=None, assess=None, observer=None, schedule=(), roots=None, record_choices=False,policy=None,schedule_sampler=None,root_state=None,replan_once=None):
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
    if schedule_sampler is not None and (not callable(schedule_sampler) or schedule or continuation is not None or len(seeds)!=1):
        raise ValueError('a schedule pilot requires one model branch and no competing continuation')
    if root_state is not None:
        if len(roots)!=1:raise ValueError('one selected root memory required')
        check=copy.deepcopy(c);check['state']=copy.deepcopy(root_state);compile_batch([check])
    if replan_once is not None and (not callable(replan_once) or continuation is not None or schedule_sampler is not None):
        raise ValueError('one explicit feedback replan callback required')
    paths={};audit={};policy=policy or Policy()
    for root in roots:
        branches=[];traces=[]
        for seed in seeds:
            state=copy.deepcopy(initial);memory=copy.deepcopy(c['state']);flows=[];actions=[];ended=None;choices=[];fallbacks=[]
            modeled=None if observer is None else observer.fork();observations=[];replans=[];active_schedule=tuple(schedule)
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
                revised_memory=None
                if tick==1 and replan_once is not None:
                    # The callback receives a PUBLIC modeled observation, not
                    # this branch's seed. It uses its own fixed forecast bank.
                    response=replan_once(copy.deepcopy(state),copy.deepcopy(cc),
                        None if modeled is None else modeled.fork(),tuple(active_schedule),
                        None if allowed is None else allowed.copy())
                    if not isinstance(response,dict) or set(response)!= {'root','remaining','state','audit'}:
                        raise ValueError('explicit root/tail/memory/audit feedback response required')
                    key=response['root'];tail=response['remaining']
                    if key not in [a['id'] for a in cc['actions'] if a['legal'] and not a['known_failure']] or allowed is not None and key not in allowed:
                        raise ValueError('feedback replan must respect observed viable progress roots')
                    if not isinstance(tail,(tuple,list)) or len(tail)>15 or any(not isinstance(k,str) for k in tail):
                        raise ValueError('bounded feedback continuation required')
                    check=copy.deepcopy(cc);check['state']=response['state'];compile_batch([check])
                    if response['state']['intent_action']!=key:raise ValueError('feedback memory must carry the executed root')
                    revised_memory=copy.deepcopy(response['state']);active_schedule=(key,*tail)
                    replans.append(dict(tick=cc['tick'],context_hash=digest(cc),root=key,remaining=list(tail),audit=copy.deepcopy(response['audit'])))
                viable=[a['id'] for a in cc['actions'] if a['legal'] and not a['known_failure']]
                if schedule or record_choices or schedule_sampler is not None or replan_once is not None:
                    permitted=viable if allowed is None or tick==0 else [k for k in viable if k in allowed]
                    choices.append(sorted(permitted))
                    if tick>0 and schedule_sampler is not None:key=schedule_sampler(tuple(sorted(permitted)),tick)
                    if 0<tick<=len(active_schedule):
                        wanted=active_schedule[tick-1]
                        if wanted in permitted:key=wanted
                        else:fallbacks.append(tick)
                if key not in viable:raise ValueError('continuation must choose a viable root')
                if tick>0 and allowed is not None and key not in allowed:raise ValueError('continuation ignored modeled progress mask')
                # Forced first roots still carry the Policy's current mode and
                # need, with the actual selected root's intent/age transition.
                memory=replace(d,action=np.array([b.ids[0].index(key)])).records(b)[0]['next_state']
                if tick==0 and root_state is not None:memory=copy.deepcopy(root_state)
                if revised_memory is not None:memory=revised_memory
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
            if replan_once is not None:trace['feedback_replans']=replans
            if assess is not None:trace['assessment']=copy.deepcopy(assess(copy.deepcopy(state)))
            if modeled is not None:trace['modeled_observations']=observations
            traces.append(trace)
        paths[root]=tuple(branches);audit[root]=traces
    return paths,audit
