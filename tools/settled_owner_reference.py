"""Exact first-action deviation against the actual future learning controller.

Only finite deterministic PUBLIC response rules are supported. This is not a
global optimal policy or a hidden-law oracle. Every alternative is completed
by the same declared controller, learning and replanning at every real step.
"""
import copy
from dataclasses import replace
import math

import numpy as np
from reflex.core import character_key, compile_batch, digest
from reflex.decision_loop import DecisionLoop
from reflex.deliberation import select
from reflex.intertemporal import Branch
from reflex.model_observer import resolve_feedback
from reflex.purpose_plan import Goal
from reflex._horizon_consensus import prefix_forecast
from tools.run_adverse_recovery import decide
from tools.run_owned_value import cumulative_basis


def one_public_event(game, world, action):
    events=tuple(game.events(world,action))
    if len(events)!=1:
        raise ValueError('one deterministic public outcome required; no realized hidden law')
    event=events[0]
    return copy.deepcopy(event.public),copy.deepcopy(event.flow)


def evaluate(game, actor, world, arm, *, maximum_steps=16, continuation_scope='root'):
    if arm not in ('stable','expected') or type(maximum_steps) is not int or not 1<=maximum_steps<=16:
        raise ValueError('declared stable/expected controller and bounded complete reference required')
    if continuation_scope not in ('root','shared') or (continuation_scope!='root' and arm!='expected'):
        raise ValueError('explicit expected-arm continuation scope required')
    if game.terminal(world):raise ValueError('live public root required')
    original=digest(actor.record());original_world=copy.deepcopy(world)
    template=game.request(game.context(world,actor.state))
    stage=copy.deepcopy(actor)._prepare(copy.deepcopy(template))
    allowed=set(stage['progress_allowed'])
    legal=[a['id'] for a in stage['effective']['actions'] if a['legal'] and not a['known_failure']]
    if not allowed or not set(allowed)<=set(legal):raise ValueError('nonempty public progress support required')
    # The candidate is selected before constructing any complete alternatives.
    actual=copy.deepcopy(actor)
    chosen,_=decide(actual,game,copy.deepcopy(world),arm,continuation_scope=continuation_scope)
    chosen_key=chosen['decision']['action_id']
    if chosen_key not in allowed:raise ValueError('candidate root outside matched reference support')
    actual_next,actual_flow=one_public_event(game,world,chosen_key)
    resolve_feedback(actual,chosen['ticket'],game.feedback(world,actual_next,chosen_key,chosen['context'],actual_flow))
    paths={};modeled={};trajectories={}
    axes=character_key(template.context)
    for index,key in enumerate(legal):
        own=copy.deepcopy(actor);w=copy.deepcopy(world)
        flows=[];actions=[];prefixes=[];trace=[]
        for step in range(maximum_steps):
            if game.terminal(w):break
            before=digest(own.record());request=game.request(game.context(w,own.state))
            if step==0:
                result=DecisionLoop.decide_batch([(own,request)],False,_counterfactual_roots=(key,))[0]
            else:result,_=decide(own,game,w,arm,continuation_scope=continuation_scope)
            action=result['decision']['action_id']
            n,flow=one_public_event(game,w,action)
            resolve_feedback(own,result['ticket'],game.feedback(w,n,action,result['context'],flow))
            assert character_key(result['context'])==axes
            record=own.record()
            restored=DecisionLoop.from_record(game.context(w),copy.deepcopy(record),predictor=own.predictor)
            assert restored.record()==record
            own=restored
            if step==0 and key==chosen_key:assert own.record()==actual.record()
            flows.append(flow);actions.append(action)
            trace.append(dict(before=w,after=n,action=action,before_owner=before,owner=digest(record)))
            w=n
            prefixes.append(dict(assessment=game.goal(w).record(),terminal=game.terminal(w),correction=0.))
        if not game.terminal(w):raise ValueError('reference did not settle; a cutoff proxy cannot certify C2')
        name=f'root-{index:03d}'
        paths[name]=(Branch(1.,tuple(flows),(1.,)*len(flows)),)
        modeled[name]=[dict(actions=actions,prefixes=prefixes)]
        trajectories[name]=trace
    horizon=max(len(rows[0]['actions']) for rows in modeled.values())
    f,_,_=prefix_forecast(chosen['context'],paths,modeled,horizon,actor=actor,target=game.target,
        origin=Goal(**game.goal(world).record()),cumulative_preference=cumulative_basis(game,True))
    decisions,audit=select([chosen['context']],f,actor.policy,[allowed])
    if decisions is None:raise ValueError('no common feasible reference selection')
    names=tuple(sorted(f.roots))
    supported=np.array([f.roots[k][0] in allowed for k in names])
    purpose=np.array([f.purpose[k] for k in names])
    corridor=supported&(purpose>=float(np.max(purpose[supported]))-f.max_regret-1e-12)
    context=copy.deepcopy(f.contexts[0]);by={a['id']:a for a in context['actions']}
    context['actions']=[by[k] for k in names]
    batch=compile_batch([context],outcome_limit=f.outcome_limit)
    framed=dict(return_frame=f.return_frame)
    if f.return_means is not None:framed['return_means']=f.return_means
    scores=actor.policy.decide(replace(batch,legal=batch.legal&corridor[None,:]),False,**framed)
    eligible=corridor&scores.eligible[0]
    expected_best=names[int(np.argmax(np.where(eligible,scores.scores[0],-np.inf)))]
    assert expected_best==audit['selected_plan'],'offline score extraction disagrees with actual selector'
    chosen_plan=next((name for name in names if f.roots[name][0]==chosen_key),None)
    admissible=chosen_plan is not None and bool(eligible[names.index(chosen_plan)])
    regret=None
    if admissible:
        values=scores.scores[0,eligible]
        lower,upper=float(np.min(values)),float(np.max(values))
        value=float(scores.scores[0,names.index(chosen_plan)])
        regret=0. if upper-lower<=1e-12 else max(0.,upper-value)/(upper-lower)
        assert math.isfinite(regret) and 0<=regret<=1.+1e-12
    assert digest(actor.record())==original and world==original_world
    return dict(complete=True,arm=arm,continuation_scope=continuation_scope,owner=original,selected_action=chosen_key,
        reference_action=decisions[0]['action_id'],reference_admissible=admissible,
        normalized_regret=regret,
        scores_by_root={f.roots[name][0]:float(scores.scores[0,i]) for i,name in enumerate(names)},
        incompatible_reason=None if admissible else 'outside settled persona tier or purpose/progress corridor',
        actual_continuation_steps=sum(len(t) for t in trajectories.values()),
        future_controller='same arm, H32, seeds701/702/703, actual feedback and replanning every step',
        reference_scope='all first actions, deterministic complete continuations of this controller; not global optimality',
        decisions_by_root={t[0]['action']:[r['action'] for r in t] for t in trajectories.values()},
        finals_by_root={t[0]['action']:t[-1]['after'] for t in trajectories.values()},
        traces=trajectories)
