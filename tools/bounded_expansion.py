"""Research: added continuations must not enlarge the existing purpose-loss budget.

Keep the complete expanded personality tier and new means. Compare its purpose
corridor with the ordinary bank from the same factual owner, modeled paths,
and horizon. No coefficients, physical rules, or factual learners are changed.
This is not an adoption decision or a policy-improvement theorem.
"""
import copy
from dataclasses import replace

from reflex.core import digest
from reflex.purpose_plan import Goal
from reflex._continuation_portfolio import portfolio_planner
from reflex._horizon_consensus import prefix_forecast
from reflex.decision_loop import DecisionLoop
from tools.run_adverse_recovery import decide as unchanged_decide
from tools.run_owned_value import cumulative_basis


def bounded_expansion_planner(actor,request,world,*,captured=None,**options):
    if (options.get('continuation_scope')!='shared' or options.get('retry_until_gain') is not True
            or options.get('retry_gain_reference')!='expected' or options.get('include_retry') is not True
            or options.get('coherent_cumulative_mean') is not True or options.get('purpose_estimator')!='conditional'):
        raise ValueError('explicit complete shared expected-termination comparison required')
    if 'capture_prefixes' in options:raise ValueError('the comparison owns complete prefix capture')
    source=digest(actor.record());raw={}
    original=portfolio_planner(actor,request,world,captured=raw,capture_prefixes=True,**options)

    def planner(contexts,decisions):
        original(contexts,decisions)
        longest=max(len(row['actions']) for rows in raw['modeled'].values() for row in rows)
        horizon=min(options['horizon'],longest)
        common=dict(actor=actor,target=options['target'],origin=Goal(**options['assess'](world)),
                    cumulative_preference=options['cumulative_preference'])
        expanded,paths,rows=prefix_forecast(contexts[0],raw['paths'],raw['modeled'],horizon,**common)
        policies=raw['forecast'].audit['continuation_portfolio']['policies']
        names=[name for name,policy in policies.items()
               if policy['kind']=='reflex' or ('termination' not in policy
                   and policy['preferred']==raw['modeled'][name][0]['actions'][0])]
        ordinary,_,_=prefix_forecast(contexts[0],{n:raw['paths'][n] for n in names},
                                    {n:raw['modeled'][n] for n in names},horizon,**common)
        ceiling=min(expanded.max_regret,ordinary.max_regret)
        audit=copy.deepcopy(expanded.audit)
        audit['continuation_portfolio']=copy.deepcopy(raw['forecast'].audit['continuation_portfolio'])
        audit['expansion_regret_ceiling']=dict(ordinary=ordinary.max_regret,expanded=expanded.max_regret,
            applied=ceiling,source_owner=source,reference='ordinary and same-root retry; same modeled paths',
            semantics='Additional means cannot increase the ordinary-bank purpose-loss allowance. Personality tier is unchanged.')
        forecast=replace(expanded,max_regret=ceiling,audit=audit)
        if digest(actor.record())!=source:raise ValueError('comparison changed the factual owner')
        if captured is not None:
            captured.update(forecast=forecast,paths=paths,modeled=rows,simulated_steps=raw['simulated_steps'],
                ordinary_forecast=ordinary,unbounded_forecast=expanded)
        return forecast

    return planner


def decide(actor,game,world,arm,seeds=None,*,continuation_scope='root'):
    """Experiment-runner hook; all non-shared arms keep their original chooser."""
    if arm!='expected' or continuation_scope!='shared':
        return unchanged_decide(actor,game,world,arm,seeds=seeds,continuation_scope=continuation_scope)
    request=game.request(game.context(world,actor.state));captured={}
    planner=bounded_expansion_planner(actor,request,world,captured=captured,observe=game.context,
        events=game.events,terminal=game.terminal,assess=lambda w:game.goal(w).record(),
        request_for=game.request,feedback=game.feedback,model_id=game.model_id,target=game.target,
        cumulative_preference=cumulative_basis(game,True),seeds=(701,702,703) if seeds is None else seeds,
        recovery_cache=64,horizon=32,world_model='posterior',purpose_estimator='conditional',
        tail_recovery=2,include_retry=True,coherent_cumulative_mean=True,continuation_scope='shared',
        retry_until_gain=True,retry_gain_reference='expected')
    result=DecisionLoop.decide_batch([(actor,request)],False,planner)[0]
    if not result.get('deliberation',{}).get('adopted'):
        raise ValueError('incomplete bounded-expansion forecast; no fallback')
    return result,captured
