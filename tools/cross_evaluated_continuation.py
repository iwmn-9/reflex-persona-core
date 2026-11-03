"""Research: discover continuations and evaluate them on disjoint model samples.

Retain the ordinary bank for every root, plus one discovered continuation per
root. Freeze its public stopping curve before evaluation. The final personality
tier and purpose corridor use evaluation paths only. This is not Double Q-learning
or a safety guarantee; model bias, shortlist omissions and evaluation noise remain.
"""
import copy
from dataclasses import replace

from reflex.core import digest
from reflex.deliberation import select
from reflex._continuation_portfolio import portfolio_planner
from reflex._horizon_consensus import prefix_forecast
from reflex.purpose_plan import Goal
from tools.bounded_expansion import bounded_expansion_planner


def shortlist(context, forecast, policies, modeled, policy):
    """Use the unchanged owner selector; no score learned from game labels."""
    names={name for name,p in policies.items() if p['kind']=='reflex' or
        ('termination' not in p and p['preferred']==modeled[name][0]['actions'][0])}
    chosen={};references={}
    roots=sorted({row[0]['actions'][0] for row in modeled.values()})
    for root in roots:
        decisions,audit=select([context],forecast,policy,root_allowed=[{root}])
        if decisions is None:
            continue  # The full discovery personality tier may exclude a root.
        name=forecast.audit['plans'][audit['selected_plan']]['proposal']
        names.add(name);chosen[root]=name
        if name.endswith('-until-gain'):
            names.add(name.removesuffix('-until-gain'))
            references[name.split('-retry')[0]+'-reflex']=policies[name]['reference_goal_curve']
    return names,references,chosen


def cross_evaluated_planner(actor,request,world,*,evaluation_seeds=(1701,1702,1703),captured=None,**options):
    discovery_seeds=options.get('seeds',(701,702,703))
    if (not isinstance(evaluation_seeds,tuple) or not evaluation_seeds
            or set(evaluation_seeds)&set(discovery_seeds)):
        raise ValueError('disjoint discovery and evaluation model seeds required')
    source=digest(actor.record());discovery={}
    discover=bounded_expansion_planner(actor,request,world,captured=discovery,**options)

    def planner(contexts,decisions):
        first=discover(contexts,decisions)
        policies=first.audit['continuation_portfolio']['policies']
        names,references,chosen=shortlist(contexts[0],first,policies,discovery['modeled'],actor.policy)
        evaluated={}
        evaluate=portfolio_planner(actor,request,copy.deepcopy(world),captured=evaluated,
            capture_prefixes=True,policy_names=names,fixed_references=references,
            **dict(options,seeds=evaluation_seeds))
        evaluate(contexts,decisions)
        horizon=min(options['horizon'],max(len(row['actions']) for rows in evaluated['modeled'].values() for row in rows))
        common=dict(actor=actor,target=options['target'],origin=Goal(**options['assess'](world)),
            cumulative_preference=options['cumulative_preference'])
        expanded,paths,rows=prefix_forecast(contexts[0],evaluated['paths'],evaluated['modeled'],horizon,**common)
        ordinary={name for name,p in policies.items() if p['kind']=='reflex' or
            ('termination' not in p and p['preferred']==discovery['modeled'][name][0]['actions'][0])}
        base,_,_=prefix_forecast(contexts[0],{n:evaluated['paths'][n] for n in ordinary},
            {n:evaluated['modeled'][n] for n in ordinary},horizon,**common)
        audit=copy.deepcopy(expanded.audit)
        audit['continuation_portfolio']=evaluated['forecast'].audit['continuation_portfolio']
        audit['cross_evaluation']=dict(discovery_seeds=list(discovery_seeds),evaluation_seeds=list(evaluation_seeds),
            discovered=chosen,ordinary=sorted(ordinary),retained=sorted(names),fixed_references=references,
            source_owner=source,ordinary_regret=base.max_regret,expanded_regret=expanded.max_regret,
            semantics=__doc__)
        result=replace(expanded,max_regret=min(base.max_regret,expanded.max_regret),audit=audit)
        if digest(actor.record())!=source:
            raise ValueError('model sampling changed the factual owner')
        if captured is not None:
            captured.update(forecast=result,paths=paths,modeled=rows,discovery=discovery,
                evaluated=evaluated,ordinary_forecast=base,
                simulated_steps=discovery['simulated_steps']+evaluated['simulated_steps'])
        return result
    return planner
