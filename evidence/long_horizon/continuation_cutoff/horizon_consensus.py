"""Research-only hedge against one arbitrary rollout truncation phase.

Each cutoff is evaluated by the unchanged owner, including values, downside
risk and progress masks. Vote over real first actions, not duplicated plans.
Return the longest full forecast supporting a plurality (longer support breaks
ties). This is a heuristic, not a policy-improvement or calibrated-risk theorem.
No game/action names, hidden laws or factual future outcomes are consulted.
"""
import copy
from collections import Counter
from dataclasses import replace
from reflex.core import digest
from reflex.decision_loop import DecisionLoop
from reflex.intertemporal import Branch
from reflex.purpose_plan import goal_forecast,Goal
from tools.continuation_portfolio import portfolio_planner


def project_paths(paths,modeled,horizon):
    projected={};audits={};mapping={}
    for name,branches in paths.items():
        projected[name]=[];audits[name]=[]
        for branch,row in zip(branches,modeled[name]):
            prefix=row['prefixes'][min(horizon,len(row['prefixes']))-1]
            projected[name].append(Branch(branch.probability,branch.effects[:horizon],branch.confidence[:horizon]))
            audits[name].append(dict(actions=row['actions'][:horizon],assessment=prefix['assessment'],
                terminal=prefix['terminal'],purpose_mean_estimate=prefix['assessment']['value']+prefix['correction']))
        mapping[name]=audits[name][0]['actions'][0]
        projected[name]=tuple(projected[name])
    return projected,audits,mapping


def prefix_forecast(context,paths,modeled,horizon,*,actor,target,origin,cumulative_preference):
    paths,rows,mapping=project_paths(paths,modeled,horizon)
    f=goal_forecast(context,paths,rows,horizon=horizon,unit='owner-decisions',target=target,
        policy=actor.policy,plan_roots=mapping,settlement_weight='absolute',objective_origin=origin,
        opportunity_regret=True,cumulative_preference=cumulative_preference,outcome_limit=256,
        coherent_cumulative_mean=True)
    best={}
    for name,roots in f.roots.items():best[roots[0]]=max(best.get(roots[0],-1.),f.purpose[name])
    spread=max(best.values())-min(best.values())
    audit=copy.deepcopy(f.audit)
    audit['purpose_band'].update(spread=spread,basis='best current-tier continuation per real root',best_by_root=best)
    return replace(f,max_regret=f.audit['purpose_band']['fraction']*spread,audit=audit),paths,rows


def select_support(ballots):
    if not ballots or any(b['action'] is None for b in ballots):
        raise ValueError('a complete real-root choice at every cutoff is required')
    counts=Counter(b['action'] for b in ballots)
    largest=max(counts.values())
    # Longest supporting forecast, without lexical action-name preference.
    selected=next(b for b in reversed(ballots) if counts[b['action']]==largest)
    return selected,dict(counts)


def consensus_planner(actor,request,world,*,captured=None,minimum_horizon=16,horizon=32,**options):
    if (type(minimum_horizon) is not int or type(horizon) is not int
            or not 2<=minimum_horizon<=horizon<=128):
        raise ValueError('explicit bounded consecutive cutoffs required')
    if options.get('purpose_estimator')!='conditional' or not options.get('coherent_cumulative_mean'):
        raise ValueError('same conditional mean for cumulative value and purpose required')
    source=digest(actor.record());cap={}
    original=portfolio_planner(actor,request,world,horizon=horizon,captured=cap,capture_prefixes=True,**options)
    def planner(contexts,decisions):
        original(contexts,decisions)
        # A genuinely settled future gets one vote, not copies of the same
        # padded terminal return for each nominal horizon.
        longest=max(len(row['actions']) for rows in cap['modeled'].values() for row in rows)
        cutoffs=sorted({min(h,longest) for h in range(minimum_horizon,horizon+1)})
        ballots=[];forecasts={}
        for h in cutoffs:
            f,paths,rows=prefix_forecast(contexts[0],cap['paths'],cap['modeled'],h,actor=actor,
                target=options['target'],origin=Goal(**options['assess'](world)),
                cumulative_preference=options['cumulative_preference'])
            trial=DecisionLoop.decide_batch([(copy.deepcopy(actor),request)],False,lambda cs,ds:f)[0]
            if not trial.get('deliberation',{}).get('adopted'):
                raise ValueError('cutoff has no complete owner-compatible plan; do not silently omit it')
            ballots.append(dict(horizon=h,action=trial['decision']['action_id'],
                purpose=trial['deliberation']['selected_purpose']))
            forecasts[h]=(f,paths,rows)
        selected,votes=select_support(ballots);f,paths,rows=forecasts[selected['horizon']]
        audit=copy.deepcopy(f.audit)
        audit['horizon_consensus']=dict(ballots=ballots,votes=votes,selected=selected,
            maximum_horizon=horizon,minimum_horizon=minimum_horizon,source_owner=source,
            semantics=__doc__)
        audit['continuation_portfolio']=cap['forecast'].audit['continuation_portfolio']
        f=replace(f,audit=audit)
        if digest(actor.record())!=source:raise ValueError('cutoff comparison mutated factual owner')
        if captured is not None:captured.update(forecast=f,paths=paths,modeled=rows,
            simulated_steps=cap['simulated_steps'],horizon_consensus=audit['horizon_consensus'])
        return f
    return planner
