"""Known-state ablation: identical owned paths, two purpose mean estimators.

Does not write to a running study or replace its result. Both alternatives use
the same simulated preferences, risk, learning, horizon and model seeds.
"""
import argparse
import copy
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import digest
from reflex.decision_loop import DecisionLoop
from reflex.model_observer import resolve_feedback
from reflex.purpose_plan import Goal,goal_forecast
from tools.continuation_portfolio import portfolio_planner
from tools.owned_checkpoint import encoded,sha
from tools.run_continuation_portfolio import game_for
from tools.run_owned_value import cumulative_basis


def probe(args):
    root,profile,tick,recovery=args
    saved=json.loads((root/f'input-{profile}.json').read_bytes())['payload']
    spec=saved['contract']['configuration']['spec'];g=game_for(spec)
    actor=g.actor(g.context(g.start()),learning='segmented')
    for entry in saved['trace'][:tick]:
        assert digest(actor.record())==entry['before_owner']
        c=g.context(entry['before'],actor.state)
        r=DecisionLoop.decide_batch([(actor,g.request(c))],False,_counterfactual_roots=(entry['action'],))[0]
        after,flow=g.transition(entry['before'],entry['action'],entry['response'])
        resolve_feedback(actor,r['ticket'],g.feedback(entry['before'],after,entry['action'],c,flow))
        assert digest(actor.record())==entry['owner']
    recorded=saved['trace'][tick];w=recorded['before'];source=digest(actor.record())
    assert source==recorded['before_owner']
    request=g.request(g.context(w,actor.state));cap={}
    planner=portfolio_planner(actor,request,w,observe=g.context,events=g.events,
        terminal=g.terminal,assess=lambda w:g.goal(w).record(),request_for=g.request,
        feedback=g.feedback,model_id=g.model_id,target=g.target,
        horizon=16,seeds=(701,702,703),cumulative_preference=cumulative_basis(g,True),
        captured=cap,purpose_estimator='conditional',tail_recovery=recovery)
    conditional=DecisionLoop.decide_batch([(copy.deepcopy(actor),request)],False,planner)[0]
    modeled=copy.deepcopy(cap['modeled'])
    for rows in modeled.values():
        for row in rows:row.pop('purpose_mean_estimate')
    def sampled_planner(contexts,decisions):
        c=contexts[0];mapping={k:rows[0]['actions'][0] for k,rows in modeled.items()}
        f=goal_forecast(c,cap['paths'],modeled,horizon=16,unit='owner-decisions',target=g.target,
            policy=actor.policy,plan_roots=mapping,settlement_weight='absolute',
            objective_origin=Goal(**g.goal(w).record()),opportunity_regret=True,
            cumulative_preference=cumulative_basis(g,True),outcome_limit=256)
        best={}
        for k,roots in f.roots.items():best[roots[0]]=max(best.get(roots[0],-1.),f.purpose[k])
        f=replace(f,max_regret=f.audit['purpose_band']['fraction']*(max(best.values())-min(best.values())))
        cap['sampled_forecast']=f
        return f
    sampled=DecisionLoop.decide_batch([(copy.deepcopy(actor),request)],False,sampled_planner)[0]
    if not recovery:
        assert sampled['decision']['action_id']==recorded['action']
        assert abs(sampled['deliberation']['selected_purpose']-recorded['selected_purpose'])<1e-12
    assert digest(actor.record())==source
    rows=[]
    for name,result,f in (('sampled',sampled,cap['sampled_forecast']),('conditional',conditional,cap['forecast'])):
        rows.append(dict(estimator=name,action=result['decision']['action_id'],decision=result['deliberation'],
            roots=f.roots,purpose=f.purpose,max_regret=f.max_regret))
    result=dict(profile=profile,tick=tick,world=w,factual_owner=source,
        original_choice_reproduced=None if recovery else True,same_paths=True,rows=rows,
        tail_recovery=recovery,recovery_calls=sum(row['recovery_calls'] for paths in cap['modeled'].values() for row in paths),
        limit='Known-state diagnosis only. The conditional estimator retains model bias and selection noise; no long-play adoption.')
    (root/f'probe-{profile}-{tick}.json').write_bytes(encoded(result))
    print(json.dumps(dict(profile=profile,tick=tick,actions=[r['action'] for r in rows])),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--frozen',action='store_true')
    p.add_argument('--recovery',type=int,default=0,choices=(0,2))
    args=p.parse_args();root=Path(args.root).resolve()
    if not args.frozen:
        from tools.freeze_experiment import dispatch
        root.mkdir(parents=True,exist_ok=True)
        for profile in (0,2):
            data=Path(f'results/continuation_portfolio_revision/worlds/combat-{profile}-56101-outside/candidate/checkpoint.json').read_bytes()
            dest=root/f'input-{profile}.json'
            if not dest.exists():dest.write_bytes(data)
        return dispatch(root,Path(__file__).name,['--recovery',str(args.recovery)])
    plan=dict(cases=[[p,t] for p in (0,2) for t in (78,84)],
        inputs={str(p):sha((root/f'input-{p}.json').read_bytes()) for p in (0,2)},
        sources=sha((root/'source_snapshot.json').read_bytes()),tail_recovery=args.recovery,
        purpose='Observed repeated preparation at full capacity; compare estimators, not game-specific fixes.')
    (root/'plan.json').write_bytes(encoded(plan))
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(probe,[(root,p,t,args.recovery) for p,t in plan['cases']]))
    (root/'summary.json').write_bytes(encoded(dict(complete=True,rows=rows,adoption=False)))


if __name__=='__main__':main()
