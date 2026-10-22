"""Replay factual owners, then isolate proposal and valuation choices.

This is a fixed-history diagnosis, never a substitute for longitudinal results.
Every variant shares the already generated physical/learning paths. Removing
cumulative valuation here does not claim to model its different future owner.
"""
import argparse
import copy
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reflex.core import digest
from reflex.decision_loop import DecisionLoop
from reflex.model_observer import resolve_feedback
from reflex.purpose_plan import goal_forecast
from tools.continuation_portfolio import portfolio_planner
from tools.owned_checkpoint import encoded, sha
from tools.run_continuation_portfolio import game_for
from tools.run_owned_value import cumulative_basis


def probe(args):
    root, case = args
    saved = json.loads((root / case['input']).read_bytes())['payload']
    spec = saved['contract']['configuration']['spec']; g = game_for(spec)
    actor = g.actor(g.context(g.start()), learning='segmented')
    for entry in saved['trace'][:case['tick']]:
        assert digest(actor.record()) == entry['before_owner']
        c = g.context(entry['before'], actor.state)
        result = DecisionLoop.decide_batch([(actor, g.request(c))], False,
            _counterfactual_roots=(entry['action'],))[0]
        after, flow = g.transition(entry['before'], entry['action'], entry['response'])
        resolve_feedback(actor, result['ticket'], g.feedback(entry['before'], after, entry['action'], c, flow))
        assert digest(actor.record()) == entry['owner']
    entry = saved['trace'][case['tick']]; w = entry['before']; source = digest(actor.record())
    assert source == entry['before_owner']
    request = g.request(g.context(w, actor.state)); cap = {}
    planner = portfolio_planner(actor, request, w, observe=g.context, events=g.events,
        terminal=g.terminal, assess=lambda w:g.goal(w).record(), request_for=g.request,
        feedback=g.feedback, model_id=g.model_id, target=g.target,
        cumulative_preference=cumulative_basis(g, True), captured=cap,
        **spec.get('controller', {}))
    actual = DecisionLoop.decide_batch([(copy.deepcopy(actor), request)], False, planner)[0]
    assert actual['decision']['action_id'] == entry['action']
    assert abs(actual['deliberation']['selected_purpose'] - entry['selected_purpose']) < 1e-12
    variants = []
    for cumulative in (True, False):
        for retry,coherent in ((True,False),(False,False),(True,True)) if cumulative else ((True,False),(False,False)):
            paths = {k:v for k,v in cap['paths'].items() if retry or k.endswith('-reflex')}
            rows = {k:cap['modeled'][k] for k in paths}
            forecast = None
            def revalue(contexts, decisions):
                nonlocal forecast
                mapping = {k:v[0]['actions'][0] for k,v in rows.items()}
                f = goal_forecast(contexts[0], paths, rows, horizon=16, unit='owner-decisions',
                    target=g.target, policy=actor.policy, plan_roots=mapping,
                    settlement_weight='absolute', objective_origin=g.goal(w), opportunity_regret=True,
                    cumulative_preference=cumulative_basis(g, cumulative), outcome_limit=256,
                    coherent_cumulative_mean=coherent)
                best = {}
                for k, roots in f.roots.items():best[roots[0]] = max(best.get(roots[0], -1.), f.purpose[k])
                forecast = replace(f, max_regret=f.audit['purpose_band']['fraction']*(max(best.values())-min(best.values())))
                return forecast
            result = DecisionLoop.decide_batch([(copy.deepcopy(actor), request)], False, revalue)[0]
            variants.append(dict(cumulative=cumulative, retry=retry, coherent=coherent, action=result['decision']['action_id'],
                deliberation=result['deliberation'], roots=forecast.roots, purpose=forecast.purpose))
    assert digest(actor.record()) == source
    out = dict(case=case, source_owner=source, world=w, original_choice_reproduced=True,
        shared_paths=True, variants=variants, modeled=cap['modeled'],
        limit='Known factual state, fixed paths; not a full future-policy ablation or adoption result.')
    (root / (case['id']+'.json')).write_bytes(encoded(out))
    print(json.dumps(dict(case=case['id'], choices=[v['action'] for v in variants])), flush=True)
    return dict(case=case['id'], sha256=sha(encoded(out)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--frozen',action='store_true')
    p.add_argument('--workers',type=int,default=4);args=p.parse_args();root=Path(args.root).resolve()
    if not args.frozen:
        from tools.freeze_experiment import dispatch
        return dispatch(root,Path(__file__).name,['--workers',str(args.workers)])
    plan=json.loads((root/'plan.json').read_bytes())
    for case in plan['cases']:assert sha((root/case['input']).read_bytes()) == case['sha256']
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(probe,[(root,c) for c in plan['cases']]))
    (root/'summary.json').write_bytes(encoded(dict(complete=True,rows=rows,adoption=False)))


if __name__=='__main__':main()
