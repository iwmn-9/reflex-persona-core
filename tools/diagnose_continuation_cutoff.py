"""Fixed factual owners, shared continuations, different truncation boundaries.

Neither true hidden laws nor factual future draws enter the controller. All
cases are declared before execution. This is diagnosis, not an adoption test.
"""
import argparse,copy,json,sys
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import digest
from reflex.decision_loop import DecisionLoop
from reflex.model_observer import resolve_feedback
from reflex.intertemporal import Branch
from reflex.purpose_plan import goal_forecast
from tools.continuation_portfolio import portfolio_planner
from tools.owned_checkpoint import encoded,sha
from tools.run_continuation_portfolio import game_for
from tools.run_owned_value import cumulative_basis


def probe(args):
    root,case=args
    data=(root/case['input']).read_bytes();assert sha(data)==case['sha256']
    saved=json.loads(data)['payload'];spec=saved['contract']['configuration']['spec'];g=game_for(spec)
    actor=g.actor(g.context(g.start()),learning='segmented')
    for t in saved['trace'][:case['tick']]:
        assert digest(actor.record())==t['before_owner']
        c=g.context(t['before'],actor.state)
        d=DecisionLoop.decide_batch([(actor,g.request(c))],False,_counterfactual_roots=(t['action'],))[0]
        after,flow=g.transition(t['before'],t['action'],t['response'])
        resolve_feedback(actor,d['ticket'],g.feedback(t['before'],after,t['action'],c,flow))
        assert digest(actor.record())==t['owner']
    old=saved['trace'][case['tick']];w=old['before'];source=digest(actor.record());assert source==old['before_owner']
    request=g.request(g.context(w,actor.state));cap={}
    planner=portfolio_planner(actor,request,w,observe=g.context,events=g.events,terminal=g.terminal,
        assess=lambda w:g.goal(w).record(),request_for=g.request,feedback=g.feedback,model_id=g.model_id,
        target=g.target,cumulative_preference=cumulative_basis(g,True),horizon=32,
        capture_prefixes=True,captured=cap,**spec['controller'])
    DecisionLoop.decide_batch([(copy.deepcopy(actor),request)],False,planner)
    rows=[]
    for h in (8,12,15,16,17,20,24,32):
        paths={};audits={};mapping={}
        for name,branches in cap['paths'].items():
            paths[name]=[];audits[name]=[]
            for branch,audit in zip(branches,cap['modeled'][name]):
                prefix=audit['prefixes'][min(h,len(audit['prefixes']))-1]
                paths[name].append(Branch(branch.probability,branch.effects[:h],branch.confidence[:h]))
                audits[name].append(dict(actions=audit['actions'][:h],assessment=prefix['assessment'],
                    terminal=prefix['terminal'],purpose_mean_estimate=prefix['assessment']['value']+prefix['correction']))
            mapping[name]=audits[name][0]['actions'][0]
        captured={}
        def evaluate(contexts,decisions):
            f=goal_forecast(contexts[0],paths,audits,horizon=h,unit='owner-decisions',target=g.target,
                policy=actor.policy,plan_roots=mapping,settlement_weight='absolute',objective_origin=g.goal(w),
                opportunity_regret=True,cumulative_preference=cumulative_basis(g,True),outcome_limit=256,
                coherent_cumulative_mean=True)
            best={}
            for k,roots in f.roots.items():best[roots[0]]=max(best.get(roots[0],-1.),f.purpose[k])
            f=replace(f,max_regret=f.audit['purpose_band']['fraction']*(max(best.values())-min(best.values())))
            captured['forecast']=f
            return f
        result=DecisionLoop.decide_batch([(copy.deepcopy(actor),request)],False,evaluate)[0]
        f=captured['forecast']
        if h==16:
            assert result['decision']['action_id']==old['action']
            assert abs(result['deliberation']['selected_purpose']-old['selected_purpose'])<1e-12
        rows.append(dict(horizon=h,action=result['decision']['action_id'],purpose=f.purpose,roots=f.roots,
            selected_purpose=result['deliberation']['selected_purpose'],deliberation=result['deliberation']))
    assert digest(actor.record())==source
    out=dict(case=case,original_choice_reproduced=True,factual_owner=source,world=w,rows=rows,
        modeled=cap['modeled'],adoption=False,limit=__doc__)
    (root/(case['id']+'.json')).write_bytes(encoded(out))
    print(json.dumps(dict(case=case['id'],choices=[(r['horizon'],r['action']) for r in rows])),flush=True)
    return dict(case=case['id'],sha256=sha(encoded(out)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--frozen',action='store_true')
    p.add_argument('--workers',type=int,default=3);args=p.parse_args();root=Path(args.root).resolve()
    if not args.frozen:
        from tools.freeze_experiment import dispatch
        return dispatch(root,Path(__file__).name,['--workers',str(args.workers)])
    plan=json.loads((root/'plan.json').read_bytes())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(probe,[(root,c) for c in plan['cases']]))
    (root/'summary.json').write_bytes(encoded(dict(complete=True,rows=rows,adoption=False)))

if __name__=='__main__':main()
