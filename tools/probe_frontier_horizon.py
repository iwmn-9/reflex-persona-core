"""Isolate tail truncation on the already registered No Thanks changed roots.

This is a diagnostic followup, not independent actual-policy strength evidence.
Keep eight scenarios and the two-step tree fixed; vary only tail length 16/64.
"""
import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import digest
from reflex.decision_loop import DecisionLoop
from reflex.frontier_value import PolicyTail
from reflex.integrated_trials import request_for
from reflex.laboratory import profiles
from tools.closed_loop_models import controller
from tools.run_foundation_audit import prepare
from tools.run_frontier_value import owned,pool_run


def evaluate(job):
    ranking,prior,baseline,profile=job;case=ranking['case'];b,w,_=prepare(case,profile,ranking['seed']);actor=owned(b,w)
    for t in baseline[:ranking['divergence']]:
        c=b.observe(b.project(w),actor.state);d=DecisionLoop.decide_batch([(actor,request_for(c,case['game']))],False,_counterfactual_roots=(t['root'],))[0]
        actor.abandon(d['ticket']);w,_=b.actual(w,t['root'],ranking['seed'])
        assert digest(actor.record())==t['owner_after'] and digest(b.record(w))==digest(t['after'])
    outputs={};q={k:sum(v)/len(v) for k,v in ranking['samples'].items()}
    for steps in (16,64):
        ctrl=controller(b,horizon=2,nodes=4096,outcomes=4096)
        ctrl.frontier=PolicyTail(observe=b.observe,advance=b.modeled_advance,terminal=b.terminal,assess=lambda w:b.goal(w).record(),
            request_for=ctrl.request_for,feedback=ctrl.feedback,model_id=ctrl.model_id,target=b.target,seeds=tuple(range(701,709)),steps=steps,record=b.record)
        own=copy.deepcopy(actor);c=b.observe(b.project(w),own.state);req=request_for(c,case['game']);cap={}
        d=DecisionLoop.decide_batch([(own,req)],False,ctrl.planner(own,req,b.project(w),captured=cap))[0]
        assert cap['audit']['complete'];choice=d['decision']['action_id']
        fs=[f for stamp in cap['node']['dependencies'] for f in ctrl.cache[stamp].get('frontiers',[])]
        outputs[str(steps)]=dict(choice=choice,reference_gain=q[choice]-q[ranking['baseline']],
            frontiers=len(fs),settled_fraction=sum(f['settled_fraction'] for f in fs)/len(fs) if fs else 1.,
            prediction=[(p,g['value']) for p,_,_,g,_ in cap['node']['returns']])
    assert outputs['16']['choice']==prior['choices']['8'],'length-only comparator must reproduce prior width-eight root'
    return dict(case=case,profile=profile['id'],seed=ranking['seed'],at=ranking['divergence'],baseline=ranking['baseline'],outputs=outputs,reproduced_prior=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--budget',required=True);p.add_argument('--rankings',required=True);p.add_argument('--sources',nargs=2,required=True);p.add_argument('--workers',type=int,default=2);p.add_argument('--frozen',action='store_true');a=p.parse_args();root=Path(a.root).resolve()
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        return dispatch(root,Path(__file__).name,['--budget',str(Path(a.budget).resolve()),'--rankings',str(Path(a.rankings).resolve()),'--sources',*[str(Path(x).resolve()) for x in a.sources],'--workers',str(a.workers)])
    read=lambda p:[json.loads(s) for s in (Path(p)/'trajectories.jsonl').read_text().splitlines()]
    rankings=read(a.rankings);budgets=read(a.budget);sources=[r for source in a.sources for r in read(source)];pmap={p['id']:p for p in profiles()};jobs=[]
    for r in sorted(rankings,key=lambda r:r['index']):
        if r['case']['game']!='no_thanks' or r['arm']!='tail':continue
        match=lambda x:x['case']==r['case'] and x['profile']==r['profile'] and x['seed']==r['seed']
        prior=next(x for x in budgets if match(x));baseline=next(x for x in sources if match(x) and x['arm']=='proxy')
        jobs.append((r,prior,baseline['trace'],pmap[r['profile']]))
    pre=dict(version='frontier-tail-horizon-followup-v1',source_snapshot=digest(json.loads((root/'source_snapshot.json').read_text())),
        rankings=digest(rankings),budgets=digest(budgets),jobs=[dict(case=j[0]['case'],profile=j[0]['profile'],seed=j[0]['seed'],at=j[0]['divergence']) for j in jobs],
        settings=dict(tree_horizon=2,tree_seeds=[401,402],tail_seeds=list(range(701,709)),tail_steps=[16,64]),
        reason='No Thanks tails often remain unsettled at step 16; isolate cutoff from sample-count changes',
        limits=['same prior changed-root sample, not independent','reference action values use existing independent 64-scenario reflex continuations','not actual replanning-policy value'])
    (root/'preregister.json').write_text(json.dumps(pre,indent=2)+'\n');rows,errors,elapsed=pool_run(jobs,evaluate,root,a.workers)
    report=dict(planned=len(jobs),complete=len(rows),errors=errors,elapsed=elapsed,groups=[])
    if not errors:
        for steps in ('16','64'):
            gs=[r['outputs'][steps]['reference_gain'] for r in rows]
            report['groups'].append(dict(steps=int(steps),states=len(rows),mean_reference_gain=sum(gs)/len(gs),positive=sum(g>.15 for g in gs),negative=sum(g<-.15 for g in gs),near=sum(abs(g)<=.15 for g in gs),
                mean_state_settled_fraction=sum(r['outputs'][steps]['settled_fraction'] for r in rows)/len(rows),changed=sum(r['outputs'][steps]['choice']!=r['outputs']['16']['choice'] for r in rows)))
    (root/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    if errors:raise RuntimeError('length followup failed; retained')
if __name__=='__main__':main()
