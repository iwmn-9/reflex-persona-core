"""More endpoint samples at preregistered changed roots; no teacher feedback."""
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
    ranking,baseline,profile=job;case=ranking['case'];b,w,prefix=prepare(case,profile,ranking['seed']);actor=owned(b,w)
    for t in baseline[:ranking['divergence']]:
        c=b.observe(b.project(w),actor.state);d=DecisionLoop.decide_batch([(actor,request_for(c,case['game']))],False,_counterfactual_roots=(t['root'],))[0]
        actor.abandon(d['ticket']);w,flow=b.actual(w,t['root'],ranking['seed']);assert digest(actor.record())==t['owner_after'] and digest(b.record(w))==digest(t['after'])
    choices={};predictions={}
    for n in (2,8,32):
        ctrl=controller(b,horizon=2,nodes=4096,outcomes=4096)
        ctrl.frontier=PolicyTail(observe=b.observe,advance=b.modeled_advance,terminal=b.terminal,assess=lambda w:b.goal(w).record(),
            request_for=ctrl.request_for,feedback=ctrl.feedback,model_id=ctrl.model_id,target=b.target,seeds=tuple(range(701,701+n)),steps=16,record=b.record)
        own=copy.deepcopy(actor);c=b.observe(b.project(w),own.state);req=request_for(c,case['game']);cap={}
        d=DecisionLoop.decide_batch([(own,req)],False,ctrl.planner(own,req,b.project(w),captured=cap))[0]
        assert cap['audit']['complete'];choices[str(n)]=d['decision']['action_id'];predictions[str(n)]=[(p,g['value']) for p,_,_,g,_ in cap['node']['returns']]
    assert choices['2']==ranking['candidate'],'memoization/frozen decision bridge changed original root'
    samples=copy.deepcopy(ranking['samples'])
    for move in set(choices.values())-set(samples):
        samples[move]=[]
        for seed in range(61000,61064):
            own=copy.deepcopy(actor);world=b.project(w);first=True;steps=0
            while not b.terminal(world):
                c=b.observe(world,own.state);req=request_for(c,case['game'])
                d=DecisionLoop.decide_batch([(own,req)],False,_counterfactual_roots=(move,))[0] if first else own.decide(req,False)
                own.abandon(d['ticket']);world,_=b.modeled_advance(world,d['decision']['action_id'],seed);first=False;steps+=1
                if steps>256:raise AssertionError('unsettled independent reference')
            samples[move].append(b.goal(world).value)
    q={k:sum(v)/len(v) for k,v in samples.items()};return dict(case=case,profile=profile['id'],seed=ranking['seed'],at=ranking['divergence'],
        baseline=ranking['baseline'],choices=choices,predictions=predictions,samples=samples,
        gains={n:q[k]-q[ranking['baseline']] for n,k in choices.items()},source_decision_equal=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--rankings',required=True);p.add_argument('--sources',nargs=2,required=True);p.add_argument('--workers',type=int,default=3);p.add_argument('--frozen',action='store_true');a=p.parse_args();root=Path(a.root).resolve()
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        return dispatch(root,Path(__file__).name,['--rankings',str(Path(a.rankings).resolve()),'--sources',*[str(Path(s).resolve()) for s in a.sources],'--workers',str(a.workers)])
    prior=[json.loads(s) for s in (Path(a.rankings)/'trajectories.jsonl').read_text().splitlines()];allrows=[]
    for source in map(Path,a.sources):allrows.extend(json.loads(s) for s in (source/'trajectories.jsonl').read_text().splitlines())
    pmap={p['id']:p for p in profiles()};jobs=[]
    for r in sorted(prior,key=lambda r:r['index']):
        if r['arm']!='tail':continue
        b=next(x for x in allrows if x['case']==r['case'] and x['profile']==r['profile'] and x['seed']==r['seed'] and x['arm']=='proxy')
        jobs.append((r,b['trace'],pmap[r['profile']]))
    pre=dict(version='frontier-endpoint-budget-followup-v1',source_snapshot=digest(json.loads((root/'source_snapshot.json').read_text())),
        ranking_rows=digest(prior),jobs=[dict(case=j[0]['case'],profile=j[0]['profile'],seed=j[0]['seed'],at=j[0]['divergence']) for j in jobs],
        endpoint_samples=[2,8,32],tree_seeds=[401,402],endpoint_seeds='701..700+n',validation_seeds='61000..61063',
        scope='all tail changed-root pairs from prior ranking probe; followup not independent confirmation',
        limits=['reference-policy action values under public model','prefix chance budget remains two','no validation sample given to choosing controller'])
    (root/'preregister.json').write_text(json.dumps(pre,indent=2)+'\n');rows,errors,elapsed=pool_run(jobs,evaluate,root,a.workers)
    report=dict(planned=len(jobs),complete=len(rows),errors=errors,elapsed=elapsed,rows=[{k:v for k,v in r.items() if k not in ('samples','predictions')} for r in rows])
    (root/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(complete=len(rows),errors=errors)),flush=True)
    if errors:raise RuntimeError('sample-budget followup failures retained')


if __name__=='__main__':main()
