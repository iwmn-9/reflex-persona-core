"""Exact whole-policy diagnostic: isolate cutoff, sample and continuation error."""
import argparse,copy,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import digest
from reflex.decision_loop import DecisionLoop
from reflex.frontier_value import EndpointEstimate,PolicyTail
from reflex.contingent_games import make_bridge
from reflex.integrated_trials import request_for
from reflex.laboratory import profiles
from reflex.closed_loop import reveal
from tools.closed_loop_models import controller
from tools.run_frontier_value import owned,pool_run
from tools.run_foundation_audit import graph_values


class EnumeratedReflexTail:
    """Diagnostic reference for finite public graphs only, not a fitted oracle."""
    def __init__(self,b,ctrl):
        self.b=b;self.ctrl=ctrl;self.cache={};self.identity=digest(['exact-graph-owned-reflex-tail',b.rules.graph,b.rules.length])

    def evaluate(self,w,actor):
        if self.b.terminal(w):return EndpointEstimate(((1.,self.b.goal(w).value),),self.identity,1.)
        key=digest([self.b.record(w),actor.record()])
        if key in self.cache:return self.cache[key]
        own=copy.deepcopy(actor);c=self.b.observe(w,own.state);req=request_for(c,'public_graph')
        effective=own._prepare(req)['effective'];d=own.decide(req,False);move=d['decision']['action_id'];values={}
        for t in self.ctrl.transitions(w,move):
            branch=copy.deepcopy(own);reveal(branch,d['ticket'],self.ctrl.feedback(w,t.state,move,effective,t.flow))
            estimate=self.evaluate(t.state,branch)
            for p,v in estimate.outcomes:values[v]=values.get(v,0.)+t.probability*p
        result=EndpointEstimate(tuple((p,v) for v,p in sorted(values.items())),self.identity,1.);self.cache[key]=result;return result


def evaluate(job):
    graph,length,profile,arm=job;start=time.perf_counter();case=dict(game='public_graph',graph=graph,length=length);b=make_bridge(profile,case)
    actor=owned(b,b.start());ctrl=controller(b,horizon=length if arm=='full' else 2,exact=True,nodes=4096,outcomes=4096)
    if arm=='exact_tail':ctrl.frontier=EnumeratedReflexTail(b,ctrl)
    if arm.startswith('sampled_tail'):
        samples=int(arm.removeprefix('sampled_tail') or '2')
        ctrl.frontier=PolicyTail(observe=b.observe,advance=b.modeled_advance,terminal=b.terminal,assess=lambda w:b.goal(w).record(),
            request_for=ctrl.request_for,feedback=ctrl.feedback,model_id=ctrl.model_id,target=b.target,seeds=tuple(range(701,701+samples)),steps=16,record=b.record)
    cache={};trace=[]
    def visit(w,owner):
        if b.terminal(w):return b.goal(w).value
        stamp=digest([b.record(w),owner.record()])
        if stamp in cache:return cache[stamp]
        own=copy.deepcopy(owner);c=b.observe(w,own.state);req=request_for(c,'public_graph');cap={}
        d=DecisionLoop.decide_batch([(own,req)],False,ctrl.planner(own,req,w,remaining=length-w.turns if arm=='full' else min(2,length-w.turns),captured=cap))[0]
        assert cap['audit']['complete'];move=d['decision']['action_id'];value=0.
        for t in ctrl.transitions(w,move):
            branch=copy.deepcopy(own);reveal(branch,d['ticket'],ctrl.feedback(w,t.state,move,c,t.flow));value+=t.probability*visit(t.state,branch)
        prediction=sum(p*g['value'] for p,flows,actions,g,end in cap['node']['returns'])
        trace.append(dict(world=b.record(w),owner=digest(owner.record()),action=move,value=value,prediction=prediction,gap=prediction-value))
        cache[stamp]=value;return value
    value=visit(b.start(),actor);optimum,_=graph_values(b.rules)
    return dict(case=case,profile=profile['id'],arm=arm,value=value,optimal=2*optimum[0,0]-1,trace=trace,states=len(cache),elapsed=time.perf_counter()-start)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--workers',type=int,default=3);p.add_argument('--resume');p.add_argument('--confirmation',action='store_true');p.add_argument('--frozen',action='store_true');a=p.parse_args();root=Path(a.root).resolve()
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        return dispatch(root,Path(__file__).name,['--workers',str(a.workers)]+(['--confirmation'] if a.confirmation else [])+(['--resume',str(Path(a.resume).resolve())] if a.resume else []))
    arms=('proxy','sampled_tail','sampled_tail8','sampled_tail32','exact_tail','full') if a.confirmation else ('proxy','sampled_tail','exact_tail','full')
    first=53000 if a.confirmation else 52000
    jobs=[(g,length,profile,arm) for g in range(first,first+8) for length in (6,10) for profile in profiles() for arm in arms]
    pre=dict(version='frontier-exact-graph-diagnostic-v1',jobs=jobs,source_snapshot=digest(json.loads((root/'source_snapshot.json').read_text())),
        scope='exact public chance under actual repeated policy; no human/game win-rate claim',
        caveats=['exact reference tail differs from future replanning policy','two-step prefix preferences versus whole-horizon preferences in full arm','generated narrow public graphs only'])
    previous=[]
    if a.resume:
        prior=Path(a.resume);old=json.loads((prior/'preregister.json').read_text())
        for k in ('version','jobs','scope','caveats'):
            if digest(pre[k])!=digest(old[k]):raise ValueError('registered graph protocol changed')
        previous=[json.loads(s) for s in (prior/'trajectories.jsonl').read_text().splitlines()]
        for r in previous:
            graph,length,profile,arm=jobs[r['index']]
            assert (r['case']['graph'],r['case']['length'],r['profile'],r['arm'])==(graph,length,profile['id'],arm)
        pre['resumed']=dict(source=prior.name,registration=digest(old),source_snapshot=old['source_snapshot'],rows={str(r['index']):digest(r) for r in previous},reason='full-owner endpoint memoization; same rules and sampling protocol')
    (root/'preregister.json').write_text(json.dumps(pre,indent=2)+'\n');rows,errors,elapsed=pool_run(jobs,evaluate,root,a.workers,previous)
    report=dict(planned=len(jobs),complete=len(rows),errors=errors,elapsed=elapsed,groups=[])
    for length in (6,10):
        for arm in arms[1:]:
            pairs=[]
            for r in rows:
                if r['case']['length']!=length or r['arm']!=arm:continue
                old=next(x for x in rows if x['case']==r['case'] and x['profile']==r['profile'] and x['arm']=='proxy')
                pairs.append(dict(graph=r['case']['graph'],profile=r['profile'],before=old['value'],after=r['value'],delta=r['value']-old['value']))
            report['groups'].append(dict(length=length,arm=arm,pairs=pairs,better=sum(p['delta']>1e-12 for p in pairs),worse=sum(p['delta']< -1e-12 for p in pairs),same=sum(abs(p['delta'])<=1e-12 for p in pairs)))
    (root/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='groups'}),flush=True)
    if errors:raise RuntimeError('failed exact policy evaluations retained')


if __name__=='__main__':main()
