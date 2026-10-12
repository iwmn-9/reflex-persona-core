"""Registered follow-up to Goofspiel losses: terminal window vs future chooser.

Uses the SAME confirmation positions, not a fresh holdout. Shared policies and
coefficients stay fixed. The reference H3 results remain in the main cohort.
"""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import digest
from reflex.decision_loop import DecisionLoop
from reflex.laboratory import profiles
from tools.run_closed_loop import setup,decision
from tools.run_foundation_audit import prepare
from tools.closed_loop_models import controller
from tools.run_contingent_coverage import cases

def episode(job):
    case,profile,seed,arm=job;start=time.perf_counter();b,p,_=setup(case,profile,'sampled_tree',4)
    b,w,prefix=prepare(case,profile,seed);actor=DecisionLoop(b.observe(b.project(w)),policy=p)
    ctrl=controller(b,horizon=4,future_choices='reflex' if arm=='sampled_reflex4' else 'planned',nodes=4096,outcomes=4096)
    trace=[]
    while not b.terminal(w):
        before=b.record(w);result,cap,remaining=decision(b,p,ctrl,w,actor,'sampled_tree',4,len(trace))
        key=result['decision']['action_id'];w,row=b.actual(w,key,seed)
        forecast=cap['node']['forecast'] if cap.get('node') else None
        trace.append(dict(before=before,after=b.record(w),root=key,flow=row,goal=b.goal(w).record(),remaining=remaining,
            owner=digest(actor.record()),tree=cap.get('audit'),decision=result['decision'],
            purpose=None if forecast is None else {forecast.roots[k][0]:v for k,v in forecast.purpose.items()},
            settled_branches=None if forecast is None else sum(g['status']!='running' for gs in forecast.audit['endpoint'].values() for g in gs)))
    check=b.start()
    for t in prefix+trace:
        assert b.record(check)==t['before'];check,row=b.actual(check,t['root'],seed)
        assert b.record(check)==t['after'] and row==t['flow']
    assert b.terminal(check)
    return dict(job=job,case=case,profile=profile['id'],seed=seed,arm=arm,goal=b.goal(w).record(),prefix=prefix,
        trace=trace,actions=[t['root'] for t in trace],replay=True,elapsed=time.perf_counter()-start)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--workers',type=int,default=2);p.add_argument('--frozen',action='store_true')
    a=p.parse_args();root=Path(a.root).resolve()
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        return dispatch(root,Path(__file__).name,['--workers',str(a.workers)])
    if (root/'preregister.json').exists():raise FileExistsError('fresh probe required')
    selected=[c for c in cases('confirmation') if c['game'] in ('auction_contest','hagetaka','gambling','empire_race','team_combat','team_projects','no_thanks','goofspiel')]
    index=next(i for i,c in enumerate(selected) if c['game']=='goofspiel');case=selected[index]
    jobs=[(dict(copy.deepcopy(case),opening_steps=3,opening_seed=36100+index+seed-36100),profile,seed,arm)
        for seed in range(36100,36104) for profile in profiles() for arm in ('sampled_reflex4','sampled_tree4')]
    pre=dict(format='closed-loop-frontier-probe-v1',stage='frontier',jobs=jobs,
        timing='after observing confirmation Goofspiel losses; same positions, not fresh holdout',
        hypothesis='terminal-length window changes the H3 cutoff failures; isolate future reflex from future planning within the same sampled tree',
        controls=['all four confirmation seeds and personas','same coefficients and two model seeds','full execution and replay'],
        source_snapshot_sha256=hashlib.sha256((root/'source_snapshot.json').read_bytes()).hexdigest())
    (root/'preregister.json').write_text(json.dumps(pre,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    rows=[];errors=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool,(root/'trajectories.jsonl').open('x',encoding='utf-8') as stream:
        pending={pool.submit(episode,j):i for i,j in enumerate(jobs)}
        for f in as_completed(pending):
            i=pending[f]
            try:r=f.result()
            except Exception as error:errors.append(dict(index=i,job=jobs[i],error=repr(error)));print('ERROR',i,repr(error),flush=True);continue
            r['index']=i;stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush();rows.append({k:v for k,v in r.items() if k!='trace'})
            print(len(rows)+len(errors),'/',len(jobs),r['profile'],r['seed'],r['arm'],r['goal']['value'],round(r['elapsed'],2),flush=True)
    (root/'evaluation.json').write_text(json.dumps(dict(planned=len(jobs),complete=len(rows),errors=errors,runs=rows,
        preregister_sha256=digest(pre),trajectories_sha256=hashlib.sha256((root/'trajectories.jsonl').read_bytes()).hexdigest()),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if errors:raise RuntimeError('probe failures retained')
if __name__=='__main__':main()
