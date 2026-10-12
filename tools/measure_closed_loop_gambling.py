"""Independent finite-bankroll transfer of the same generic tree controller."""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import Policy,character_key,digest
from reflex.decision_loop import DecisionLoop
from reflex.contingent_games import make_bridge,Frame
from reflex.gambling import Table,legal,offers
from reflex.laboratory import profiles
from reflex.integrated_trials import request_for
from tools.closed_loop_models import controller,gambling_kernel
from tools.run_closed_loop import baseline


def measure(job):
    regime,cash,left,profile,arm=job;start=time.perf_counter()
    b=make_bridge(profile,dict(game='gambling',scenario=regime));w=Frame((Table(cash,16-left,16),regime),16-left)
    kernel=gambling_kernel(b);c=b.observe(w);p=Policy(principle_priority='personal',commitments={character_key(c):.5})
    owner=DecisionLoop(c,policy=p)
    ctrl=None if arm=='baseline' else controller(b,horizon=left,exact=arm.startswith('exact'),future_choices='reflex' if arm=='exact_reflex' else 'planned',outcomes=256)
    @lru_cache(None)
    def optimal(world):
        if b.terminal(world):return b.goal(world).value
        return max(sum(t.probability*optimal(t.state) for t in kernel(world,key)) for key in legal(world.position[0],regime))
    memo={};trace=[]
    def visit(world,actor):
        if b.terminal(world):return b.goal(world).value
        stamp=digest([b.record(world),actor.record()])
        if stamp in memo:return memo[stamp]
        own=copy.deepcopy(actor);c=b.observe(world,own.state);req=request_for(c,'gambling');remaining=16-world.position[0].tick;cap={}
        planner=(lambda cs,ds:baseline(cs[0],world,b,p,remaining,(401,402))) if ctrl is None else ctrl.planner(own,req,world,remaining=remaining,captured=cap)
        result=DecisionLoop.decide_batch([(own,req)],False,planner)[0];key=result['decision']['action_id'];own.abandon(result['ticket'])
        value=sum(t.probability*visit(t.state,own) for t in kernel(world,key))
        predicted=None
        if cap.get('node'):
            predicted=sum(prob*g['value'] for prob,_,_,g,_ in cap['node']['returns'])
            if arm=='exact_tree':assert abs(value-predicted)<1e-10,'model/actual continuation drift under exact public odds'
        trace.append(dict(key=stamp,public=b.record(world),action=key,value=value,predicted_goal=predicted,tree=cap.get('audit')))
        memo[stamp]=value;return value
    value=visit(w,owner);best=optimal(w)
    assert value<=best+1e-10
    return dict(job=job,regime=regime,cash=cash,left=left,profile=profile['id'],arm=arm,goal=value,optimal=best,
        states=len(memo),trace=trace,elapsed=time.perf_counter()-start)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--workers',type=int,default=3);ap.add_argument('--frozen',action='store_true')
    a=ap.parse_args();root=Path(a.root).resolve()
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        return dispatch(root,Path(__file__).name,['--workers',str(a.workers)])
    if (root/'preregister.json').exists():raise FileExistsError('fresh cohort required')
    jobs=[(regime,cash,left,p,arm) for regime in ('opportunity','unfavorable') for cash in (6,12,17) for left in (2,4)
        for p in profiles() for arm in ('baseline','sampled_tree','exact_reflex','exact_tree')]
    pre=dict(format='closed-loop-gambling-transfer-v1',jobs=jobs,
        timing='follow-up after discovery graph results; no coefficient fitting',
        hypothesis='accurate public chance and recursively chosen continuation transfer beyond binary random graphs',
        controls=['existing bankroll rules and public current odds','same four personas/rigidity .5',
            'all possible actual chance branches integrated','exact_tree predicted goal must equal closed-loop actual expectation at every reached state'],
        limitations=['near-terminal positions with two/four rounds left','fixed current odds; no hidden future regime knowledge',
            'expected goal, not human win rate; correlated positions and personas'],
        source_snapshot_sha256=hashlib.sha256((root/'source_snapshot.json').read_bytes()).hexdigest())
    (root/'preregister.json').write_text(json.dumps(pre,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    rows=[];errors=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool,(root/'trajectories.jsonl').open('x',encoding='utf-8') as stream:
        pending={pool.submit(measure,j):i for i,j in enumerate(jobs)}
        for f in as_completed(pending):
            i=pending[f]
            try:r=f.result()
            except Exception as e:errors.append(dict(index=i,job=jobs[i],error=repr(e)));print('ERROR',i,repr(e),flush=True);continue
            r['index']=i;stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush();rows.append({k:v for k,v in r.items() if k!='trace'})
            print(len(rows)+len(errors),'/',len(jobs),r['regime'],r['cash'],r['left'],r['profile'],r['arm'],round(r['goal'],6),flush=True)
    (root/'evaluation.json').write_text(json.dumps(dict(planned=len(jobs),complete=len(rows),errors=errors,runs=rows,
        preregister_sha256=digest(pre),trajectories_sha256=hashlib.sha256((root/'trajectories.jsonl').read_bytes()).hexdigest()),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if errors:raise RuntimeError('incomplete transfer jobs retained')


if __name__=='__main__':main()
