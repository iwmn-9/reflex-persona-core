"""Preregistered planning mechanism comparison; no game-specific policy tuning."""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import Policy,character_key,digest
from reflex.decision_loop import DecisionLoop
from reflex.closed_loop import DecisionWindow
from reflex.contingent_games import make_bridge,Frame
from reflex.integrated_trials import request_for
from reflex.flow_rollout import rollout
from reflex.purpose_plan import goal_forecast
from reflex.laboratory import profiles
from tools.closed_loop_models import controller
from tools.run_foundation_audit import prepare,graph_values,GAMES
from tools.run_contingent_coverage import cases


def baseline(c,w,b,p,h,seeds):
    paths,audit=rollout(c,w,observe=b.observe,advance=b.modeled_advance,terminal=b.terminal,
        assess=lambda s:b.goal(s).record(),horizon=h,seeds=seeds,policy=p)
    return goal_forecast(c,paths,audit,horizon=h,unit=b.unit,target=b.target,policy=p,settlement_weight='absolute')


def setup(case,profile,arm,horizon):
    b=make_bridge(profile,case);c=b.observe(b.project(b.start()))
    p=Policy(principle_priority='personal',commitments={character_key(c):.5})
    ctrl=None if arm.startswith('baseline') else controller(b,horizon=horizon,exact=arm.startswith('exact'),
        future_choices='reflex' if arm=='exact_reflex' else 'planned',nodes=4096,outcomes=4096)
    return b,p,ctrl


def decision(b,p,ctrl,w,actor,arm,horizon,used):
    c=b.observe(b.project(w),actor.state);req=request_for(c,b.case['game']);captured={}
    remaining=horizon if arm in ('baseline_rolling','baseline_roll2') else horizon-used%horizon
    if arm=='exact_full':remaining=horizon-w.turns
    planner=(lambda cs,ds:baseline(cs[0],b.project(w),b,p,remaining,(401,402,403,404) if arm=='baseline_rolling' else (401,402))) if ctrl is None else ctrl.planner(actor,req,b.project(w),remaining=remaining,captured=captured)
    result=DecisionLoop.decide_batch([(actor,req)],False,planner)[0]
    actor.abandon(result['ticket'])
    return result,captured,remaining


def exact_policy(job):
    case,profile,arm=job;start=time.perf_counter();h=case['length'] if arm=='exact_full' else 4
    b,p,ctrl=setup(case,profile,arm,h);owner=DecisionLoop(b.observe(b.start()),policy=p);memo={};trace=[]
    optimum,_=graph_values(b.rules)
    def visit(w,actor,used):
        if b.terminal(w):return float(w.position[1] in b.rules.winning)
        stamp=digest([b.record(w),actor.record(),used]);
        if stamp in memo:return memo[stamp]
        own=copy.deepcopy(actor);result,cap,remaining=decision(b,p,ctrl,w,own,arm,h,used);key=result['decision']['action_id'];value=0.
        predicted=[]
        for node in b.rules.edges[*w.position,key]:
            after=Frame((w.position[0]+1,node),w.turns+1);value+=visit(after,own,used+1)/2
        if cap.get('node'):
            predicted=[dict(public=x['public'],owner=x['owner'],child=x['child']) for x in cap['node']['children'][key]]
        trace.append(dict(key=stamp,world=b.record(w),owner_before=digest(actor.record()),owner_after=digest(own.record()),
            used=used,remaining=remaining,action=key,value=value,tree=cap.get('audit'),modeled_children=predicted))
        memo[stamp]=value;return value
    success=visit(b.start(),owner,0)
    return dict(job=job,case=case,profile=profile['id'],arm=arm,success_probability=success,optimal_probability=optimum[0,0],
        states=len(memo),trace=trace,elapsed=time.perf_counter()-start)


def episode(job):
    case,profile,seed,arm,h=job;start=time.perf_counter();b,p,ctrl=setup(case,profile,arm,h)
    b,w,prefix=prepare(case,profile,seed);c=b.observe(b.project(w));actor=DecisionLoop(c,policy=p);trace=[]
    # The controller must belong to this bridge's public model/caches.
    if ctrl is not None:ctrl=controller(b,horizon=h,nodes=4096,outcomes=4096)
    while not b.terminal(w):
        before=b.record(w);result,cap,remaining=decision(b,p,ctrl,w,actor,arm,h,len(trace));key=result['decision']['action_id']
        w,row=b.actual(w,key,seed)
        trace.append(dict(before=before,after=b.record(w),root=key,flow=row,goal=b.goal(w).record(),remaining=remaining,
            owner=digest(actor.record()),tree=cap.get('audit'),decision=result['decision']))
        if len(trace)>1000:raise AssertionError('unsettled episode')
    check=b.start()
    for t in prefix+trace:
        assert b.record(check)==t['before'];check,flow=b.actual(check,t['root'],seed)
        assert b.record(check)==t['after'] and flow==t['flow']
    assert b.terminal(check) and b.goal(check).record()==b.goal(w).record()
    return dict(job=job,case=case,profile=profile['id'],seed=seed,arm=arm,horizon=h,goal=b.goal(w).record(),prefix=prefix,
        trace=trace,actions=[t['root'] for t in trace],replay=True,elapsed=time.perf_counter()-start)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--stage',choices=('pilot','graphs','graph_confirmation','development','confirmation'),required=True)
    ap.add_argument('--workers',type=int,default=4);ap.add_argument('--frozen',action='store_true');a=ap.parse_args();root=Path(a.root).resolve()
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        return dispatch(root,Path(__file__).name,['--stage',a.stage,'--workers',str(a.workers)])
    if (root/'preregister.json').exists():raise FileExistsError('fresh cohort required')
    if a.stage in ('graphs','graph_confirmation'):
        arms=('baseline_rolling','baseline_window','exact_reflex','sampled_tree','exact_tree','exact_full')
        if a.stage=='graph_confirmation':arms+=('baseline_roll2',)
        ids=range(25100,25108) if a.stage=='graphs' else range(25108,25116)
        lengths=(5,8) if a.stage=='graphs' else (6,10)
        jobs=[(dict(game='public_graph',graph=g,length=h),p,arm) for g in ids for h in lengths for p in profiles() for arm in arms]
        worker=exact_policy
    else:
        mode='confirmation' if a.stage=='confirmation' else 'development'
        selected=[c for c in cases(mode) if c['game'] in GAMES]
        if a.stage=='confirmation':selected=[dict(c,opening_steps=3,opening_seed=36100+i) for i,c in enumerate(selected)]
        jobs=[]
        seeds=range(35100,35102) if a.stage!='confirmation' else range(36100,36104)
        for c in selected:
            for seed in (list(seeds)[:1] if a.stage=='pilot' else seeds):
                cc=copy.deepcopy(c)
                if c['game'] in ('auction_contest','hagetaka','no_thanks'):cc['deal']=c.get('deal',0)+(seed-min(seeds))*23
                if 'opening_seed' in cc:cc['opening_seed']+=seed-min(seeds)
                for p in (profiles()[:1] if a.stage=='pilot' else profiles()):
                    for arm in ('baseline_window','sampled_tree'):
                        jobs.append((cc,p,seed,arm,3))
        worker=episode
    pre=dict(format='closed-loop-study-v1',stage=a.stage,jobs=jobs,
        hypotheses=['conditional future decisions improve on a reflex continuation model',
            'exact chance integration exposes smaller purpose differences than four terminal samples',
            'a fixed owner-decision window removes the artificial extension of the modeled planning deadline'],
        controls=['same fixed persona and rigidity .5','full distribution, no average-outcome loss suppression',
            'no actual future RNG input','every error and work-limit fallback retained','no tuning by game name'],
        limitations=['authored public models/opponents and nonterminal proxies','finite window differs from stationary optimal control',
            'exact graph probabilities are not observed human win rates','no automatic learning promotion'],
        source_snapshot_sha256=hashlib.sha256((root/'source_snapshot.json').read_bytes()).hexdigest())
    (root/'preregister.json').write_text(json.dumps(pre,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    rows=[];errors=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool,(root/'trajectories.jsonl').open('x',encoding='utf-8') as stream:
        pending={pool.submit(worker,j):i for i,j in enumerate(jobs)}
        for future in as_completed(pending):
            i=pending[future]
            try:r=future.result()
            except Exception as e:errors.append(dict(index=i,job=jobs[i],error=repr(e)));print('ERROR',i,repr(e),flush=True);continue
            r['index']=i;stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush();rows.append({k:v for k,v in r.items() if k!='trace'})
            print(len(rows)+len(errors),'/',len(jobs),r['case']['game'],r['profile'],r['arm'],r.get('success_probability',r.get('goal')),round(r['elapsed'],2),flush=True)
    evaluation=dict(planned=len(jobs),complete=len(rows),errors=errors,runs=rows,elapsed=time.perf_counter()-start,
        preregister_sha256=digest(pre),trajectories_sha256=hashlib.sha256((root/'trajectories.jsonl').read_bytes()).hexdigest())
    (root/'evaluation.json').write_text(json.dumps(evaluation,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if errors:raise RuntimeError('failed jobs retained')


if __name__=='__main__':main()
