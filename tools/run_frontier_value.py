"""Frozen cross-rule comparison of cutoff proxy, factual memory and policy tails."""
import argparse,copy,json,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import Policy,character_key,digest,TRAITS,VALUES,NEEDS
from reflex.decision_loop import DecisionLoop
from reflex.frontier_value import TerminalMemory,LearnedFrontier,PolicyTail
from reflex.contingent_games import make_bridge
from reflex.integrated_trials import request_for
from reflex.laboratory import profiles
from reflex.planning import vector
from tools.closed_loop_models import controller
from tools.run_foundation_audit import prepare,GAMES
from tools.run_contingent_coverage import cases

SCHEMA='public-semantic-option-summary-v1'
REFERENCE='personal-rigidity-.5-owned-reflex-no-learning-v1'


def features(b,w,actor):
    """Same public semantic summary for every game; no IDs or actual RNG.

    This intentionally tests how far supplied generic effects suffice. It does
    NOT encode all game resources, topology, private cards or rival identities.
    Losing information here is a measured limitation, not universal rule learning.
    """
    c=b.observe(copy.deepcopy(w),copy.deepcopy(actor.state));rows=[]
    for a in c['actions']:
        if a['legal'] and not a['known_failure']:
            rows.append(sum((out['p']*np.asarray(vector(out)) for out in a['outcomes']),np.zeros(22)))
    x=np.asarray(rows)
    return [b.goal(w).value,float(w.turns),float(len(rows)),
        *[c['personality'][k] for k in TRAITS],*[c['values'][k] for k in VALUES],
        *[c['needs'][k]['deficit'] if c['needs'][k]['enabled'] and c['needs'][k]['supported'] else 0. for k in NEEDS],
        float(c['state']['age']),float(c['state']['mode_urgency']),
        *np.mean(x,axis=0).tolist(),*np.min(x,axis=0).tolist(),*np.max(x,axis=0).tolist()]


def owned(b,w):
    c=b.observe(b.project(w));p=Policy(principle_priority='personal',commitments={character_key(c):.5})
    return DecisionLoop(c,policy=p)


def crps(outcomes,value):
    # Integral discrete empirical CRPS; no uncertainty/confidence interpretation.
    return float(sum(p*abs(v-value) for p,v in outcomes)-.5*sum(p*q*abs(v-u) for p,v in outcomes for q,u in outcomes))


def train_episode(job):
    case,profile,seed=job;b,w,prefix=prepare(case,profile,seed);actor=owned(b,w);trace=[];xs=[]
    while not b.terminal(w):
        xs.append(features(b,b.project(w),actor));c=b.observe(b.project(w),actor.state);req=request_for(c,case['game'])
        result=actor.decide(req,False);key=result['decision']['action_id'];before=b.record(w)
        w,flow=b.actual(w,key,seed);actor.abandon(result['ticket'])
        trace.append(dict(before=before,root=key,after=b.record(w),flow=flow))
        if len(trace)>1000:raise AssertionError('unsettled training episode')
    verify(b,prefix+trace,seed,w)
    return dict(case=case,profile=profile['id'],seed=seed,features=xs,goal=b.goal(w).record(),trace=trace,prefix=prefix,
        episode=digest([profile['id'],prefix,trace]),replay=True)


def verify(b,trace,seed,final):
    w=b.start()
    for row in trace:
        assert b.record(w)==row['before'];w,flow=b.actual(w,row['root'],seed)
        assert b.record(w)==row['after'] and flow==row['flow']
    assert b.record(w)==b.record(final) and b.terminal(w)


def evaluate_episode(job):
    case,profile,seed,arm,model_record=job;start=time.perf_counter();b,w,prefix=prepare(case,profile,seed);actor=owned(b,w)
    ctrl=controller(b,horizon=2,nodes=4096,outcomes=4096)
    if arm=='memory':
        memory=TerminalMemory.from_record(model_record)
        # Deliberately permissive cross-configuration transfer for this study.
        # The later contract-bound probe tests the safer integration separately.
        ctrl.frontier=LearnedFrontier(memory,lambda world,own:features(b,world,own),schema_id=SCHEMA,reference=model_record['reference'])
    elif arm=='tail':
        ctrl.frontier=PolicyTail(observe=b.observe,advance=b.modeled_advance,terminal=b.terminal,assess=lambda world:b.goal(world).record(),
            request_for=ctrl.request_for,feedback=ctrl.feedback,model_id=ctrl.model_id,target=b.target,seeds=(701,702),steps=16,record=b.record)
    trace=[]
    while not b.terminal(w):
        c=b.observe(b.project(w),actor.state);req=request_for(c,case['game']);cap={};saved=digest(actor.record())
        result=DecisionLoop.decide_batch([(actor,req)],False,ctrl.planner(actor,req,b.project(w),captured=cap))[0]
        key=result['decision']['action_id'];before=b.record(w);pred=None;frontiers=[]
        if 'node' in cap:
            pred=[(p,g['value']) for p,flows,actions,g,settled in cap['node']['returns']]
            for stamp in cap['node']['dependencies']:frontiers.extend(ctrl.cache[stamp].get('frontiers',[]))
        actor.abandon(result['ticket']);w,flow=b.actual(w,key,seed)
        trace.append(dict(before=before,root=key,after=b.record(w),flow=flow,goal=b.goal(w).record(),
            prediction=pred,tree=cap.get('audit'),owner_before=saved,owner_after=digest(actor.record()),
            frontier_count=len(frontiers),supported=sum(f['supported'] for f in frontiers),
            settled_fraction_sum=sum(f['settled_fraction'] or 0. for f in frontiers),decision=result['decision']))
        if len(trace)>1000:raise AssertionError('unsettled evaluation episode')
    verify(b,prefix+trace,seed,w);g=b.goal(w).record()
    scores=[crps(t['prediction'],g['value']) for t in trace if t['prediction'] is not None]
    return dict(case=case,profile=profile['id'],seed=seed,arm=arm,goal=g,trace=trace,prefix=prefix,replay=True,
        actions=[t['root'] for t in trace],mean_terminal_crps=None if not scores else float(np.mean(scores)),elapsed=time.perf_counter()-start)


def pool_run(jobs,worker,root,workers,previous=()):
    rows=copy.deepcopy(list(previous));errors=[];start=time.perf_counter();done={r['index'] for r in rows}
    if len(done)!=len(rows) or not done<=set(range(len(jobs))):raise ValueError('unique registered completed indices required')
    with ProcessPoolExecutor(max_workers=workers) as pool,(root/'trajectories.jsonl').open('x',encoding='utf-8') as stream:
        for r in rows:stream.write(json.dumps(r,ensure_ascii=False)+'\n')
        stream.flush()
        # Independent episodes are scheduled in a fixed, outcome-blind order.
        # A long rule family must not prevent all other families from starting.
        order=sorted((i for i in range(len(jobs)) if i not in done),key=lambda i:digest(['execution-order',i]))
        pending={pool.submit(worker,jobs[i]):i for i in order}
        for f in as_completed(pending):
            i=pending[f]
            try:r=f.result()
            except Exception as e:errors.append(dict(index=i,error=repr(e)));print('ERROR',i,repr(e),flush=True);continue
            r['index']=i;rows.append(r);stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush()
            if len(rows)%8==0:print(len(rows),'/',len(jobs),'elapsed',round(time.perf_counter()-start,1),flush=True)
    return rows,errors,time.perf_counter()-start


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--stage',choices=('train','pilot','development','confirmation'),required=True)
    ap.add_argument('--models');ap.add_argument('--resume');ap.add_argument('--workers',type=int,default=4);ap.add_argument('--frozen',action='store_true');a=ap.parse_args();root=Path(a.root).resolve()
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        args=['--stage',a.stage,'--workers',str(a.workers)]
        if a.models:args+=['--models',str(Path(a.models).resolve())]
        if a.resume:args+=['--resume',str(Path(a.resume).resolve())]
        return dispatch(root,Path(__file__).name,args)
    if (root/'preregister.json').exists():raise FileExistsError('fresh cohort required')
    selected=[c for c in cases('confirmation' if a.stage=='confirmation' else 'development') if c['game'] in GAMES]
    models=None if a.models is None else json.loads(Path(a.models).read_text(encoding='utf-8'))
    counts={'train':16,'pilot':1,'development':4,'confirmation':8};base={'train':41000,'pilot':43000,'development':44000,'confirmation':45000}[a.stage]
    jobs=[]
    for case in selected:
        for i in range(counts[a.stage]):
            cc=copy.deepcopy(case)
            if case['game'] in ('auction_contest','hagetaka','no_thanks'):cc['deal']=cc.get('deal',0)+i*29
            cc.update(opening_steps=i%7,opening_seed=base+i+1000)
            for p in (profiles()[:1] if a.stage=='pilot' else profiles()):
                if a.stage=='train':jobs.append((cc,p,base+i))
                else:
                    for arm in ('proxy','memory','tail'):jobs.append((cc,p,base+i,arm,models[case['game']] if arm=='memory' else None))
    registration=dict(version='frontier-value-study-v1',stage=a.stage,reference=REFERENCE,feature_schema=SCHEMA,
        jobs=[j if a.stage=='train' else (*j[:4],None if j[4] is None else digest(j[4])) for j in jobs],
        source_snapshot=digest(json.loads((root/'source_snapshot.json').read_text())),
        hypotheses=['completed factual reference episodes can improve cutoff ranking','cheap public reflex tails can improve cutoff ranking'],
        fixed=dict(tree_horizon=2,tree_seeds=[401,402],tail_seeds=[701,702],tail_steps=16,regret=.15,rigidity=.5,neighbors=16,max_distance=2.),
        limits=['purpose bootstrap only; future personality flows omitted beyond tree cutoff','reference reflex differs from actual replanning policy',
            'semantic summary omits game-specific resource/topology detail','support heuristic is not calibrated model confidence',
            'changed rules and hidden regime shifts are separately reported; not presumed supported','training episodes may duplicate; model votes once per distinct trajectory'],
        acceptance='retain mechanism only if controls pass; do not promote universal policy on aggregate gains hiding game losses')
    previous=[]
    if a.resume:
        if a.stage=='train':raise ValueError('training resumption is outside this study contract')
        prior=Path(a.resume);old=json.loads((prior/'preregister.json').read_text(encoding='utf-8'))
        # Same scheduled episodes and settings; resume is NOT new independent
        # evidence. Preserve original source provenance for every reused row.
        for key in ('stage','jobs','reference','feature_schema','fixed','hypotheses','limits','acceptance'):
            if digest(old[key])!=digest(registration[key]):raise ValueError('resumption changed registered study: '+key)
        previous=[json.loads(s) for s in (prior/'trajectories.jsonl').read_text(encoding='utf-8').splitlines()]
        for r in previous:
            case,profile,seed,arm,_=jobs[r['index']]
            if (r['case'],r['profile'],r['seed'],r['arm'])!=(case,profile['id'],seed,arm):raise ValueError('completed row/job mismatch')
        registration['resumed']=dict(source=prior.name,registration=digest(old),source_snapshot=old['source_snapshot'],
            rows={str(r['index']):digest(r) for r in previous},
            reason='result-preserving public-rule and full-owner tail memoization; completed rows retained without double counting; identical strength settings')
    (root/'preregister.json').write_text(json.dumps(registration,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if models is not None:(root/'models.json').write_text(json.dumps(models)+'\n',encoding='utf-8')
    rows,errors,elapsed=pool_run(jobs,train_episode if a.stage=='train' else evaluate_episode,root,a.workers,previous)
    report=dict(stage=a.stage,planned=len(jobs),complete=len(rows),errors=errors,elapsed=elapsed)
    if a.stage=='train' and not errors:
        fitted={};counts={}
        for game in GAMES:
            group=[r for r in rows if r['case']['game']==game];x=[];y=[];ids=[];seen=set()
            for r in sorted(group,key=lambda r:r['index']):
                if not r['features']:continue
                if r['episode'] in seen:continue
                seen.add(r['episode']);x+=r['features'];y+=[r['goal']['value']]*len(r['features']);ids+=[r['episode']]*len(r['features'])
            m=TerminalMemory(x,y,ids,reference=REFERENCE+':'+game,neighbors=min(16,len(seen)));fitted[game]=m.record()
            counts[game]=dict(episodes=len(group),unique_episodes=len(seen),states=len(x),model=m.identity)
        (root/'models.json').write_text(json.dumps(fitted)+'\n',encoding='utf-8');report['training']=counts
    elif not errors:
        from tools.analyze_frontier_value import summarize
        report.update(summarize(rows,a.stage))
    (root/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)
    if errors:raise RuntimeError('failed jobs retained')


if __name__=='__main__':main()
