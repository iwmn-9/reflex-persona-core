"""Score all estimators on the SAME factual reference-policy states."""
import argparse,copy,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import digest
from reflex.frontier_value import TerminalMemory,PolicyTail
from reflex.integrated_trials import request_for
from reflex.laboratory import profiles
from tools.closed_loop_models import controller
from tools.run_foundation_audit import prepare,GAMES
from tools.run_contingent_coverage import cases
from tools.run_frontier_value import features,owned,crps,verify,pool_run,SCHEMA,REFERENCE


def episode(job):
    case,profile,seed,model=job;b,w,prefix=prepare(case,profile,seed);actor=owned(b,w);m=TerminalMemory.from_record(model)
    ctrl=controller(b,horizon=2)
    tail=PolicyTail(observe=b.observe,advance=b.modeled_advance,terminal=b.terminal,assess=lambda world:b.goal(world).record(),
        request_for=ctrl.request_for,feedback=ctrl.feedback,model_id=ctrl.model_id,target=b.target,seeds=(701,702),steps=16)
    predictions=[];trace=[]
    while not b.terminal(w):
        if len(trace)%4==0:
            public=b.project(w);saved=digest(actor.record());learned=m.predict(features(b,public,actor));sampled=tail.evaluate(public,actor)
            predictions.append(dict(step=len(trace),public=b.record(public),proxy=((1.,b.goal(public).value),),
                memory=None if learned is None else learned.outcomes,tail=sampled.outcomes,tail_settled=sampled.settled_fraction))
            assert digest(actor.record())==saved
        c=b.observe(b.project(w),actor.state);d=actor.decide(request_for(c,case['game']),False);key=d['decision']['action_id']
        before=b.record(w);w,flow=b.actual(w,key,seed);actor.abandon(d['ticket']);trace.append(dict(before=before,root=key,after=b.record(w),flow=flow))
        if len(trace)>1000:raise AssertionError('unsettled reference episode')
    verify(b,prefix+trace,seed,w);goal=b.goal(w).record()
    for row in predictions:
        row['crps']={k:crps(row[k] if row[k] is not None else row['proxy'],goal['value']) for k in ('proxy','memory','tail')}
    return dict(case=case,profile=profile['id'],seed=seed,goal=goal,prefix=prefix,trace=trace,predictions=predictions,replay=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--models',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--frozen',action='store_true');a=p.parse_args();root=Path(a.root).resolve()
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        return dispatch(root,Path(__file__).name,['--models',str(Path(a.models).resolve()),'--workers',str(a.workers)])
    if (root/'preregister.json').exists():raise FileExistsError('fresh prediction probe required')
    models=json.loads(Path(a.models).read_text());jobs=[]
    for stage,base in (('development',46000),('confirmation',47000)):
        for case in [c for c in cases(stage) if c['game'] in GAMES]:
            for i in range(4):
                cc=copy.deepcopy(case);cc.update(opening_steps=i,opening_seed=base+i+1000)
                if cc['game'] in ('auction_contest','hagetaka','no_thanks'):cc['deal']=cc.get('deal',0)+29*i
                for profile in profiles():jobs.append((cc,profile,base+i,models[case['game']]))
    pre=dict(version='frontier-shared-state-predictions-v1',reference=REFERENCE,feature_schema=SCHEMA,
        source_snapshot=digest(json.loads((root/'source_snapshot.json').read_text())),models=digest(models),
        jobs=[(*j[:3],digest(j[3])) for j in jobs],sampling='every fourth real reference-policy decision, before actual transition',
        metric='empirical CRPS on eventual factual result; each episode equal weight; unsupported memory uses proxy',
        limitation='held-out diagnostic, not estimator selection/calibration training; current public model may miss later regime changes')
    (root/'preregister.json').write_text(json.dumps(pre,indent=2)+'\n');(root/'models.json').write_text(json.dumps(models)+'\n')
    rows,errors,elapsed=pool_run(jobs,episode,root,a.workers);report=dict(planned=len(jobs),complete=len(rows),errors=errors,elapsed=elapsed,groups=[])
    for game in GAMES:
        for stage in ('development','confirmation'):
            group=[r for r in rows if r['case']['game']==game and (r['seed']<47000)==(stage=='development')]
            means={arm:sum(sum(p['crps'][arm] for p in r['predictions'])/len(r['predictions']) for r in group)/len(group) for arm in ('proxy','memory','tail')} if group else {}
            report['groups'].append(dict(game=game,stage=stage,episodes=len(group),states=sum(len(r['predictions']) for r in group),
                supported=sum(p['memory'] is not None for r in group for p in r['predictions']),crps=means))
    (root/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    if errors:raise RuntimeError('failed prediction probe retained')


if __name__=='__main__':main()
