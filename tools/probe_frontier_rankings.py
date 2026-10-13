"""Frozen changed-root pairs, independently evaluated under one continuation.

This diagnoses ranking under the PUBLIC model and reflex continuation. It is
not a causal oracle for a different future replanning policy or hidden deck.
"""
import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import digest
from reflex.decision_loop import DecisionLoop
from reflex.laboratory import profiles
from reflex.integrated_trials import request_for
from tools.run_frontier_value import owned,pool_run
from tools.run_foundation_audit import prepare


def evaluate(job):
    case,profile,seed,arm,baseline,candidate,at=job;b,w,prefix=prepare(case,profile,seed);actor=owned(b,w)
    for t in baseline[:at]:
        c=b.observe(b.project(w),actor.state);req=request_for(c,case['game'])
        d=DecisionLoop.decide_batch([(actor,req)],False,_counterfactual_roots=(t['root'],))[0];actor.abandon(d['ticket'])
        w,flow=b.actual(w,t['root'],seed)
        assert digest(b.record(w))==digest(t['after']),'public replay differs'
        assert digest(actor.record())==t['owner_after'],'owner replay differs'
    assert digest(b.record(w))==digest(baseline[at]['before'])==digest(candidate[at]['before']),'changed-root states differ'
    moves=(baseline[at]['root'],candidate[at]['root']);samples={move:[] for move in moves}
    for model_seed in range(61000,61064):
        for move in moves:
            own=copy.deepcopy(actor);world=b.project(w);first=True;steps=0
            while not b.terminal(world):
                c=b.observe(world,own.state);req=request_for(c,case['game'])
                d=DecisionLoop.decide_batch([(own,req)],False,_counterfactual_roots=(move,))[0] if first else own.decide(req,False)
                key=d['decision']['action_id'];own.abandon(d['ticket']);world,flow=b.modeled_advance(world,key,model_seed)
                first=False;steps+=1
                if steps>256:raise AssertionError('unsettled reference ranking sample')
            samples[move].append(b.goal(world).value)
    diffs=[y-x for x,y in zip(samples[moves[0]],samples[moves[1]])]
    return dict(case=case,profile=profile['id'],seed=seed,arm=arm,divergence=at,public=b.record(b.project(w)),
        baseline=moves[0],candidate=moves[1],samples=samples,paired_gain=sum(diffs)/len(diffs),
        baseline_prediction=baseline[at]['prediction'],candidate_prediction=candidate[at]['prediction'],
        owner=digest(actor.record()),actual_continuation='owned reflex',public_model_only=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--sources',nargs=2,required=True);p.add_argument('--workers',type=int,default=3);p.add_argument('--frozen',action='store_true');a=p.parse_args();root=Path(a.root).resolve()
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        return dispatch(root,Path(__file__).name,['--sources',*[str(Path(s).resolve()) for s in a.sources],'--workers',str(a.workers)])
    jobs=[];parents={};profile_map={p['id']:p for p in profiles()}
    for source in map(Path,a.sources):
        rows=[json.loads(s) for s in (source/'trajectories.jsonl').read_text().splitlines()];parents[source.name]=digest(rows)
        for game in ('no_thanks','goofspiel'):
            seeds=sorted({r['seed'] for r in rows if r['case']['game']==game})[:2]
            for r in sorted(rows,key=lambda r:r['index']):
                if r['case']['game']!=game or r['seed'] not in seeds or r['arm']=='proxy':continue
                old=next(q for q in rows if q['case']==r['case'] and q['profile']==r['profile'] and q['seed']==r['seed'] and q['arm']=='proxy')
                at=next((i for i,(x,y) in enumerate(zip(old['trace'],r['trace'])) if x['root']!=y['root']),None)
                if at is not None:jobs.append((r['case'],profile_map[r['profile']],r['seed'],r['arm'],old['trace'],r['trace'],at))
    pre=dict(version='frontier-independent-ranking-probe-v1',parents=parents,source_snapshot=digest(json.loads((root/'source_snapshot.json').read_text())),
        jobs=[dict(case=j[0],profile=j[1]['id'],seed=j[2],arm=j[3],at=j[6],traces=digest(j[4:6])) for j in jobs],
        sampling='first two registered seeds per source/game, every persona, first changed root; no selection by final result',
        seeds=list(range(61000,61064)),continuation='same owned reflex to terminal; public model only',
        limits=['not actual hidden deck or future policy oracle','64 model samples are not 64 observed matches','discovery diagnosis, not policy fitting'])
    (root/'preregister.json').write_text(json.dumps(pre,indent=2)+'\n');rows,errors,elapsed=pool_run(jobs,evaluate,root,a.workers)
    report=dict(planned=len(jobs),complete=len(rows),errors=errors,elapsed=elapsed,comparisons=[{k:v for k,v in r.items() if k not in ('samples','public')} for r in rows])
    (root/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(planned=len(jobs),complete=len(rows),errors=errors)),flush=True)
    if errors:raise RuntimeError('ranking failures retained')


if __name__=='__main__':main()
