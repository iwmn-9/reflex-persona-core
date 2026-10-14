"""Reconstruct factual memory and replay final-game openings at identical states.

This is a choice/forecast audit, not additional independent game trials. It
compares future-persona and objective-only continuation on the SAME evidence.
"""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import Policy, character_key, digest
from reflex.decision_loop import DecisionLoop
from reflex.integrated_trials import request_for
from tools.open_model_games import OpenGame
from tools.opportunity_models import OpenOpportunities
from tools.run_frontier_value import pool_run
from reflex.laboratory import profiles

INPUTS={};GRAPHS={}


def completed_rows(path):
    text=path.read_text(encoding='utf-8')
    lines=text.splitlines()
    if text and not text.endswith('\n'):lines=lines[:-1]
    return {r['index']:r for r in map(json.loads,lines)}


def evaluate(job):
    source,index=job
    if source not in INPUTS or index not in INPUTS[source]:
        INPUTS[source]=completed_rows(Path(source)/'trajectories.jsonl')
    r=INPUTS[source][index];assert r['arm']=='persona'
    profile=next(p for p in profiles() if p['id']==r['profile'])
    g=OpenGame(r['game'],profile,variant=r['variant']);memory=None;target=r['episodes']-1
    for t in r['trace']:
        if t['tick']==0:s=g.start(t['episode'],memory)
        assert t['public']==s.public and t['memory_before']==digest(g.kernel.learner.record(s.memory))
        if t['episode']==target:break
        events=g.events(s.public,t['action']);event=next(e for e in events if e.symbol==t['event'])
        s=g.kernel.reveal(s,t['action'],event.public,event.flow);memory=s.memory
    assert t['tick']==0 and t['episode']==target
    c=g.observe(s);owner=DecisionLoop(c,policy=Policy(principle_priority='personal',commitments={character_key(c):.5}))
    graphkey=(g.game,g.variant,target%2 if g.game!='auction' else 0)
    adapter=OpenOpportunities(g,target,GRAPHS.get(graphkey));GRAPHS[graphkey]=adapter.graph
    actual_before=digest([owner.record(),g.kernel.record(s)])
    result={}
    for continuation in ('persona','purpose'):
        endpoint=adapter.solve(s,owner,continuation);assert endpoint.solution is not None
        cap={};req=request_for(c,'open-model');ctrl=g.controller();ctrl.frontier=endpoint
        replica=copy.deepcopy(owner)
        d=DecisionLoop.decide_batch([(replica,req)],False,ctrl.planner(replica,req,s,captured=cap))[0]
        forecast=cap['node']['forecast'];values={forecast.roots[k][0]:v for k,v in forecast.purpose.items()}
        result[continuation]=dict(action=d['decision']['action_id'],root_purpose=values,table_action=endpoint.action(s),identity=endpoint.identity)
        if continuation=='persona':
            assert result[continuation]['action']==endpoint.action(s)==t['action']
            assert set(values)==set(t['root_purpose']) and all(abs(v-t['root_purpose'][k])<1e-12 for k,v in values.items())
        assert digest([owner.record(),g.kernel.record(s)])==actual_before
    return dict(source=Path(source).name,campaign=index,game=g.game,variant=g.variant,scenario=r['scenario'],profile=r['profile'],seed=r['seed'],
        episode=target,public=s.public,memory=digest(g.kernel.learner.record(s.memory)),
        choices=result,recorded_choice_and_values_reproduced=True)


def stream_run(jobs,root,workers):
    """Audit every registered opening as its input campaign becomes complete.

    Selection is fixed from the upstream registration, never from its outcome.
    A failed upstream study cannot be published as a complete streaming audit.
    """
    started=time.perf_counter();rows=[];errors=[];submitted=set();pending={}
    sources={source for source,_ in jobs};observed={};available={}
    with ProcessPoolExecutor(max_workers=workers) as pool,(root/'trajectories.jsonl').open('x',encoding='utf-8') as output:
        completed_sources=set()
        while len(rows)+len(errors)<len(jobs) or completed_sources!=sources:
            if time.perf_counter()-started>7200:raise TimeoutError('upstream/probe did not finish within two hours')
            for source in sources:
                path=Path(source)/'trajectories.jsonl'
                if not path.exists():continue
                stamp=path.stat().st_mtime_ns
                if observed.get(source)!=stamp:
                    available[source]=set(completed_rows(path));observed[source]=stamp
                finished=Path(source)/'evaluation.json'
                if finished.exists():
                    try:ev=json.loads(finished.read_text(encoding='utf-8'))
                    except json.JSONDecodeError:continue  # writer has not closed its final record yet
                    if ev['errors'] or ev['complete']!=ev['planned']:raise RuntimeError('upstream campaigns failed')
                    completed_sources.add(source)
            ready=[i for i,(source,index) in enumerate(jobs) if i not in submitted and index in available.get(source,set())]
            for i in sorted(ready,key=lambda i:digest(['execution-order',i])):
                pending[pool.submit(evaluate,jobs[i])]=i;submitted.add(i)
            if not pending:time.sleep(5);continue
            done,_=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
            for future in done:
                index=pending.pop(future)
                try:r=future.result()
                except Exception as e:errors.append(dict(index=index,error=repr(e)));print('ERROR',index,repr(e),flush=True);continue
                r['index']=index;rows.append(r);output.write(json.dumps(r,ensure_ascii=False)+'\n');output.flush()
                if len(rows)%8==0:print(len(rows),'/',len(jobs),'elapsed',round(time.perf_counter()-started,1),flush=True)
    return rows,errors,time.perf_counter()-started


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--inputs',nargs='+',required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--stream',action='store_true');p.add_argument('--frozen',action='store_true');a=p.parse_args();root=Path(a.root)
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        return dispatch(root,Path(__file__).name,['--workers',str(a.workers)]+(['--stream'] if a.stream else [])+['--inputs',*[str(Path(x).resolve()) for x in a.inputs]])
    jobs=[]
    for source in a.inputs:
        if not a.stream:
            ev=json.loads((Path(source)/'evaluation.json').read_text(encoding='utf-8'));assert not ev['errors'] and ev['complete']==ev['planned']
        registered=json.loads((Path(source)/'preregister.json').read_text(encoding='utf-8'))
        jobs.extend((source,i) for i,j in enumerate(registered['jobs']) if j[3]=='persona')
    registration=dict(selection='last-game first decision of every persona campaign, outcome-blind; factual memory replay; same evidence for both continuations',
        jobs=jobs,stream=a.stream,source=digest(json.loads((root/'source_snapshot.json').read_text(encoding='utf-8'))))
    (root/'preregister.json').write_text(json.dumps(registration,indent=2)+'\n',encoding='utf-8')
    rows,errors,elapsed=(stream_run if a.stream else lambda jobs,root,workers:pool_run(jobs,evaluate,root,workers))(jobs,root,a.workers)
    result=dict(planned=len(jobs),complete=len(rows),errors=errors,elapsed=elapsed,
        reproduced=sum(r['recorded_choice_and_values_reproduced'] for r in rows),
        purpose_continuation_changes_current_action=sum(r['choices']['persona']['action']!=r['choices']['purpose']['action'] for r in rows))
    (root/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result),flush=True)
    if errors:raise RuntimeError('failed probes retained')


if __name__=='__main__':main()
