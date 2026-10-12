"""Exact closed-loop policy success under finite public chance rules.

This is an offline evaluator, not an oracle secretly supplied to an NPC.
Dynamic actor state participates in the memo key. Goal-only backward induction
is an upper bound; personality sacrifice is not automatically a defect.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import Policy,character_key,digest
from reflex.contingent_games import make_bridge,Frame
from reflex.decision_loop import DecisionLoop
from reflex.integrated_trials import request_for
from reflex.flow_rollout import rollout
from reflex.purpose_plan import goal_forecast
from reflex.deliberation import select
from reflex.forecast_control import restrict
from reflex.laboratory import profiles
from tools.run_foundation_audit import graph_values


def measure(job):
    graph,length,profile,arm=job;started=time.perf_counter()
    b=make_bridge(profile,dict(game='public_graph',graph=graph,length=length));w=b.start();c=b.observe(w)
    p=Policy(principle_priority='personal',commitments={character_key(c):.5});actor=DecisionLoop(c,policy=p)
    optimal, q=graph_values(b.rules);cache={};trace=[]
    def visit(world,owner):
        if b.terminal(world):return optimal[world.position],0.
        stamp=digest([world.position,owner.record()])
        if stamp in cache:return cache[stamp]
        current=copy.deepcopy(owner);c=b.observe(world,current.state);req=request_for(c,'public_graph')
        def planner(cs,ds):
            paths,audit=rollout(cs[0],world,observe=b.observe,advance=b.modeled_advance,terminal=b.terminal,
                assess=lambda s:b.goal(s).record(),horizon=4,seeds=(401,402,403,404),policy=p)
            f=goal_forecast(cs[0],paths,audit,horizon=4,unit=b.unit,target=b.target,policy=p,settlement_weight='absolute')
            if arm!='purpose4':return f
            all_f=goal_forecast(cs[0],paths,audit,horizon=4,unit=b.unit,target=b.target,
                policy=Policy(principle_priority='finite'),settlement_weight='absolute')
            _,vote=select(cs,f,p);persona=vote['selected_plan'];best=max(all_f.purpose.values())
            wanted=persona if persona in all_f.purpose and abs(all_f.purpose[persona]-best)<1e-12 else min(
                k for k,v in all_f.purpose.items() if abs(v-best)<1e-12)
            return restrict(all_f,wanted)
        result=DecisionLoop.decide_batch([(current,req)],False,None if arm=='reflex' else planner)[0]
        key=result['decision']['action_id'];current.abandon(result['ticket'])
        t,n=world.position;success=regret=0.
        children=[]
        for after in b.rules.edges[t,n,key]:
            v,g=visit(Frame((t+1,after),world.turns+1),current);success+=v/2;regret+=g/2;children.append(after)
        regret+=optimal[t,n]-q[t,n,key]
        # Telescoping exact expectation; no Monte Carlo interval or repeated
        # persona/seed count can turn into extra independent evidence here.
        assert abs((optimal[t,n]-success)-regret)<1e-12
        cache[stamp]=(success,regret)
        trace.append(dict(key=stamp,node=list(world.position),actor_state=owner.state,action=key,
            next_nodes=children,success_probability=success,expected_opportunity_loss=regret))
        return success,regret
    value,regret=visit(w,actor)
    return dict(job=job,graph=graph,length=length,profile=profile['id'],arm=arm,
        success_probability=value,optimal_probability=optimal[0,0],expected_opportunity_loss=regret,
        unique_owned_states=len(cache),trace=trace,elapsed_seconds=time.perf_counter()-started)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--frozen',action='store_true');a=ap.parse_args();root=Path(a.root).resolve()
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        return dispatch(root,Path(__file__).name,['--workers',str(a.workers)])
    jobs=[(graph,length,p,arm) for graph in range(23108,23116) for length in (5,12)
        for p in profiles() for arm in ('reflex','persona4','purpose4')]
    pre=dict(format='foundation-exact-policy-v1',jobs=jobs,
        hypothesis='the current controller still leaves objective opportunity after persona root preference is removed',
        timing='follow-up after the sampled graph result; new graph identifiers, no parameter fitting',
        semantics='exact expectation under the declared equiprobable finite public model, not an actual game win-rate estimate',
        controls=['memo key includes full owned checkpoint','entire future chance tree, no sampled actual seeds',
            'offline optimal values are never supplied to the tested decision policy',
            'expected cumulative oracle opportunity loss equals optimal initial probability minus actual policy probability'],
        source_snapshot_sha256=hashlib.sha256((root/'source_snapshot.json').read_bytes()).hexdigest())
    (root/'preregister.json').write_text(json.dumps(pre,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    rows=[];errors=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool,(root/'policies.jsonl').open('x',encoding='utf-8') as stream:
        fs={pool.submit(measure,j):i for i,j in enumerate(jobs)}
        for f in as_completed(fs):
            i=fs[f]
            try:r=f.result()
            except Exception as e:errors.append(dict(index=i,error=repr(e)));print('ERROR',i,repr(e),flush=True);continue
            r['index']=i;stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush()
            rows.append({k:v for k,v in r.items() if k!='trace'})
            print(len(rows)+len(errors),'/',len(jobs),r['graph'],r['length'],r['profile'],r['arm'],r['success_probability'],flush=True)
    result=dict(format=pre['format'],planned=len(jobs),complete=len(rows),errors=errors,runs=rows,
        elapsed_seconds=time.perf_counter()-start,preregister_sha256=digest(pre),
        policies_sha256=hashlib.sha256((root/'policies.jsonl').read_bytes()).hexdigest())
    (root/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if errors:raise RuntimeError('incomplete exact evaluation retained')


if __name__=='__main__':main()
