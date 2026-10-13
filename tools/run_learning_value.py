"""Frozen exact-policy evaluation and separately seeded hidden-world execution."""
import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import Policy, character_key, digest
from reflex.decision_loop import DecisionLoop
from reflex.closed_loop import reveal
from reflex.integrated_trials import request_for
from reflex.laboratory import profiles
from reflex.model_observer import ModelFeedback
from tools.learning_value_games import LearningGame
from tools.run_frontier_value import pool_run


def cases(stage):
    if stage=='holdout':
        return [dict(game='auction',limit=3,prizes=(3,8,5),budget=7,prior=(.35,.65)),
                dict(game='auction',limit=3,prizes=(6,2,9),budget=8,flip=.1),
                dict(game='resources',limit=5,cash=5,fee=2,prior=(.35,.65)),
                dict(game='resources',limit=4,poor_yield=1,prior=(.8,.2),flip=.1),
                dict(game='combat',limit=4,hp=2,enemy=6,signal_accuracy=.75,flip=.1),
                dict(game='combat',limit=4,hp=1,enemy=4,signal_accuracy=.6),
                dict(game='gambling',limit=4,cash=4,target=8,good_odds=.75,bad_odds=.25,signal_accuracy=.7,prior=(.45,.55)),
                dict(game='gambling',limit=4,cash=2,target=6,good_odds=.9,bad_odds=.3,signal_accuracy=.6,flip=.15)]
    base=[dict(game='auction',limit=3),dict(game='resources',limit=4),dict(game='combat',limit=3,hp=1),dict(game='gambling',limit=4)]
    if stage=='pilot':return base
    if stage=='continuation':
        return [{**c,'label':'continuation','prior':(.6,.4),'signal_accuracy':.8} for c in base]
    if stage=='development':
        return base+[{**c,'label':'noisy','signal_accuracy':.65,'prior':(.7,.3)} for c in base]
    # Registered after development, before opening confirmation results.
    return [{**c,**variation} for c in base for variation in (
        dict(label='shift',flip=.25,prior=(.4,.6)),
        dict(label='uninformative',signal_accuracy=.5),
        dict(label='expensive',fee=3,signal_accuracy=.85),
        dict(label='short',limit=2,signal_accuracy=.95))]


def evaluate(job):
    case,profile,arm,seeds=job;started=time.perf_counter();g=LearningGame(case,profile);start=g.start();c=g.observe(start)
    policy=Policy(principle_priority='personal',commitments={character_key(c):.5});owner=DecisionLoop(c,policy=policy)
    ctrl=g.controller(arm);cache={};trace=[];choice_cache={};oracle_cache={}
    if arm=='tail':
        from reflex.frontier_value import PolicyTail
        def advance(s,key,seed):
            rows=g.kernel(s,key);draw=random.Random(int(digest(['belief-tail',seed,s.tick])[:16],16)).random();mass=0.
            for t in rows:
                mass+=t.probability
                if draw<mass:return t.state,t.flow
            return rows[-1].state,rows[-1].flow
        ctrl.frontier=PolicyTail(observe=g.observe,advance=advance,terminal=g.terminal,assess=lambda s:g.goal(s).record(),
            request_for=ctrl.request_for,feedback=ctrl.feedback,model_id=g.kernel.identity,target=g.target,
            seeds=tuple(range(701,709)),steps=8,record=g.kernel.record)

    def oracle(s):
        if g.terminal(s):return g.goal(s).value
        stamp=digest(g.kernel.record(s))
        if stamp not in oracle_cache:
            oracle_cache[stamp]=max(sum(t.probability*oracle(t.state) for t in g.kernel(s,key)) for key in g.keys(s))
        return oracle_cache[stamp]

    def choose(s,actor):
        stamp=digest([g.kernel.record(s),actor.record()])
        if stamp in choice_cache:
            key,record,pred,work=choice_cache[stamp]
            # Preserve the same resolved owner continuation, not a fresh NPC.
            restored=DecisionLoop.from_record(g.observe(s),record,policy=policy)
            return key,restored,pred,work
        before=digest(actor.record());own=copy.deepcopy(actor);cc=g.observe(s,own.state);req=request_for(cc,'learning-value');cap={}
        h=1 if arm in ('myopic','tail') else case['limit']-s.public['tick']
        d=DecisionLoop.decide_batch([(own,req)],False,ctrl.planner(own,req,s,remaining=h,captured=cap))[0]
        if not cap.get('audit',{}).get('complete'):raise AssertionError('incomplete registered planning evaluation')
        key=d['decision']['action_id'];prediction=sum(p*goal['value'] for p,_,_,goal,_ in cap['node']['returns'])
        reveal(own,d['ticket'],ModelFeedback(None));assert before==digest(actor.record())
        assert own.personality==owner.personality and own.values==owner.values
        choice_cache[stamp]=(key,own.record(),prediction,cap['audit']['nodes'])
        return key,own,prediction,cap['audit']['nodes']

    def visit(s,actor):
        if g.terminal(s):return g.goal(s).value
        stamp=digest([g.kernel.record(s),actor.record()])
        if stamp in cache:return cache[stamp]
        key,after,pred,work=choose(s,actor);value=0.
        for t in g.kernel(s,key):value+=t.probability*visit(t.state,after)
        trace.append(dict(public=s.public,belief=list(s.weights),action=key,expected=value,predicted=pred,gap=pred-value,tree_nodes=work))
        cache[stamp]=value
        return value

    value=visit(start,owner);best=oracle(start);executions=[]
    # Actual hidden type is sampled once, then evolves according to the declared
    # transition. It NEVER goes to choose, observation, or the public belief.
    for seed in seeds:
        s=start;actor=copy.deepcopy(owner);r=random.Random(seed);u=r.random();h=g.names[0] if u<s.weights[0] else g.names[1];steps=[]
        while not g.terminal(s):
            key,actor,pred,work=choose(s,actor);outcomes=g.model(h,s.public,key)
            draw=random.Random(int(digest(['learning-value-actual',seed,s.tick])[:16],16)).random();total=0.
            for out in outcomes:
                total+=out.probability
                if draw<total:break
            before=g.kernel.record(s);s=g.kernel.reveal(s,key,out.public,out.flow);h=out.next_hypothesis
            restored=g.kernel.restore(json.loads(json.dumps(g.kernel.record(s))),lambda p:p)
            assert digest(g.kernel.record(restored))==digest(g.kernel.record(s))
            steps.append(dict(before=before,action=key,after=g.kernel.record(s),flow=out.flow))
        executions.append(dict(seed=seed,goal=g.goal(s).record(),steps=steps))
    if arm=='adaptive':assert all(abs(t['gap'])<1e-8 for t in trace),'same finite future policy must agree'
    return dict(case=case,profile=profile['id'],arm=arm,value=value,oracle=best,initial=next(t['action'] for t in trace if t['public']['tick']==0),
                trace=trace,executions=executions,states=len(cache),elapsed=time.perf_counter()-started)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--stage',choices=('pilot','development','confirmation','continuation','holdout'),default='development')
    p.add_argument('--workers',type=int,default=3);p.add_argument('--frozen',action='store_true');a=p.parse_args();root=Path(a.root)
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        return dispatch(root,Path(__file__).name,['--stage',a.stage,'--workers',str(a.workers)])
    first={'confirmation':63000,'holdout':65000}.get(a.stage,62000);seeds=list(range(first,first+32))
    ps=profiles()[:1] if a.stage=='pilot' else profiles()
    arms=('myopic','frozen','adaptive','tail') if a.stage=='continuation' else ('myopic','frozen','adaptive')
    jobs=[(case,profile,arm,seeds) for case in cases(a.stage) for profile in ps for arm in arms]
    registration=dict(version='learning-value-study-v1',jobs=jobs,source_snapshot=digest(json.loads((root/'source_snapshot.json').read_text())),
        contract='exact finite modeled expectations; 32 paired seeded hidden executions per policy are not independent exact-policy samples',
        scope='existing auction settlement plus three authored finite diagnostic games; not broad win-rate improvement',
        unchanged='personality, personal rigidity .5, goal corridor .15; factual posterior updated in all arms',
        contrasts='myopic = one decision; frozen = future inference disabled but actual inference enabled; adaptive = forecast inference enabled')
    (root/'preregister.json').write_text(json.dumps(registration,indent=2)+'\n',encoding='utf-8')
    rows,errors,elapsed=pool_run(jobs,evaluate,root,a.workers)
    pairs=[]
    for r in rows:
        if r['arm'] not in ('adaptive','tail'):continue
        for arm in ('myopic','frozen'):
            old=next((x for x in rows if x['case']==r['case'] and x['profile']==r['profile'] and x['arm']==arm),None)
            if old is None:continue  # Preserve errors and registration gaps below.
            pairs.append(dict(case=r['case'],profile=r['profile'],candidate=r['arm'],baseline=arm,before=old['value'],after=r['value'],delta=r['value']-old['value'],
                              initial_before=old['initial'],initial_after=r['initial'],oracle=r['oracle']))
    report=dict(planned=len(jobs),complete=len(rows),errors=errors,elapsed=elapsed,pairs=pairs,
                actual_executions=sum(len(r['executions']) for r in rows),actual_decisions=sum(len(e['steps']) for r in rows for e in r['executions']),
                policy_states=sum(r['states'] for r in rows),max_adaptive_prediction_gap=max((abs(t['gap']) for r in rows if r['arm']=='adaptive' for t in r['trace']),default=None))
    (root/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='pairs'}),flush=True)
    if errors:raise RuntimeError('failed registered runs retained')


if __name__=='__main__':main()
