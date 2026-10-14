"""Pre-registered public-only online adaptation; frozen source and full traces."""
import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import Policy, character_key, digest
from reflex.decision_loop import DecisionLoop
from reflex.closed_loop import reveal
from reflex.integrated_trials import request_for
from reflex.model_observer import ModelFeedback
from reflex.laboratory import profiles
from tools.open_model_games import OpenGame
from tools.run_frontier_value import pool_run


def truth(g,w,scenario,step,total):
    """Experiment-only world process; never supplied to predictor or chooser."""
    legal=g.symbols(w);high=2 if g.game!='auction' else 3;scene=int(g.scene(w))
    target=0
    if scenario=='novel':target=high
    elif scenario=='conditional':target=high if scene else 0
    elif scenario=='reaction':target=high if w['previous']=='0' else 1
    elif scenario=='interaction':target=high if (scene==1) != (w['previous']=='0') else 0
    elif scenario=='shift':target=0 if step<total//2 else high
    elif scenario=='return':target=high if total//3<=step<2*total//3 else 0
    elif scenario=='noise':return dict.fromkeys(legal,1/len(legal))
    elif scenario not in ('known','outlier'):raise ValueError('registered hidden scenario required')
    if scenario=='outlier' and step%31==17:target=high
    target=str(min(target,len(legal)-1))
    # Residual noise is real and cannot be diagnosed perfectly as a change.
    noise=.35 if g.variant else .1
    return {s:(1-noise)*float(s==target)+noise/len(legal) for s in legal}


def evaluate(job):
    game,scenario,profile,arm,seed,episodes,horizon,variant=job;started=time.perf_counter()
    g=OpenGame(game,profile,arm,length=6,variant=variant);memory=None;logs=[];goals=[]
    ctrl=g.controller(horizon=horizon);total=episodes*g.length
    for episode in range(episodes):
        s=g.start(episode,memory);c=g.observe(s);policy=Policy(principle_priority='personal',commitments={character_key(c):.5});owner=DecisionLoop(c,policy=policy)
        while not g.terminal(s):
            before=g.kernel.record(s);cc=g.observe(s,owner.state);req=request_for(cc,'open-model');cap={}
            d=DecisionLoop.decide_batch([(owner,req)],False,ctrl.planner(owner,req,s,remaining=min(horizon,g.length-s.public['tick']),captured=cap))[0]
            if not cap.get('audit',{}).get('complete'):raise AssertionError('incomplete tree is not an evaluated policy')
            key=d['decision']['action_id'];reveal(owner,d['ticket'],ModelFeedback(None))
            assert owner.personality==dict(zip(cc['personality'],cc['personality'].values())) and owner.values==cc['values']
            step=episode*g.length+s.public['tick'];actual=truth(g,s.public,scenario,step,total)
            rows,_,_,_,forecast=g.kernel.forecast(s,key)
            rng=random.Random(int(digest(['open-world-actual',seed,step])[:16],16));draw=rng.random();mass=0.
            for row in rows:
                mass+=actual[row.symbol]
                if draw<mass:break
            # Objective-only one-step regret is diagnostic, not policy training.
            values={}
            for option in g.keys(s):
                values[option]=sum(actual[r.symbol]*g.goal(type(s)(r.public,s.memory,s.identity,s.tick+1)).value for r in g.events(s.public,option))
            after=g.kernel.reveal(s,key,row.public,row.flow)
            assert g.kernel.record(s)==before,'forecast changed factual state'
            if s.tick==0:
                restored=g.kernel.restore(json.loads(json.dumps(g.kernel.record(after))),lambda w:w)
                assert restored==after
            logs.append(dict(episode=episode,tick=s.tick,public=s.public,action=key,event=row.symbol,flow=row.flow,
                prediction=forecast,actual=actual,log_loss=-math.log(forecast[row.symbol]),
                regret=max(values.values())-values[key],diagnostics=g.kernel.learner.diagnostics(s.memory),
                memory_before=digest(g.kernel.learner.record(s.memory)),memory_after=digest(g.kernel.learner.record(after.memory)),
                tree_nodes=cap['audit']['nodes']))
            s=after
        memory=s.memory;goals.append(g.goal(s).value)
    return dict(game=game,scenario=scenario,profile=profile['id'],arm=arm,seed=seed,episodes=episodes,horizon=horizon,variant=variant,
        goals=goals,trace=logs,final_memory=g.kernel.learner.record(memory),elapsed=time.perf_counter()-started)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--stage',choices=('pilot','development','holdout','planning'),default='development')
    p.add_argument('--workers',type=int,default=3);p.add_argument('--frozen',action='store_true');a=p.parse_args();root=Path(a.root)
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        return dispatch(root,Path(__file__).name,['--stage',a.stage,'--workers',str(a.workers)])
    games=('auction','resources','combat','gambling');scenarios=('known','novel','conditional','shift','noise','outlier')
    arms=('library','marginal','long','adaptive');seeds=range(71000,71002);ps=profiles();episodes=24;horizon=1;variant=0
    if a.stage=='pilot':scenarios=('novel','conditional');arms=('library','adaptive');seeds=(70000,);ps=profiles()[:1];episodes=12
    elif a.stage=='holdout':scenarios=('reaction','return','interaction');seeds=range(73000,73004);variant=1
    elif a.stage=='planning':scenarios=('conditional','shift');arms=('library','adaptive');seeds=(74000,);episodes=12;horizon=2
    jobs=[(g,s,p,arm,seed,episodes,horizon,variant) for g in games for s in scenarios for p in ps for arm in arms for seed in seeds]
    registration=dict(version='open-model-study-v1',stage=a.stage,jobs=jobs,
        source=digest(json.loads((root/'source_snapshot.json').read_text())),
        scope='known observable legal rule alphabet; unnamed conditional event distributions learned from actual public observations',
        control='same fixed personalities and rigidity .5; goal corridor .15; all arms see same permitted event, no current hidden policy or unchosen outcome',
        outcomes='per-campaign goals and prequential prediction loss; oracle one-step purpose regret only offline; seeds/steps not independent game families')
    (root/'preregister.json').write_text(json.dumps(registration,indent=2)+'\n',encoding='utf-8')
    rows,errors,elapsed=pool_run(jobs,evaluate,root,a.workers)
    report=dict(planned=len(jobs),complete=len(rows),errors=errors,elapsed=elapsed,decisions=sum(len(r['trace']) for r in rows),
                episodes=sum(len(r['goals']) for r in rows))
    (root/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report),flush=True)
    if errors:raise RuntimeError('failed registered runs retained')


if __name__=='__main__':main()
