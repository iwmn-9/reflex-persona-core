"""Frozen paired campaigns: short horizon, objective future, persona future."""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import Policy,character_key,digest
from reflex.decision_loop import DecisionLoop
from reflex.closed_loop import reveal
from reflex.integrated_trials import request_for
from reflex.model_observer import ModelFeedback
from reflex.laboratory import profiles
from tools.open_model_games import OpenGame
from tools.opportunity_models import OpenOpportunities
from tools.run_open_model import truth
from tools.run_frontier_value import pool_run

GRAPHS={}


def evaluate(job):
    game,scenario,profile,arm,seed,episodes,variant=job;started=time.perf_counter()
    g=OpenGame(game,profile,arm='library' if arm=='library' else 'adaptive',variant=variant);memory=None;goals=[];trace=[];total=episodes*g.length
    for episode in range(episodes):
        s=g.start(episode,memory);c=g.observe(s)
        owner=DecisionLoop(c,policy=Policy(principle_priority='personal',commitments={character_key(c):.5}))
        while not g.terminal(s):
            before=g.kernel.record(s);cc=g.observe(s,owner.state);req=request_for(cc,'open-model');ctrl=g.controller();cap={};table_action=None;work=0
            if arm not in ('short','library'):
                graphkey=(game,variant,episode%2 if game!='auction' else 0)
                adapter=OpenOpportunities(g,episode,GRAPHS.get(graphkey));GRAPHS[graphkey]=adapter.graph
                solution=adapter.solve(s,owner,arm)
                if solution.solution is None:raise AssertionError('incomplete graph is not evaluated as a policy')
                ctrl.frontier=solution;table_action=solution.action(s);work=adapter.graph.last['solve_nodes']
            d=DecisionLoop.decide_batch([(owner,req)],False,ctrl.planner(owner,req,s,captured=cap))[0]
            if not cap.get('audit',{}).get('complete'):raise AssertionError('incomplete root forecast')
            key=d['decision']['action_id']
            if arm=='persona' and table_action!=key:raise AssertionError('projected future chooser and actual chooser differ')
            f=cap['node']['forecast'];root_values={f.roots[k][0]:v for k,v in f.purpose.items()}
            reveal(owner,d['ticket'],ModelFeedback(None))
            assert owner.personality==cc['personality'] and owner.values==cc['values']
            step=episode*g.length+s.public['tick'];actual=truth(g,s.public,scenario,step,total)
            rows,_,_,_,prediction=g.kernel.forecast(s,key)
            draw=random.Random(int(digest(['opportunity-actual',seed,step])[:16],16)).random();mass=0.
            for row in rows:
                mass+=actual[row.symbol]
                if draw<mass:break
            after=g.kernel.reveal(s,key,row.public,row.flow)
            assert g.kernel.record(s)==before
            trace.append(dict(episode=episode,tick=s.tick,public=s.public,action=key,event=row.symbol,flow=row.flow,
                prediction=prediction,actual=actual,log_loss=-math.log(prediction[row.symbol]),
                memory_before=digest(g.kernel.learner.record(s.memory)),memory_after=digest(g.kernel.learner.record(after.memory)),
                root_purpose=root_values,selected_purpose=d['deliberation']['selected_purpose'],table_action=table_action,
                graph_nodes=work,solution=None if arm in ('short','library') else solution.identity))
            s=after
        memory=s.memory;goals.append(g.goal(s).value)
    return dict(game=game,scenario=scenario,profile=profile['id'],arm=arm,seed=seed,episodes=episodes,variant=variant,
        goals=goals,trace=trace,final_memory=g.kernel.learner.record(memory),elapsed=time.perf_counter()-started)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--stage',choices=('pilot','development','holdout'),default='pilot')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--baseline-only',action='store_true');p.add_argument('--frozen',action='store_true');a=p.parse_args();root=Path(a.root)
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        return dispatch(root,Path(__file__).name,['--stage',a.stage,'--workers',str(a.workers)]+(['--baseline-only'] if a.baseline_only else []))
    ps=profiles();games=('auction','resources','combat','gambling');arms=('short','purpose','persona')
    if a.stage=='pilot':scenarios=('novel',);ps=ps[:1];seeds=(82000,);episodes=6;variant=0
    elif a.stage=='development':scenarios=('known','novel','conditional','noise');seeds=(82100,);episodes=12;variant=0
    else:scenarios=('reaction','shift','return','interaction');seeds=(82300,82301);episodes=12;variant=1
    if a.baseline_only:arms=('library',)
    jobs=[(g,s,p,arm,seed,episodes,variant) for g in games for s in scenarios for p in ps for arm in arms for seed in seeds]
    registration=dict(version='opportunity-study-v1',stage=a.stage,baseline_only=a.baseline_only,jobs=jobs,source=digest(json.loads((root/'source_snapshot.json').read_text(encoding='utf-8'))),
        scope='finite known public rules; per-decision learned prediction snapshot; all terminal opportunities; personality coefficient unchanged',
        comparison='short adaptive baseline; terminal expected-purpose continuation; terminal persona continuation; current persona chooses actual action in all arms',
        omissions='future learning within a solve and distant persona flow returns; real observation still updates memory before every next decision',
        outcomes='declared terminal purpose, within-game action and future/actual consistency; not win rate or human-level generalization')
    (root/'preregister.json').write_text(json.dumps(registration,indent=2)+'\n',encoding='utf-8')
    rows,errors,elapsed=pool_run(jobs,evaluate,root,a.workers)
    result=dict(planned=len(jobs),complete=len(rows),errors=errors,elapsed=elapsed,episodes=sum(len(r['goals']) for r in rows),decisions=sum(len(r['trace']) for r in rows))
    (root/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result),flush=True)
    if errors:raise RuntimeError('failed registered campaigns retained')

if __name__=='__main__':main()
