"""Frozen cross-game tests of live prediction repair with terminal valuation.

All hidden schedules and reference scores stay in this runner. Actor receives
only actual public outcomes. Same-state revision scores distinguish a changed
belief from a different board; a switch count alone is not a failure measure.
"""
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
from reflex.preference_basis import PreferenceBasis
from tools.open_model_games import OpenGame
from tools.opportunity_models import OpenOpportunities
from tools.run_frontier_value import pool_run

GRAPHS={}
ARMS=('frozen','long','live','legacy')


def world(g,w,scenario,episode,warm=False):
    legal=g.symbols(w);high=legal[-1];noise=.1 if g.variant==0 else .35
    t=episode*g.length+w['tick']
    high_mass=0.
    if not warm:
        if scenario=='shift':high_mass=1.
        elif scenario=='return':high_mass=float(episode<6)
        elif scenario=='outlier':high_mass=float(t==0)
        elif scenario=='gradual':high_mass=min(1.,(t+1)/(6*g.length))
        elif scenario not in ('stable','noise'):raise ValueError('registered world required')
    if scenario=='noise':return dict.fromkeys(legal,1/len(legal))
    return {s:noise/len(legal)+(1-noise)*((1-high_mass)*(s==legal[0])+high_mass*(s==high)) for s in legal}


def sample(rows,p,seed,episode,tick,phase):
    u=random.Random(int(digest(['adaptation-events',seed,phase,episode,tick])[:16],16)).random();mass=0.
    for row in rows:
        mass+=p[row.symbol]
        if u<mass:return row
    return rows[-1]


def reference_value(graph,solution,initial,law):
    """Diagnostic only: exact value of an already fixed policy under world law."""
    cache={}
    def visit(i):
        if i in cache:return cache[i]
        n=graph.nodes[i]
        if not n['branches']:v=n['goal'].value
        else:
            key=solution.choices[i]
            if key is None:raise AssertionError('missing fixed continuation')
            events=n['branches'][key];p=law(n['state'])
            v=math.fsum(p[symbol]*visit(child) for symbol,child,_ in events)
        cache[i]=v;return v
    return visit(graph.lookup[digest(graph.record(initial))])


def warmup(g,scenario,seed):
    memory=None;records=[]
    for ep in range(8):
        s=g.start(ep,memory)
        while not g.terminal(s):
            keys=g.keys(s)
            key=random.Random(int(digest(['public-opening',seed,ep,s.tick])[:16],16)).choice(keys)
            rows=g.events(s.public,key);p=world(g,s.public,scenario,ep,warm=True)
            row=sample(rows,p,seed,ep,s.tick,'warm');records.append([s.public,key,row.symbol])
            s=g.kernel.reveal(s,key,row.public,row.flow)
        memory=s.memory
    return memory,digest(records)


def run_arm(game,scenario,profile,seed,variant,arm,episodes):
    g=OpenGame(game,profile,arm='long' if arm=='long' else 'adaptive',length=3,variant=variant)
    memory,warm_id=warmup(g,scenario,seed)
    warm_memory=g.kernel.learner.record(memory)
    if arm=='frozen':g.kernel.update=False;g.kernel.identity=digest([g.kernel.state_identity,False])
    basis=None if arm=='legacy' else PreferenceBasis(g.target,purpose=(('values.achievement',1.),))
    trace=[];goals=[];starts=[]
    for ep in range(episodes):
        s=g.start(ep,memory);c=g.observe(s)
        owner=DecisionLoop(c,policy=Policy(principle_priority='personal',commitments={character_key(c):.5}))
        old=None
        while not g.terminal(s):
            before=g.kernel.record(s);cc=g.observe(s,owner.state);req=request_for(cc,'open-model')
            cache_key=(game,variant,ep%2 if game!='auction' else 0)
            adapter=OpenOpportunities(g,ep,GRAPHS.get(cache_key));GRAPHS[cache_key]=adapter.graph
            solved=adapter.solve(s,owner,preference_basis=basis,bound=True)
            ctrl=g.controller(opportunity=solved,preference_basis=basis);cap={}
            d=DecisionLoop.decide_batch([(owner,req)],False,ctrl.planner(owner,req,s,captured=cap))[0]
            key=d['decision']['action_id']
            if key!=solved.action(s) or not cap.get('audit',{}).get('complete'):raise AssertionError('forecast/execution mismatch')
            if g.kernel.record(s)!=before:raise AssertionError('planning mutated factual evidence')
            f=cap['node']['forecast'];purposes={f.roots[k][0]:v for k,v in f.purpose.items()}
            # Both policies evaluated at the SAME current public board. Hidden
            # truth is used only below, after the actor has committed its action.
            old_action=None if old is None else old.solution.action(s.public)
            revision=None
            if old_action is not None and old_action!=key:
                law=lambda w:world(g,w,scenario,ep)
                old_value=reference_value(adapter.graph,old.solution,s.public,law)
                new_value=reference_value(adapter.graph,solved.solution,s.public,law)
                revision=dict(previous_action=old_action,old_value=old_value,new_value=new_value,delta=new_value-old_value)
            if s.tick==0:starts.append(key)
            reveal(owner,d['ticket'],ModelFeedback(None))
            if owner.personality!=cc['personality'] or owner.values!=cc['values']:raise AssertionError('changed persona')
            rows,_,_,_,prediction=g.kernel.forecast(s,key)
            actual=world(g,s.public,scenario,ep);row=sample(rows,actual,seed,ep,s.tick,'test')
            after=g.kernel.reveal(s,key,row.public,row.flow)
            if after.memory.observations!=s.memory.observations+(arm!='frozen'):raise AssertionError('lost or double reveal')
            trace.append(dict(episode=ep,tick=s.tick,public=s.public,action=key,event=row.symbol,
                prediction=prediction,actual=actual,log_loss=-math.log(prediction[row.symbol]),
                probability_error=sum((prediction[k]-actual[k])**2 for k in actual),
                surprise=-math.log(prediction[row.symbol]),revision=revision,previous_action=old_action,
                root_purpose=purposes,memory_before=digest(g.kernel.learner.record(s.memory)),
                memory_after=digest(g.kernel.learner.record(after.memory)),solution=solved.identity,
                short_weight=sum(r['weight'] for r in g.kernel.learner.diagnostics(after.memory)['experts'] if r['window']==8)))
            s=after;old=solved
        memory=s.memory;goals.append(g.goal(s).value)
    return dict(arm=arm,warmup_id=warm_id,warm_memory=warm_memory,goals=goals,starts=starts,trace=trace,
                final_memory=g.kernel.learner.record(memory))


def evaluate(job):
    game,scenario,profile,seed,variant,episodes=job;started=time.perf_counter()
    arms=[run_arm(game,scenario,profile,seed,variant,a,episodes) for a in ARMS]
    if len({a['warmup_id'] for a in arms})!=1:raise AssertionError('unmatched public warmup')
    if arms[0]['warm_memory']!=arms[2]['warm_memory']:raise AssertionError('live/frozen initial beliefs differ')
    return dict(game=game,scenario=scenario,profile=profile['id'],seed=seed,variant=variant,episodes=episodes,
                arms=arms,elapsed=time.perf_counter()-started)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--stage',choices=('pilot','development','holdout'),default='pilot')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--frozen',action='store_true');a=p.parse_args();root=Path(a.root)
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        return dispatch(root,Path(__file__).name,['--stage',a.stage,'--workers',str(a.workers)])
    ps=profiles();scenarios=('stable','noise','outlier','shift','return','gradual');seeds=(83100,83101);variant=0;episodes=12
    if a.stage=='pilot':ps=ps[:1];scenarios=('shift','outlier');seeds=(83000,);episodes=3
    if a.stage=='holdout':seeds=(83300,83301,83302,83303);variant=1
    jobs=[(g,s,p,seed,variant,episodes) for g in ('auction','resources','combat','gambling') for s in scenarios for p in ps for seed in seeds]
    registration=dict(version='adaptive-opportunity-v1',stage=a.stage,jobs=jobs,arms=ARMS,
        source=digest(json.loads((root/'source_snapshot.json').read_text(encoding='utf-8'))),
        fixed='original four personas; commitment .5; corridor .15; original EventLearner 8/32/128, long control 128; no game-dependent adaptation constants',
        controls='frozen warmup vs live: same starting evidence; long window: sensitivity control; legacy: no PreferenceBasis; warmup public actions shared',
        metrics='actual terminal goals per family, pre-observation prediction loss, same-board changed-policy terminal effect; flips alone not judged harmful',
        scope='four public diagnostic games of 3 decisions; 8 warmup games and 12 test games; finite terminal planning with current belief; replan after actual reveal; no future learning within solve',
        warning='hidden world schedules and reference probabilities diagnostic only; correlated steps/personas not independent games; no unconditional improvement claim')
    (root/'preregister.json').write_text(json.dumps(registration,indent=2)+'\n',encoding='utf-8')
    rows,errors,elapsed=pool_run(jobs,evaluate,root,a.workers)
    result=dict(planned=len(jobs),complete=len(rows),errors=errors,elapsed=elapsed,
        games=sum(len(a['goals']) for r in rows for a in r['arms']),decisions=sum(len(a['trace']) for r in rows for a in r['arms']))
    (root/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result),flush=True)
    if errors:raise RuntimeError('registered failed jobs retained')


if __name__=='__main__':main()
