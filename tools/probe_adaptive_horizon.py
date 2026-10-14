"""Supplement: six-decision games, actual settlements, identical live binding.

Mechanism/transfer check, separate from the registered three-step primary study.
Two prediction modes see the same scripted public warmup and world draws.
"""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.closed_loop import reveal
from reflex.core import Policy,character_key,digest
from reflex.decision_loop import DecisionLoop
from reflex.integrated_trials import request_for
from reflex.laboratory import profiles
from reflex.model_observer import ModelFeedback
from reflex.preference_basis import PreferenceBasis
from tools.open_model_games import OpenGame
from tools.opportunity_models import OpenOpportunities
from tools.run_adaptive_opportunities import warmup,world,sample
from tools.run_frontier_value import pool_run


def evaluate(job):
    game,scenario,profile,seed=job;started=time.perf_counter();arms=[];graphs={}
    for arm in ('frozen','live'):
        g=OpenGame(game,profile,length=6,variant=1);memory,warm_id=warmup(g,scenario,seed)
        if arm=='frozen':g.kernel.update=False
        basis=PreferenceBasis(g.target,purpose=(('values.achievement',1.),));goals=[];trace=[];early=0
        for ep in range(4):
            s=g.start(ep,memory);c=g.observe(s)
            owner=DecisionLoop(c,policy=Policy(principle_priority='personal',commitments={character_key(c):.5}))
            while not g.terminal(s):
                before=g.kernel.record(s);req=request_for(g.observe(s,owner.state),'open-model')
                k=0 if game=='auction' else ep%2
                adapter=OpenOpportunities(g,ep,graphs.get(k));graphs[k]=adapter.graph
                table=adapter.solve(s,owner,preference_basis=basis,bound=True)
                ctrl=g.controller(opportunity=table,preference_basis=basis)
                decision=DecisionLoop.decide_batch([(owner,req)],False,ctrl.planner(owner,req,s))[0]
                action=decision['decision']['action_id'];assert action==table.action(s) and g.kernel.record(s)==before
                reveal(owner,decision['ticket'],ModelFeedback(None))
                rows,_,_,_,prediction=g.kernel.forecast(s,action)
                actual=world(g,s.public,scenario,ep);event=sample(rows,actual,seed,ep,s.tick,'test')
                after=g.kernel.reveal(s,action,event.public,event.flow)
                assert after.memory.observations==s.memory.observations+(arm=='live')
                trace.append(dict(episode=ep,tick=s.tick,public=s.public,action=action,event=event.symbol,
                    prediction=prediction,actual=actual,before_memory=digest(g.kernel.learner.record(s.memory)),
                    after_memory=digest(g.kernel.learner.record(after.memory)),graph_nodes=len(adapter.graph.nodes)))
                s=after
            early+=s.tick<6;memory=s.memory;goals.append(g.goal(s).value)
        arms.append(dict(arm=arm,warmup_id=warm_id,goals=goals,early_settlements=early,trace=trace))
    assert arms[0]['warmup_id']==arms[1]['warmup_id']
    return dict(game=game,scenario=scenario,profile=profile['id'],seed=seed,arms=arms,elapsed=time.perf_counter()-started)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--workers',type=int,default=2);p.add_argument('--frozen',action='store_true');a=p.parse_args();root=Path(a.root)
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        return dispatch(root,Path(__file__).name,['--workers',str(a.workers)])
    jobs=[(g,s,profiles()[0],83500) for g in ('auction','resources','combat','gambling') for s in ('shift','outlier')]
    reg=dict(jobs=jobs,source=digest(json.loads((root/'source_snapshot.json').read_text(encoding='utf-8'))),
        scope='six-decision extension, four test games after eight public warmup games; frozen/live; one profile/seed; mechanism not population strength',
        relation='supplement added after development; excluded from primary; no coefficient changes')
    (root/'preregister.json').write_text(json.dumps(reg,indent=2)+'\n',encoding='utf-8')
    rows,errors,elapsed=pool_run(jobs,evaluate,root,a.workers)
    result=dict(planned=len(jobs),complete=len(rows),errors=errors,elapsed=elapsed,games=sum(len(a['goals']) for r in rows for a in r['arms']),
        decisions=sum(len(a['trace']) for r in rows for a in r['arms']),early_settlements=sum(a['early_settlements'] for r in rows for a in r['arms']))
    (root/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result),flush=True)
    if errors:raise RuntimeError('extension failures retained')


if __name__=='__main__':main()
