"""Retrospective resource-allocation diagnosis; oracle never feeds the owner."""
import argparse
import copy
import json
import sys
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import Policy, character_key, compile_batch, digest
from reflex.decision_loop import DecisionLoop
from reflex.closed_loop import reveal
from reflex.integrated_trials import request_for
from reflex.model_observer import ModelFeedback
from reflex.laboratory import profiles
from reflex.deliberation import select
from tools.open_model_games import OpenGame
from tools.run_open_model import truth


def oracle_roots(g,world,scenario,total):
    # Exact full-horizon risk-neutral PURPOSE reference in these declared rules.
    # Equal clock/scores/budgets/last own category have identical rules/future
    # law here. Last opponent bid and signal do not affect the actual process.
    @lru_cache(None)
    def visit(tick,scores,budgets,previous):
        w=copy.deepcopy(world);w['tick']=tick;w['previous']=previous
        p=w['position'];p.update(round=tick,scores=list(scores),budgets=list(budgets))
        prizes=(3,6,2,5,4,7) if not g.variant else (5,2,7,3,6,4)
        p.update(prize=prizes[min(tick,5)],remaining=list(prizes[tick+1:]))
        s=g.kernel.start(w)
        if tick>=g.length:return g.goal(s).value,{}
        law=truth(g,w,scenario,w['episode']*g.length+tick,total);values={}
        for key in g.keys(s):
            value=0.
            for event in g.events(w,key):
                q=event.public;pos=q['position']
                value+=law[event.symbol]*visit(tick+1,tuple(pos['scores']),tuple(pos['budgets']),q['previous'])[0]
            values[key]=value
        return max(values.values()),values
    p=world['position'];answer=visit(world['tick'],tuple(p['scores']),tuple(p['budgets']),world['previous'])
    return dict(value=answer[0],roots=answer[1],states=visit.cache_info().currsize)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--study',required=True);p.add_argument('--frozen',action='store_true');a=p.parse_args();root=Path(a.root)
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        return dispatch(root,Path(__file__).name,['--study',str(Path(a.study).resolve())])
    rows=[json.loads(l) for l in (Path(a.study)/'trajectories.jsonl').read_text().splitlines()]
    selected=sorted((r for r in rows if r['game']=='auction' and r['arm']=='adaptive'),key=lambda r:r['index'])
    (root/'preregister.json').write_text(json.dumps(dict(source=str(a.study),selection='all adaptive auction pilot conditions; start of last complete episode',
        source_rows=digest(rows),cases=[r['index'] for r in selected],diagnosis='same actual public history; fixed forecast style removal; deeper reading; retrospective full-horizon objective oracle only'),indent=2)+'\n')
    findings=[]
    for row in selected:
        profile=next(p for p in profiles() if p['id']==row['profile']);g=OpenGame('auction',profile,row['arm'],variant=row['variant']);memory=None
        for episode in range(row['episodes']-1):
            s=g.start(episode,memory)
            for t in (r for r in row['trace'] if r['episode']==episode):
                assert t['public']==s.public
                event=next(e for e in g.events(s.public,t['action']) if e.symbol==t['event'])
                s=g.kernel.reveal(s,t['action'],event.public,event.flow)
            memory=s.memory
        s=g.start(row['episodes']-1,memory);c=g.observe(s);policy=Policy(principle_priority='personal',commitments={character_key(c):.5});owner=DecisionLoop(c,policy=policy);comparisons=[]
        for h in (1,2):
            own=copy.deepcopy(owner);req=request_for(g.observe(s,own.state),'open-model');ctrl=g.controller(h);cap={}
            d=DecisionLoop.decide_batch([(own,req)],False,ctrl.planner(own,req,s,remaining=h,captured=cap))[0];assert cap['audit']['complete']
            if h==row['horizon']:
                recorded=next(t for t in row['trace'] if t['episode']==row['episodes']-1 and t['tick']==0)
                assert d['decision']['action_id']==recorded['action'], 'restored factual history changed the original choice'
            f=cap['node']['forecast'];contexts=copy.deepcopy(f.contexts)
            for cc in contexts:
                for act in cc['actions']:
                    for outcome in act['outcomes']:outcome['style']={}
            restyled,_=select([owner._prepare(req)['effective']],replace(f,contexts=contexts),policy)
            batch=compile_batch(f.contexts,outcome_limit=f.outcome_limit);scores=policy.decide(batch,False,return_frame=f.return_frame)
            comparisons.append(dict(horizon=h,selected=d['decision']['action_id'],fixed_forecast_without_style=restyled[0]['action_id'],
                roots=[dict(root=f.roots[k][0],purpose=f.purpose[k],score=float(scores.scores[0,j]),eligible=bool(scores.eligible[0,j])) for j,k in enumerate(batch.ids[0])]))
        oracle=oracle_roots(g,s.public,row['scenario'],row['episodes']*g.length)
        findings.append(dict(scenario=row['scenario'],profile=row['profile'],state=g.kernel.record(s),comparisons=comparisons,oracle=oracle))
        print(json.dumps(dict(scenario=row['scenario'],choices=[x['selected'] for x in comparisons],oracle=oracle)),flush=True)
    (root/'evaluation.json').write_text(json.dumps(dict(findings=findings),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
