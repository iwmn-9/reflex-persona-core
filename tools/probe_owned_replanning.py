"""Same-state prediction of the actual scheduled controller's next 15 turns.

These are held-fixed played development trajectories, not new performance wins.
Future game outcomes enter only the evaluator after both forecasts are made.
"""
import argparse,copy,json,sys,time,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import digest
from reflex.laboratory import PROFILES
from reflex.decision_loop import DecisionLoop
from reflex.model_observer import resolve_feedback
from reflex.closed_loop import owned_rollout
from tools.continuing_games import ContinuingGame


def probe(r,tick):
    p=next(p for p in PROFILES if p['id']==r['profile']);g=ContinuingGame(r['game'],p,r['turns'],r['seed']%2)
    w=g.start();own=g.actor(g.context(w))
    for t in r['trace'][:tick]:
        c=g.context(w,own.state);req=g.request(c)
        result=DecisionLoop.decide_batch([(own,req)],False,_counterfactual_roots=(t['action'],))[0]
        n,flow=g.transition(w,t['action'],t['response'])
        resolve_feedback(own,result['ticket'],g.feedback(w,n,t['action'],c,flow));w=n
        assert digest(own.record())==t['owner']
    args=dict(observe=g.context,events=g.events,terminal=g.terminal,assess=lambda w:g.goal(w).record(),
        record=lambda w:w,request_for=g.request,feedback=g.feedback,model_id=g.model_id,target=g.target)
    calls=[]
    def future(person,req,state):
        if state['tick']%r['interval']:return None
        calls.append(dict(tick=state['tick'],observations=person.predictor.observations))
        return owned_rollout(person,**args,tail_steps=r['tail_steps']).planner(person,req,state)
    saved=digest(own.record());forecasts={};started=time.monotonic()
    for name,kw in [('reflex_tail',{}),('scheduled_tail',dict(future_planner=future,
            future_planner_id='same-controller-every-'+str(r['interval']),future_planner_limit=1))]:
        ctrl=owned_rollout(own,**args,tail_steps=15,**kw)
        rows=ctrl.frontier.trajectories(w,own)
        forecasts[name]=dict(value=sum(p*g['value'] for p,flows,actions,g,ended in rows),
            paths=[dict(probability=p,actions=actions,goal=g) for p,flows,actions,g,ended in rows])
        assert digest(own.record())==saved
    actual=r['trace'][min(tick+15,len(r['trace']))-1]['goal']
    for value in forecasts.values():value['error']=value['value']-actual
    return dict(game=r['game'],profile=r['profile'],regime=r['regime'],seed=r['seed'],tick=tick,
        actual=actual,forecasts=forecasts,calls=calls,seconds=time.monotonic()-started,owner=saved)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',required=True)
    p.add_argument('--frozen',action='store_true');a=p.parse_args();root=Path(a.root);source=Path(a.source).resolve()
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        dispatch(root,Path(__file__).name,['--source',str(source)]);return
    rows=[json.loads(l) for l in source.read_text().splitlines()]
    jobs=[(r,t) for r in rows if r['profile']=='steady' and r['regime']=='outside' and r['arm']=='endpoint'
        for t in (17,81) if len(r['trace'])>=t+15]
    (root/'registration.json').write_text(json.dumps(dict(jobs=[(r['game'],t) for r,t in jobs],
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),semantics=__doc__),indent=2),encoding='utf8')
    results=[]
    with ProcessPoolExecutor(max_workers=2) as pool,(root/'predictions.jsonl').open('w') as f:
        fs=[pool.submit(probe,*j) for j in jobs]
        for fu in as_completed(fs):
            r=fu.result();results.append(r);f.write(json.dumps(r)+'\n');f.flush()
            print(json.dumps(dict(game=r['game'],tick=r['tick'],errors={k:v['error'] for k,v in r['forecasts'].items()})),flush=True)
    (root/'summary.json').write_text(json.dumps(dict(cases=len(results),
        mae={k:float(np.mean([abs(r['forecasts'][k]['error']) for r in results])) for k in ('reflex_tail','scheduled_tail')},
        cases_detail=results),indent=2),encoding='utf8')


if __name__=='__main__':main()
