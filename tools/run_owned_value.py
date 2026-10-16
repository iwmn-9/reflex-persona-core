"""One persistent world: learned owner + endpoint evaluation + real replanning.

The hidden response law is used ONLY by the real world and diagnostic scores.
All control parameters are shared across the four rule families.
"""
import argparse,copy,json,sys,time
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import digest,character_key
from reflex.laboratory import PROFILES
from reflex.closed_loop import owned_rollout
from reflex.actor_model import ActorModel
from reflex.decision_loop import DecisionLoop
from reflex.model_observer import resolve_feedback
from reflex.flow_rollout import rollout
from reflex.intertemporal import forecast
from tools.continuing_games import ContinuingGame,GAMES


def sample(p,u):
    total=0.
    for k,v in p.items():
        total+=v
        if u<total:return k
    return list(p)[-1]


def law(t,turns,regime):
    stage=min(3,4*t//turns)
    top=0 if regime=='steady' else (2 if regime=='outside' else (0,2,1,0)[stage])
    return {str(i):(.8 if i==top else .1) for i in range(3)}


def run(job,arm):
    game,pid,seed,regime,turns,interval,tail_steps=job
    g=ContinuingGame(game,PROFILES[pid],turns,seed%2);w=g.start();c=g.context(w);own=g.actor(c,learned=arm!='endpoint_library')
    axes=character_key(c);rng=np.random.default_rng(seed);trace=[];checks=Counter();started=time.monotonic()
    for tick in range(turns):
        if g.terminal(w):break
        c=g.context(w,own.state);req=g.request(c);before=digest(own.record())
        p=own.predictor.predict(c);actual=law(tick,turns,regime)
        cap={};planner=None;checkpoint=own.record() if arm!='reflex' and tick%interval==0 else None
        if arm!='reflex' and tick%interval==0:
            if arm in ('endpoint','short_endpoint','endpoint_library'):
                ctrl=owned_rollout(own,observe=g.context,events=g.events,terminal=g.terminal,
                    assess=lambda w:g.goal(w).record(),record=lambda w:w,request_for=g.request,
                    feedback=g.feedback,model_id=g.model_id,target=g.target,
                    tail_steps=3 if arm=='short_endpoint' else tail_steps)
                planner=ctrl.planner(own,req,w,captured=cap)
            elif arm=='flow':
                def planner(cs,ds):
                    model=ActorModel(own,root_request=req,request_for=g.request,
                        feedback=lambda b,e,k,cc:g.feedback(b,e,k,cc,g.transition(b,k,e['signal'])[1]))
                    def advance(s,k,seed):
                        u=np.random.default_rng(np.random.SeedSequence([seed,s['tick']])).random()
                        return g.transition(s,k,sample(p,u))
                    paths,audit=rollout(cs[0],w,observe=g.context,advance=advance,terminal=g.terminal,
                        horizon=3,seeds=(701,702,703),observer=model,policy=own.policy)
                    cap['flow']=True
                    return forecast(cs[0],paths,horizon=3,unit='owner-decisions',target='incremental-purpose')
        mirror=ActorModel(own,root_request=req,request_for=g.request,
            feedback=lambda b,e,k,cc:g.feedback(b,e,k,cc,g.transition(b,k,e['signal'])[1]))
        prepared,mreq,_,_=mirror.prepare(w,c,prepared=True);mirror.preview()
        result=DecisionLoop.decide_batch([(own,req)],False,planner)[0];key=result['decision']['action_id']
        mirror.commit_choice(key);mirror.adopt_state(result['decision']['next_state'])
        end,flow=g.transition(w,key,sample(actual,rng.random()))
        resolve_feedback(own,result['ticket'],g.feedback(w,end,key,c,flow));mirror.advance(w,end,key,prepared,mreq)
        assert own.record()==mirror.actor.record();checks['owner_parity']+=1
        assert character_key(c)==axes;checks['fixed_axes']+=1
        if tick%23==0:
            record=json.loads(json.dumps(own.record()));own=DecisionLoop.from_record(c,record,predictor=own.predictor)
            assert own.record()==record;checks['restore']+=1
        diag=result.get('deliberation',{})
        trace.append(dict(tick=tick,before=w,after=end,action=key,response=end['signal'],
            goal=g.goal(end).value,law=p,truth=actual,mode=result['decision']['next_state']['mode'],
            brier=sum((p[z]-actual[z])**2 for z in p),owner=digest(own.record()),
            planned=bool(diag.get('adopted')),predicted=diag.get('selected_purpose'),
            before_owner=before,needs=own.pressure.debt,
            forecast={}))
        if checkpoint is not None:trace[-1]['checkpoint']=checkpoint
        if 'node' in cap:
            node=cap['node'];f=node['forecast']
            trace[-1]['forecast']={f.roots[k][0]:v for k,v in f.purpose.items()}
            trace[-1]['future_coverage']=[v['settled_fraction'] for v in node.get('frontiers',[])]
            trace[-1]['forecast_end']=min(turns,tick+f.horizon)
        w=end
    for t in trace:
        if t.get('predicted') is not None and 'forecast_end' in t:
            end=trace[min(t['forecast_end'],len(trace))-1]
            t['realized_at_forecast_end']=end['goal']
            t['forecast_error']=t['predicted']-end['goal']
    return dict(game=game,profile=PROFILES[pid]['id'],seed=seed,regime=regime,arm=arm,
        turns=turns,executed=len(trace),interval=interval,tail_steps=tail_steps,
        score=g.goal(w).value,final=w,checks=dict(checks),trace=trace,seconds=time.monotonic()-started)


def evaluate(job,arms):return [run(job,arm) for arm in arms]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--frozen',action='store_true')
    p.add_argument('--turns',type=int,default=128);p.add_argument('--interval',type=int,default=8)
    p.add_argument('--tail-steps',type=int,default=15);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--games',nargs='+',default=GAMES);p.add_argument('--profiles',nargs='+',type=int,default=[0,1,2,3])
    p.add_argument('--regimes',nargs='+',default=['steady','outside','shift-return'])
    p.add_argument('--seeds',nargs='+',type=int,default=[43]);p.add_argument('--arms',nargs='+',default=['reflex','flow','endpoint'])
    a=p.parse_args();root=Path(a.root)
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        args=[]
        for k in ('turns','interval','tail_steps','workers'):args+=['--'+k.replace('_','-'),str(getattr(a,k))]
        for k in ('games','profiles','regimes','seeds','arms'):args+=['--'+k,*map(str,getattr(a,k))]
        dispatch(root,Path(__file__).name,args);return
    root.mkdir(parents=True,exist_ok=True)
    jobs=[(g,i,s,r,a.turns,a.interval,a.tail_steps) for g in a.games for i in a.profiles for s in a.seeds for r in a.regimes]
    (root/'registration.json').write_text(json.dumps(dict(jobs=jobs,arms=a.arms,
        holdout_seeds=[170,203],semantics=__doc__),indent=2),encoding='utf8')
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool,(root/'trajectories.jsonl').open('w',encoding='utf8') as f:
        fs={pool.submit(evaluate,j,a.arms):j for j in jobs}
        for fu in as_completed(fs):
            for row in fu.result():rows.append(row);f.write(json.dumps(row,separators=(',',':'))+'\n');f.flush()
            print(json.dumps(dict(done=len(rows),total=len(jobs)*len(a.arms),job=fs[fu],scores={r['arm']:r['score'] for r in rows[-len(a.arms):]})),flush=True)
    (root/'summary.json').write_text(json.dumps([{k:v for k,v in row.items() if k!='trace'} for row in rows],indent=2),encoding='utf8')


if __name__=='__main__':main()
