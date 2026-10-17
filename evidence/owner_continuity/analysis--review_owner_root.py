"""Read-only full root valuation for a saved recovery-owner decision."""
import argparse,hashlib,json,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import reflex.closed_loop as loop
from reflex.decision_loop import DecisionLoop
from reflex.laboratory import PROFILES
from tools.continuing_games import ContinuingGame

p=argparse.ArgumentParser();p.add_argument('study');p.add_argument('--profile',required=True)
p.add_argument('--regime',required=True);p.add_argument('--ticks',type=int,nargs='+',required=True)
p.add_argument('--tail-steps',type=int,help='diagnostic horizon only; does not change the registered run')
p.add_argument('--seeds',type=int,nargs='+',default=[701,702,703],help='diagnostic model samples only; never actual world randomness')
p.add_argument('--output',required=True);a=p.parse_args()
source=Path(a.study)/'trajectories.jsonl'
run=next(x for x in map(json.loads,source.read_text().splitlines())
    if x['arm']=='receding_recovery' and x['profile']==a.profile and x['regime']==a.regime)
evaluation_tail=run['tail_steps'] if a.tail_steps is None else a.tail_steps
original=loop.goal_forecast;captured=[]

def inspect(c,paths,audit,**kwargs):
    if kwargs['horizon']!=evaluation_tail+1:return original(c,paths,audit,**kwargs)
    policy=kwargs['policy'];records=[]
    class Observe:
        def decide(self,b,*args,**kw):
            d=policy.decide(b,*args,**kw)
            records.extend(dict(plan=k,eligible=bool(d.eligible[0,i]),score=float(d.scores[0,i]),
                objective=float(d.features[0,i,0]),primary_need=float(d.features[0,i,1]),
                strongest_value=float(d.features[0,i,2]),loss=float(d.features[0,i,3]))
                for i,k in enumerate(b.ids[0]))
            return d
    kwargs['policy']=Observe();f=original(c,paths,audit,**kwargs)
    for row in records:
        k=row['plan'];row['action']=f.audit['plans'][k]['root']
        row['purpose_mean']=f.audit['purpose_mean_estimator']['roots'][k]['unprojected']
    captured.append(dict(discount=f.audit['discount'],rows=records))
    return f

loop.goal_forecast=inspect
outputs=[]
for tick in a.ticks:
    t=run['trace'][tick];assert t['tick']==tick
    g=ContinuingGame(run['game'],next(p for p in PROFILES if p['id']==a.profile),run['turns'],run['seed']%2)
    w=t['before'];c=g.context(w)
    own=DecisionLoop.from_record(c,t['checkpoint'],predictor=g.actor(c,learning='segmented').predictor)
    captured.clear();cap={};start=time.monotonic()
    ctrl=loop.owned_rollout(own,observe=g.context,events=g.events,terminal=g.terminal,
        assess=lambda w:g.goal(w).record(),record=lambda w:w,request_for=g.request,
        feedback=g.feedback,model_id=g.model_id,target=g.target,tail_steps=evaluation_tail,
        world_model='posterior',purpose_estimator='conditional',tail_recovery=2,seeds=tuple(a.seeds))
    req=g.request(c);result=DecisionLoop.decide_batch([(own,req)],False,ctrl.planner(own,req,w,captured=cap))[0]
    assert len(captured)==1
    outputs.append(dict(tick=tick,saved_action=t['action'],recomputed_action=result['decision']['action_id'],
        world=w,root_valuation=captured[0],seconds=time.monotonic()-start))
out=dict(game=run['game'],profile=run['profile'],seed=run['seed'],regime=run['regime'],
    registered_tail_steps=run['tail_steps'],diagnostic_tail_steps=evaluation_tail,diagnostic_model_seeds=a.seeds,
    source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),cases=outputs,
    limitations=['Post-hoc valuation under the declared model, not proof that its future is correct.',
                'The core, personality coefficients, and benchmark rules are not modified.',
                'Cross-platform recomputation is reported alongside the original saved action.'])
Path(a.output).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
