"""Post-hoc root valuation audit; no changes to the policy or its decisions."""
import hashlib,json,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import reflex.closed_loop as loop
from reflex.decision_loop import DecisionLoop
from reflex.laboratory import PROFILES
from tools.continuing_games import ContinuingGame

original=loop.goal_forecast;captured=[]
def inspect(c,paths,audit,**kwargs):
    if kwargs['horizon']!=16:return original(c,paths,audit,**kwargs)
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
for pid in (0,1):
    source=R/f'results/owner_recovery_cloud/owner-recovery-resources-{pid}/trajectories.jsonl'
    run=next(x for x in map(json.loads,source.read_text().splitlines()) if x['arm']=='receding_recovery' and x['regime']=='outside')
    t=run['trace'][64];g=ContinuingGame('resources',PROFILES[pid],128,1);w=t['before'];c=g.context(w)
    own=DecisionLoop.from_record(c,t['checkpoint'],predictor=g.actor(c,learning='segmented').predictor)
    captured.clear();cap={};start=time.monotonic()
    ctrl=loop.owned_rollout(own,observe=g.context,events=g.events,terminal=g.terminal,assess=lambda w:g.goal(w).record(),
        record=lambda w:w,request_for=g.request,feedback=g.feedback,model_id=g.model_id,target=g.target,
        world_model='posterior',purpose_estimator='conditional',tail_recovery=2)
    req=g.request(c);result=DecisionLoop.decide_batch([(own,req)],False,ctrl.planner(own,req,w,captured=cap))[0]
    assert len(captured)==1
    outputs.append(dict(profile=run['profile'],tick=64,source=source.relative_to(R).as_posix(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        saved_action=t['action'],recomputed_action=result['decision']['action_id'],world=w,
        root_valuation=captured[0],seconds=time.monotonic()-start))
out=dict(cases=outputs,limitations=['Post-hoc valuation under the declared model, not proof that its future is correct.',
    'Profiles reached different public states. This is not a same-state causal comparison of personalities.',
    'All candidate actions are evaluated before the personality tier; the audit does not weaken that tier.'])
(R/'results/owner_resource_preference_diagnostic.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
