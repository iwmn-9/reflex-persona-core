import json,sys,time,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from reflex.closed_loop import owned_rollout
from reflex.decision_loop import DecisionLoop
from reflex.laboratory import PROFILES
from tools.continuing_games import ContinuingGame
source=R/'results/owner_continuity_cloud/owner-continuity-combat-1/frequent/trajectories.jsonl'
row=next(r for r in map(json.loads,source.read_text().splitlines()) if r['arm']=='endpoint_posterior' and r['regime']=='outside')
t=row['trace'][64];g=ContinuingGame('combat',PROFILES[1],128,1);w=t['before'];c=g.context(w)
own=DecisionLoop.from_record(c,t['checkpoint'],predictor=g.actor(c,learning='segmented').predictor)
started=time.perf_counter()
ctrl=owned_rollout(own,observe=g.context,events=g.events,terminal=g.terminal,assess=lambda w:g.goal(w).record(),
 record=lambda w:w,request_for=g.request,feedback=g.feedback,model_id=g.model_id,target=g.target,
 world_model='posterior',purpose_estimator='conditional',tail_recovery=2)
req=g.request(c);r=DecisionLoop.decide_batch([(own,req)],False,ctrl.planner(own,req,w))[0]
result=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),tick=64,
 action=r['decision']['action_id'],seconds=time.perf_counter()-started,future_calls=ctrl.frontier.future_calls,
 purpose=r['deliberation']['selected_purpose'],profiled=False,seeds=[701,702,703],
 note='Single local state capacity measurement, not a decision-quality test or a cloud runtime estimate.')
(R/'results/owner_recovery_runtime.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
