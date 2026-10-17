import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import compile_batch,digest
from reflex.closed_loop import owned_rollout
from reflex.decision_loop import DecisionLoop
from reflex.laboratory import PROFILES
from tools.continuing_games import ContinuingGame
root=Path('results/owner_continuity_cloud/owner-continuity-combat-1/frequent')
r=next(r for r in map(json.loads,(root/'trajectories.jsonl').read_text().splitlines()) if r['arm']=='endpoint_posterior' and r['regime']=='outside')
results=[]
for tick in (1,16,64):
 t=r['trace'][tick];g=ContinuingGame('combat',PROFILES[1],r['turns'],r['seed']%2);w=t['before'];c=g.context(w)
 own=DecisionLoop.from_record(c,t['checkpoint'],predictor=g.actor(c,learning='segmented').predictor);req=g.request(c)
 for name,estimator,recovery in [('sampled','sampled',0),('corrected','conditional',0),('recovery','sampled',2),('both','conditional',2)]:
  start=time.monotonic();cap={};person=copy.deepcopy(own)
  ctrl=owned_rollout(person,observe=g.context,events=g.events,terminal=g.terminal,assess=lambda w:g.goal(w).record(),
   record=lambda w:w,request_for=g.request,feedback=g.feedback,model_id=g.model_id,target=g.target,
   world_model='posterior',purpose_estimator=estimator,tail_recovery=recovery)
  result=DecisionLoop.decide_batch([(person,req)],False,ctrl.planner(person,req,w,captured=cap))[0]
  f=cap['node']['forecast']
  row=dict(tick=tick,variant=name,action=result['decision']['action_id'],purpose={f.roots[k][0]:v for k,v in f.purpose.items()},
   calls=ctrl.frontier.future_calls,seconds=time.monotonic()-start)
  results.append(row);print(json.dumps(row),flush=True)
Path('results/owner_recovery_probe.json').write_text(json.dumps(results,indent=2))
