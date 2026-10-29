import copy,json,sys
from pathlib import Path
root,archive,plan_path,out=map(Path,sys.argv[1:]);sys.path.insert(0,str(root.resolve()))
from reflex.core import digest,character_key
from reflex.decision_loop import DecisionLoop
from reflex.model_observer import resolve_feedback
from tools.run_adverse_recovery import game_for
from tools.run_owned_value import law,sample
from tools.owned_checkpoint import encoded,sha
import numpy as np
plan=json.loads(plan_path.read_bytes());results=[]
for spec in plan['specs']:
 path=archive/f"adverse-{spec['id']}-stable"/'checkpoint.json';raw=path.read_bytes();envelope=json.loads(raw);s=envelope['payload'];assert sha(encoded(s))==envelope['sha256']
 game=game_for(spec);world=game.start();actor=game.actor(game.context(world),learning='segmented');rng=np.random.default_rng(spec['seed']);owners=[];mismatches=[]
 for row in s['trace']:
  assert world==row['before'];context=game.context(world,actor.state)
  result=DecisionLoop.decide_batch([(actor,game.request(context))],False,_counterfactual_roots=(row['action'],))[0]
  truth=law(world['tick'],spec['turns'],spec['regime']);assert truth==row['truth'];response=sample(truth,rng.random());assert response==row['response']
  after,flow=game.transition(world,row['action'],response);assert after==row['after']
  resolve_feedback(actor,result['ticket'],game.feedback(world,after,row['action'],context,flow))
  saved=json.loads(encoded(actor.record()));restored=DecisionLoop.from_record(context,saved,predictor=actor.predictor);assert restored.record()==saved;actor=restored
  owners.append(saved)
  if digest(saved)!=row['owner']:mismatches.append(row['tick'])
  assert character_key(game.context(after))==actor.character
  world=after
 assert game.terminal(world) and len(owners)==128 and rng.bit_generator.state==s['rng']
 results.append(dict(id=spec['id'],checkpoint_sha256=sha(raw),owners=owners,historical_hash_mismatches=mismatches,historical_final_owner=s['owner']))
 print(spec['id'],len(mismatches),flush=True)
out.write_bytes(encoded({'root':str(root),'complete':True,'rows':results}))
