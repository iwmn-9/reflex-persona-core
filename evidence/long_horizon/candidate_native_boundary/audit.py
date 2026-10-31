from pathlib import Path
import hashlib,json,sys
v=Path(__file__).parent;b=v.parents[2];source=b/'reflex-persona-preview-runtime'
out=b/'reflex-persona-core/evidence/long_horizon/candidate_native_boundary'
if out.exists():raise FileExistsError(out)
sys.path.insert(0,str(source))
from reflex.core import digest
from reflex.fixed_horizon import FixedHorizonOwner
from reflex.native_owned import NativeOwnedBoundary
from reflex.laboratory import PROFILES
from tools.continuing_games import ContinuingGame
from tools.run_adverse_recovery import actor_for
from tools.run_owned_value import cumulative_basis
rows=[]
for family in ('auction','combat','gambling','resources'):
 game=ContinuingGame(family,PROFILES[0],turns=128);world=game.start();context=game.context(world)
 for arm in ('stable','transfer_purpose_gain'):
  actor=actor_for(game,context,arm);before=digest(actor.record())
  template=FixedHorizonOwner(actor,observe=game.context,events=game.events,terminal=game.terminal,
   assess=lambda w:game.goal(w).record(),request_for=game.request,feedback=game.feedback,
   model_id=game.model_id,target=game.target,cumulative_preference=cumulative_basis(game,True))
  accepted=False;error=None
  try:NativeOwnedBoundary(template,world);accepted=True
  except ValueError as failure:error=str(failure)
  assert accepted==(arm=='stable')
  if arm!='stable':assert error=='supported frozen public-conditional owner configuration required'
  assert before==digest(actor.record())
  rows.append(dict(family=family,arm=arm,learner=type(actor.predictor.learner).__name__,accepted=accepted,error=error,owner_unchanged=True))
names=['reflex/native_owned.py','reflex/fixed_horizon.py','tools/template_transfer_prediction.py','tools/run_adverse_recovery.py']
sources={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in names}
proof=dict(complete=True,adoption=False,source='71bf6b2f3e40724a7bb33be14dc3bd3311ee18f8',rows=rows,sources=sources,
 conclusion='The current Rust factual owner supports the stable segmented learner only. The experimental transfer learner is rejected before native export instead of silently losing its behavior. Keep new learner/planner experiments in Python. Passing future world-quality gates does not itself establish Rust equivalence.',
 limits='Configuration/boundary test only. No Rust execution, full native migration or new world-quality claim.')
out.mkdir();(out/'verification.json').write_text(json.dumps(proof,indent=2));(out/'audit.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(cases=len(rows),stable_accepts=4,experimental_rejections=4,owner_unchanged=True)))

