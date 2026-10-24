"""Isolated wheel lifecycle/host save smoke test; not long-run quality evidence.

Run with python -I tools/verify_receding_package.py <isolated-install-directory>.
The optional second argument is an internal fresh-process resume bundle.
"""
import sys, json, copy, subprocess, tempfile
from pathlib import Path
from datetime import datetime, timezone
installed=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(installed))
class BlockResearch:
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='tools' or fullname.startswith('tools.'):
   raise ImportError('Package smoke forbids research tools')
sys.meta_path.insert(0,BlockResearch())
import numpy as np
import reflex
from reflex.examples import context,action,effect
from reflex.core import FEATURES,digest
from reflex.decision_loop import DecisionLoop,Request
from reflex.event_reader import EventReader
from reflex.judgment import Binding,ReadControl
from reflex.receding_owner import RecedingOwner
from reflex.adaptive_dynamics import PublicEvent
from reflex.model_observer import ModelFeedback
from reflex.purpose_plan import Goal
from reflex.planning import vector
assert Path(reflex.__file__).resolve().is_relative_to(installed)
keys=('a','b')
def outcome(c,k,z):return effect(.1 if k==z else -.05,values={'achievement':.1 if k==z else -.05})
def observe(w,state=None):
 c=context('package-smoke',[action(k,*[dict(outcome(None,k,z),p=.5) for z in keys]) for k in keys] if w['tick']<4 else [],values={'achievement':.7})
 c.update(tick=w['tick'],facts={'world':json.dumps(w,sort_keys=True)})
 if state is not None:c['state']=copy.deepcopy(state)
 return c
def models(c):return {'left':dict(a=.9,b=.1),'right':dict(a=.1,b=.9)}
def request(c):return Request(c,{k:Binding(k,'same',FEATURES) for k in keys},{k:() for k in keys},threatened=True)
def transition(w,k,z):
 f=outcome(None,k,z)
 return dict(tick=w['tick']+1,score=w['score']+f['objective'],signal=z),f
def events(w,k):return [PublicEvent(z,*transition(w,k,z)) for z in keys]
def feedback(w,n,k,c,f):return ModelFeedback(None,observed=vector(f),event={'revealed_action':n['signal']})
def make():
 c=observe(dict(tick=0,score=0.,signal='a'))
 reader=EventReader(c['scope'],('left','right'),models,outcome,legal_for=lambda c:keys,model_id='package-rules-v1',known_conditionals=True,learning='segmented')
 actor=DecisionLoop(c,predictor=reader,read_control=ReadControl(max_nodes=64))
 return RecedingOwner(actor,observe=observe,events=events,terminal=lambda w:w['tick']>=4,assess=lambda w:Goal('smoke-score','scored' if w['tick']>=4 else 'running',w['score']).record(),record=lambda w:w,request_for=request,feedback=feedback,model_id='package-rules-v1',target='smoke-score',tail_steps=1,seeds=(701,),world_model='posterior',purpose_estimator='conditional')
def symbol(w,rng):
 # Only actual execution sees the changing world law and factual RNG.
 return 'a' if rng.random() < (.8 if w['tick']<2 else .2) else 'b'
if len(sys.argv)>2:
 bundle=json.loads(Path(sys.argv[2]).read_bytes());w=bundle['world']
 owner=make().restored(w,bundle['owner']);rng=np.random.default_rng()
 rng.bit_generator.state=bundle['rng'];trace=bundle['trace']
 while w['tick']<4:
  c=observe(w,owner.actor.state);result=owner.decide(w);k=result['decision']['action_id']
  n,f=transition(w,k,symbol(w,rng));owner.resolve(result['ticket'],feedback(w,n,k,c,f))
  w=n;trace.append(k)
 final=make().restored(w,owner.checkpoint(w))
 assert final.actor.record()==owner.actor.record()
 print(json.dumps(dict(world=w,rng=rng.bit_generator.state,owner=owner.actor.record(),trace=trace)))
 sys.exit(0)
w=dict(tick=0,score=0.,signal='a');a=make();b=make();trace=[];rng=np.random.default_rng(987)
resume_bundle=None
for tick in range(4):
 c=observe(w,a.actor.state);x=a.decide(w);y=b.decide(w)
 assert x['decision']==y['decision']
 k=x['decision']['action_id'];n,f=transition(w,k,symbol(w,rng))
 fb=feedback(w,n,k,c,f);a.resolve(x['ticket'],fb);b.resolve(y['ticket'],fb)
 assert a.actor.record()==b.actor.record()
 w=n;trace.append(k)
 saved=json.loads(json.dumps(b.checkpoint(w)))
 b=make().restored(w,saved)
 assert a.actor.record()==b.actor.record()
 if w['tick']==2:
  resume_bundle=json.loads(json.dumps(dict(world=w,rng=rng.bit_generator.state,owner=saved,trace=trace)))
try:b.decide(w)
except ValueError:pass
else:raise AssertionError('terminal execution accepted')
with tempfile.TemporaryDirectory(prefix='reflex-package-restore-') as folder:
 path=Path(folder)/'host-save.json';path.write_text(json.dumps(resume_bundle),encoding='utf-8')
 process=subprocess.run([sys.executable,'-I',str(Path(__file__).resolve()),str(installed),str(path)],
                        cwd=folder,text=True,capture_output=True,check=True)
resumed=json.loads(process.stdout)
assert resumed==json.loads(json.dumps(dict(world=w,rng=rng.bit_generator.state,owner=b.actor.record(),trace=trace)))
print(json.dumps(dict(success=True,at=datetime.now(timezone.utc).isoformat(),imported_from=str(Path(reflex.__file__).resolve()),
 no_research_imports=True,actual_steps=4,restores=4,process_restart_at=2,world_and_rng_equal=True,
 owner_equal=True,all_decisions_equal=True,terminal_restored=True,trace=trace,owner_sha256=digest(b.actor.record()))))
