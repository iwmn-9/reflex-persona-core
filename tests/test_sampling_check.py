import copy,sys,unittest
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from reflex.sampling_check import SampleBanks,DecisionSampling,checked_decide
from reflex.closed_loop import owned_rollout
from reflex.receding_owner import RecedingOwner
from reflex.decision_loop import DecisionLoop
from reflex.laboratory import PROFILES
from tools.continuing_games import ContinuingGame,GAMES
from reflex.cumulative_preference import CumulativePreference


class SamplingCheckTests(unittest.TestCase):
 def test_cached_pools_and_commit_match_fresh_independent_paths(self):
  for game in GAMES:
   g=ContinuingGame(game,PROFILES[0],3);w=g.start();own=g.actor(g.context(w),learning='segmented')
   options=dict(observe=g.context,events=g.events,terminal=g.terminal,assess=lambda w:g.goal(w).record(),
      record=lambda w:w,request_for=g.request,feedback=g.feedback,model_id=g.model_id,target=g.target,
      tail_steps=2,seeds=(701,),world_model='posterior',purpose_estimator='conditional',tail_recovery=2,tail_goal_refinement=True,
      cumulative_preference=CumulativePreference(g.target,'per-decision',(('values.achievement',float(g.turns)),)))
   loop=RecedingOwner(own,**options);spec=DecisionSampling(((701,),(711,),(721,)));saved=copy.deepcopy(own.record())
   banks=SampleBanks(own,loop.options,spec)
   banks.evaluate(w,(0,));banks.evaluate(w,(1,));pooled=banks.evaluate(w,(0,1));self.assertGreater(banks.hits,0)
   reference=copy.deepcopy(own);ctrl=owned_rollout(reference,**dict(options,seeds=(701,711)))
   req=g.request(g.context(w,reference.state));capture={}
   fresh=DecisionLoop.decide_batch([(reference,req)],False,ctrl.planner(reference,req,w,captured=capture))[0]
   self.assertEqual(pooled['captured']['node']['forecast'],capture['node']['forecast'])
   self.assertEqual(pooled['result']['decision'],fresh['decision']);self.assertEqual(own.record(),saved)
   loop=RecedingOwner(own,**options,sampling_check=spec)
   actual=loop.decide(w);self.assertIn(actual['sampling_check']['selected_samples'],(2,3))
   k=actual['decision']['action_id'];after,flow=g.transition(w,k,'2');loop.resolve(actual['ticket'],g.feedback(w,after,k,g.context(w),flow))
   with self.assertRaisesRegex(ValueError,'real feedback'):banks.evaluate(after,(0,1))

 def test_invalid_or_overlapping_banks_and_incompatible_modes_are_rejected(self):
  for banks in (((1,),(1,),(2,)),((True,),(2,),(3,)),((1,),(2,)),((1,),(),(3,)),
                ((-1,),(2,),(3,)),(tuple(range(12)),tuple(range(12,24)),tuple(range(24,36)))):
   with self.assertRaises(ValueError):DecisionSampling(banks)
  for limit in (0,4097,True):
   with self.assertRaises(ValueError):DecisionSampling(cache_entries=limit)
  g=ContinuingGame('gambling',PROFILES[0],3);own=g.actor(g.context(g.start()),learning='segmented')
  options=dict(observe=g.context,events=g.events,terminal=g.terminal,assess=lambda w:g.goal(w).record(),
      record=lambda w:w,request_for=g.request,feedback=g.feedback,model_id=g.model_id,target=g.target)
  for extra in ({'seeds':(999,)},{'tail_steps':0},{'sampling_standard_errors':2.}):
   with self.assertRaises(ValueError):RecedingOwner(own,**options,**extra,sampling_check=DecisionSampling())

 def test_expansion_is_triggered_by_disagreement_not_majority_voting(self):
  from unittest.mock import patch
  from types import SimpleNamespace
  from reflex.core import digest
  from reflex.examples import context,action,effect
  from reflex.deliberation import JointForecast
  from reflex.decision_loop import Request
  from reflex.judgment import Binding
  from reflex.core import FEATURES
  c=context('sampling-stage',[action(k,effect()) for k in ('a','b')])
  own=DecisionLoop(c);req=Request(c,{k:Binding(k,'case',FEATURES) for k in ('a','b')},{k:() for k in ('a','b')})
  def evaluate(self,public,indices):
   key='b' if indices in ((1,),(0,1)) else 'a'
   reference=copy.deepcopy(own);captured={}
   fc=copy.deepcopy(c);fc['actions']=[a for a in fc['actions'] if a['id']==key]
   f=JointForecast((fc,),{key:(key,)},{key:0.},1,max_regret=2.)
   def plan(contexts,decisions):captured['inputs']=digest([contexts,decisions]);return f
   result=DecisionLoop.decide_batch([(reference,req)],False,plan)[0]
   return dict(result=result,captured={'node':{'forecast':f}},inputs=captured['inputs'],seeds=tuple(701+i for i in indices))
  controller=SimpleNamespace(actor=own,options={},request_for=lambda c:req,observe=lambda w,state:copy.deepcopy(c),last=None)
  def initialize(self,owner,options,spec):self.before=digest(owner.record());self.hits=0;self.misses=0
  with patch.object(SampleBanks,'__init__',initialize),patch.object(SampleBanks,'evaluate',evaluate) as _:
   result=checked_decide(controller,{},DecisionSampling())
  self.assertEqual(result['decision']['action_id'],'a')
  self.assertTrue(result['sampling_check']['expanded'])
  self.assertEqual([r['action'] for r in result['sampling_check']['stages']],['a','b','b','a'])

 def test_sampling_configuration_and_feedback_survive_frozen_resume(self):
  import json,tempfile,subprocess,hashlib
  from tools.freeze_experiment import freeze
  from tools.resume_horizon_study import resume
  from tools.run_owned_value import run
  job=['gambling',0,55991,'outside',2,1,1,0]
  expected=run(job,'receding_recovery',sampling_check=True,cumulative_achievement=True,tail_goal_refinement=True)
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory)/'execution';freeze(root)
   program='''import sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from tools.run_owned_value import run
from tools.run_horizon_study import __doc__ as semantics
root=Path(sys.argv[2]);job=json.loads(sys.argv[3]);options=dict(sampling_check=True,cumulative_achievement=True,tail_goal_refinement=True)
(root/'registration.json').write_text(json.dumps(dict(jobs=[job],arms=['receding_recovery'],world_seeds=[job[2]],stage='diagnostic',semantics=semantics,configuration=options)))
assert run(job,'receding_recovery',checkpoint=root/'checkpoint.json',max_decisions=1,**options) is None
'''
   subprocess.run([sys.executable,'-c',program,str(root/'_source'),str(root),json.dumps(job)],check=True,capture_output=True)
   proof=resume(root,hashlib.sha256((root/'source_snapshot.json').read_bytes()).hexdigest(),job)
   self.assertTrue(proof['original_prefix_unchanged'])
   actual=json.loads((root/'trajectories.jsonl').read_text());actual.pop('seconds');expected.pop('seconds')
   self.assertEqual(actual,expected)

 def test_cached_worlds_are_distinct_and_returned_paths_cannot_mutate_cache(self):
  g=ContinuingGame('gambling',PROFILES[0],3);w=g.start();own=g.actor(g.context(w),learning='segmented')
  options=dict(observe=g.context,events=g.events,terminal=g.terminal,assess=lambda w:g.goal(w).record(),
      record=lambda w:w,request_for=g.request,feedback=g.feedback,model_id=g.model_id,target=g.target,
      tail_steps=1,seeds=(701,),world_model='posterior')
  banks=SampleBanks(own,options,DecisionSampling(((701,),(711,),(721,)),cache_entries=1))
  changed=banks._distribution(w,own,(0,));original=copy.deepcopy(changed)
  changed.trajectories[0][1][0]['cost']=.99
  self.assertEqual(banks._distribution(w,own,(0,)),original)
  other=dict(w,cash=9.)
  self.assertNotEqual(banks._distribution(other,own,(0,)),original)
  self.assertEqual(banks._distribution(w,own,(0,)),original)
  self.assertEqual(banks.misses,3)

 def test_failure_before_commit_preserves_factual_owner(self):
  from unittest.mock import patch
  g=ContinuingGame('gambling',PROFILES[0],3);own=g.actor(g.context(g.start()),learning='segmented')
  loop=RecedingOwner(own,observe=g.context,events=g.events,terminal=g.terminal,assess=lambda w:g.goal(w).record(),
      record=lambda w:w,request_for=g.request,feedback=g.feedback,model_id=g.model_id,target=g.target,
      tail_steps=2,sampling_check=DecisionSampling())
  saved=copy.deepcopy(own.record())
  with patch.object(SampleBanks,'evaluate',side_effect=ValueError('incomplete bank')):
   with self.assertRaisesRegex(ValueError,'incomplete bank'):loop.decide(g.start())
  self.assertEqual(own.record(),saved);self.assertIsNone(own.pending);self.assertIsNone(loop.last)

 def test_explicit_baseline_horizon_is_preserved_in_pair_jobs(self):
  from tools.value_contract_config import jobs
  spec={'job':['auction',0,55992,'outside',128,1,31,0]}
  original=copy.deepcopy(spec)
  paired=jobs(spec,{'horizons':{'baseline':16,'candidate':32}})
  self.assertEqual(paired['baseline'][6],15);self.assertEqual(paired['candidate'][6],31)
  self.assertEqual(spec,original)
  self.assertEqual(jobs(spec,{}),{'baseline':spec['job'],'candidate':spec['job']})
  with self.assertRaises(ValueError):jobs(spec,{'horizons':{'baseline':16,'candidate':64}})


if __name__=='__main__':unittest.main(verbosity=2)
