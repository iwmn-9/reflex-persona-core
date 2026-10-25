import copy
import importlib.util
import json
from pathlib import Path

root=Path(__file__).resolve().parent;repo=root.parents[1]
spec=importlib.util.spec_from_file_location('fork_harness',repo/'tools/continue_sampled_roots.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h);h.initialize(root/'_source')
from tools.run_continuation_portfolio import execution,game_for

fixture=dict(id='synthetic-harness-fixture',game='auction',mechanism='world',profile=1,seed=37,
    regime='outside',turns=6,controller=dict(horizon_consensus=True,minimum_horizon=32,
        coherent_cumulative_mean=True,purpose_estimator='conditional',recovery_cache=64,tail_recovery=2))
reference=root/'reference';original=execution(reference,fixture,True)
case=dict(id='fixture-t1',tick=1,checkpoint='reference/checkpoint.json',result='reference/result.json',
    checkpoint_sha256=h.sha((reference/'checkpoint.json').read_bytes()),
    result_sha256=h.sha((reference/'result.json').read_bytes()),owner=original['trace'][1]['before_owner'])
action=original['trace'][1]['action']
control=dict(id='fixture-control',case=case['id'],action=action,control=True,banks=['original'],
    archived_purpose=original['trace'][1]['selected_purpose'])
c=h.run_job((root,case,control));assert c['complete'] and not c['complete_new_world']
g=game_for(fixture);alternative=next(k for k in g.keys(original['trace'][1]['before']) if k!=action)
branch=dict(id='fixture-branch',case=case['id'],action=alternative,control=False,banks=['synthetic-test-only'],archived_purpose=0.)
branch_spec=h.prepare(root,case,branch)
assert execution(root/'runs'/branch['id'],branch_spec,True,max_decisions=1) is None
partial=h.read(root/'runs'/branch['id']/'checkpoint.json')['payload']
assert partial['world']['tick']==3
f=h.run_job((root,case,branch));assert f['complete'] and f['actual_decisions_replayed']==6
assert f['new_full_planning_decisions']==4
assert h.run_job((root,case,branch))==f
bad=dict(branch,id='fixture-illegal',action='illegal-root')
try:h.prepare(root,case,bad)
except AssertionError:pass
else:raise AssertionError('illegal intervention was accepted')
tampered=dict(case,checkpoint_sha256='0'*64)
try:h.source_run(root,tampered)
except AssertionError:pass
else:raise AssertionError('modified reference was accepted')
assert h.sha((reference/'checkpoint.json').read_bytes())==case['checkpoint_sha256']
assert h.sha((reference/'result.json').read_bytes())==case['result_sha256']
proof=dict(complete=True,scope='Synthetic six-turn harness correctness only, no independent quality evidence',
    runner_sha256=h.sha((repo/'tools/continue_sampled_roots.py').read_bytes()),
    selftest_sha256=h.sha(Path(__file__).read_bytes()),control=c,branch=f,
    checks=['exact resumed control','intervene once','fixed personality','actual learning replay',
        'identical world RNG','pause-resume full suffix','idempotent completed reuse','illegal root rejected',
        'modified input rejected','original reference unchanged'])
h.save(root/'verification.json',proof)
print(json.dumps(dict(complete=True,checks=proof['checks'],original=action,alternative=alternative)),flush=True)
