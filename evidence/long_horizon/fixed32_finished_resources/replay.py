"""Read-only replay of the newly finished arm; never completes its eight-pair study."""
import hashlib
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import zipfile

base = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd().resolve()
assert (base / 'reflex/core.py').is_file(), 'Run from the repository or supply its path.'
root = base / 'results/continuation_horizon_ablation'
out = base / 'evidence/long_horizon/fixed32_finished_resources'
sha = lambda b: hashlib.sha256(b).hexdigest()
manifest = json.loads((root / 'source_snapshot.json').read_bytes())
for name, expected in manifest.items():
    assert sha((root / '_source' / name).read_bytes()) == expected
sys.path.insert(0, str(root / '_source'))
from tools.owned_checkpoint import encoded
from tools.run_continuation_portfolio import game_for
from tools.run_owned_value import law, sample
from reflex.core import digest, character_key
from reflex.decision_loop import DecisionLoop
from reflex.model_observer import resolve_feedback
import numpy as np

source = '1f01060ee8d647e829a479083efc7faf20db2b6c1184e1864c9f6036de5a892b'
plan_bytes = (root / 'plan.json').read_bytes()
plan_sha = '0b8607aa40f2496048acfe03f5640796871b39c9c03c7b63eb829e497e848173'
assert sha(encoded(manifest)) == source and sha(plan_bytes) == plan_sha
plan = json.loads(plan_bytes)
assert plan['source'] == source and len(plan['specs']) == 8
spec = next(s for s in plan['specs'] if s['id'] == 'resources-2-56101-outside')
directory = root / spec['id'] / 'candidate'
data = (directory / 'result.json').read_bytes()
checkpoint = (directory / 'checkpoint.json').read_bytes()
row = json.loads(data)
envelope = json.loads(checkpoint)
saved = envelope['payload']
assert sha(encoded(saved)) == envelope['sha256']
assert row['complete'] and row['trace'] == saved['trace'] and row['final'] == saved['world']
assert row['configuration'] == saved['contract']['configuration']
assert row['configuration']['spec'] == spec and row['configuration']['retry'] is True
for name, expected in row['configuration']['sources'].items():
    assert manifest['tools/' + name] == expected
game = game_for(spec)
world = game.start()
actor = game.actor(game.context(world), learning='segmented')
axes = character_key(game.context(world))
rng = np.random.default_rng(spec['seed'])
increments = []
for entry in row['trace']:
    assert entry['before'] == world and entry['before_owner'] == digest(actor.record())
    context = game.context(world, actor.state)
    result = DecisionLoop.decide_batch([(actor, game.request(context))], False,
        _counterfactual_roots=(entry['action'],))[0]
    assert character_key(result['context']) == axes
    truth = law(world['tick'], spec['turns'], spec['regime'])
    response = sample(truth, rng.random())
    assert entry['truth'] == truth and entry['response'] == response
    increment = sum(p * (game.goal(game.transition(world, entry['action'], z)[0]).value
        - game.goal(world).value) for z, p in truth.items())
    assert increment == entry['conditional_increment']
    increments.append(increment)
    after, flow = game.transition(world, entry['action'], response)
    resolve_feedback(actor, result['ticket'], game.feedback(world, after, entry['action'], context, flow))
    assert after == entry['after'] and digest(actor.record()) == entry['owner']
    world = after
assert row['physical_replay'] == len(row['trace']) == spec['turns'] == 128
assert world == row['final'] and game.terminal(world)
assert actor.record() == saved['owner'] and rng.bit_generator.state == saved['rng']
assert row['score'] == game.goal(world).value
control_data = (root / spec['reference']['archive']).read_bytes()
control = json.loads(control_data)
receipt = next(r for r in json.loads((root / 'controls_verification.json').read_bytes())['rows'] if r['id'] == spec['id'])
assert sha(control_data) == spec['reference']['sha256'] == receipt['result_sha256']
assert control['complete'] and control['runtime'] == row['runtime']
consensus_file = base / 'results/continuation_horizon_consensus/worlds' / spec['id'] / 'candidate/result.json'
consensus_data = consensus_file.read_bytes()
consensus = json.loads(consensus_data)
assert consensus['complete'] and consensus['runtime'] == row['runtime']
for path, blob in [(directory / 'result.json', data), (directory / 'checkpoint.json', checkpoint)]:
    assert path.read_bytes() == blob
proof = dict(arm_audit_complete=True, whole_study_complete=False, adoption=False,
    checked_at=datetime.now(timezone.utc).isoformat(), id=spec['id'], registered_pairs=8,
    scope='This completed arm only. Whole-cohort forecasts, final comparison and independent40 still required.',
    source=source, plan_sha256=plan_sha, replay_script_sha256=sha(Path(__file__).read_bytes()),
    actual_decisions=128, actual_owner_replayed=True, personality_axes_unchanged=True,
    rng_replayed=True, source_verified=True, result_sha256=sha(data), checkpoint_sha256=sha(checkpoint),
    control_result_sha256=sha(control_data), consensus_result_sha256=sha(consensus_data),
    scores=dict(h16=control['score'], fixed32=row['score'], consensus=consensus['score']),
    cash=dict(h16=control['final']['cash'], fixed32=row['final']['cash'], consensus=consensus['final']['cash']),
    conditional_delta=sum(increments)-receipt['conditional_total'],
    actions={label:dict(Counter(e['action'] for e in x['trace'])) for label,x in [('h16',control),('fixed32',row),('consensus',consensus)]})
out.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(out / 'inputs.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for name, blob in [('candidate_result.json', data), ('candidate_checkpoint.json', checkpoint),
        ('control_result.json', control_data), ('consensus_result.json', consensus_data),
        ('plan.json', plan_bytes), ('source_snapshot.json', (root / 'source_snapshot.json').read_bytes()),
        ('controls_verification.json', (root / 'controls_verification.json').read_bytes())]:
        archive.writestr(name, blob)
proof['inputs_sha256'] = sha((out / 'inputs.zip').read_bytes())
(out / 'verification.json').write_bytes(encoded(proof))
(out / 'replay.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(proof, indent=2))
