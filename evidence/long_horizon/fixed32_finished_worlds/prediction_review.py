"""Compare actually learned response forecasts, without changing any action.

Equal forecasts isolate an input shared by the compared policies, not a complete
causal explanation of their different outcomes. Past actions and resources may
still differ. Known true probabilities are evaluator-only.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--id', required=True)
args = parser.parse_args()
root = Path('results/continuation_horizon_ablation')
sha = lambda b: hashlib.sha256(b).hexdigest()
manifest = json.loads((root/'source_snapshot.json').read_bytes())
for name, expected in manifest.items():
    assert sha((root/'_source'/name).read_bytes()) == expected
sys.path.insert(0, str((root/'_source').resolve()))
from reflex.core import digest, character_key
from reflex.decision_loop import DecisionLoop
from reflex.model_observer import resolve_feedback
from tools.run_continuation_portfolio import game_for

out = Path('evidence/long_horizon/fixed32_finished_worlds')/args.id
proof = json.loads((out/'verification.json').read_bytes())
assert proof['arm_audit_complete']
assert sha(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()) == proof['source']
paths = dict(h16=root/'references'/(args.id+'.json'),
    fixed32=root/args.id/'candidate/result.json',
    consensus=Path('results/continuation_horizon_consensus/worlds')/args.id/'candidate/result.json')
expected = dict(h16=proof['control_result_sha256'], fixed32=proof['result_sha256'],
    consensus=proof['consensus_result_sha256'])
arms = {}
for label, path in paths.items():
    data = path.read_bytes(); assert sha(data) == expected[label]
    row = json.loads(data); assert row['complete']
    game = game_for(row['configuration']['spec'])
    world = game.start(); actor = game.actor(game.context(world), learning='segmented')
    axes = character_key(game.context(world)); observations = []; predictions = []
    for entry in row['trace']:
        assert entry['before'] == world and digest(actor.record()) == entry['before_owner']
        context = game.context(world, actor.state)
        prediction = actor.predictor.predict(context)
        assert digest(actor.record()) == entry['before_owner']
        result = DecisionLoop.decide_batch([(actor, game.request(context))], False,
            _counterfactual_roots=(entry['action'],))[0]
        assert character_key(result['context']) == axes
        after, flow = game.transition(world, entry['action'], entry['response'])
        resolve_feedback(actor, result['ticket'], game.feedback(world, after, entry['action'], context, flow))
        assert after == entry['after'] and digest(actor.record()) == entry['owner']
        predictions.append(dict(tick=entry['tick'], prediction=prediction, truth=entry['truth'],
            action=entry['action'], owner=entry['before_owner']))
        observations.append(entry['response']); world = after
    assert world == row['final'] and game.terminal(world)
    error = sum(sum((x['prediction'][k]-p)**2 for k,p in x['truth'].items()) for x in predictions)/len(predictions)
    changes = [.5*sum(abs(b['prediction'][k]-p) for k,p in a['prediction'].items())
        for a,b in zip(predictions, predictions[1:])]
    arms[label] = dict(input_sha256=sha(data), actual_decisions=len(predictions),
        final_owner=digest(actor.record()), observations=observations, predictions=predictions,
        probability_squared_error=error, mean_temporal_total_variation=sum(changes)/len(changes),
        max_temporal_total_variation=max(changes))
comparisons = []
for other in ('h16', 'consensus'):
    a = arms[other]; b = arms['fixed32']
    assert a['observations'] == b['observations'] and len(a['predictions']) == len(b['predictions'])
    comparisons.append(dict(reference=other, matching_predictions=sum(x['prediction']==y['prediction']
        for x,y in zip(a['predictions'],b['predictions'])), decisions=len(b['predictions']),
        max_probability_difference=max(abs(x['prediction'][k]-v)
            for x,y in zip(a['predictions'],b['predictions']) for k,v in y['prediction'].items())))
result = dict(complete=True, adoption=False, case=args.id, source=proof['source'],
    script_sha256=sha(Path(__file__).read_bytes()), factual_owner_replayed=True,
    arms=arms, comparisons=comparisons, limit=__doc__)
(out/'prediction_review.json').write_text(json.dumps(result,indent=2)+'\n')
(out.parent/'prediction_review.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(comparisons=comparisons, metrics={k:{f:v[f] for f in
    ('probability_squared_error','mean_temporal_total_variation','max_temporal_total_variation')} for k,v in arms.items()})))
