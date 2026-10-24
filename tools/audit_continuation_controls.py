"""Audit finished controls without waiting for another candidate's simulation.

Replay factual owners, physics, world RNG and scores with frozen playing code.
This scheduling prerequisite never marks an unfinished paired study complete.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def audit_control(root, spec, manifest):
    from reflex.core import digest, character_key
    from reflex.decision_loop import DecisionLoop
    from reflex.model_observer import resolve_feedback
    from tools.owned_checkpoint import encoded
    from tools.run_continuation_portfolio import game_for
    from tools.run_owned_value import law, sample
    import numpy as np

    directory = root / spec['id'] / 'baseline'
    data = (directory / 'result.json').read_bytes()
    checkpoint = (directory / 'checkpoint.json').read_bytes()
    row = json.loads(data)
    envelope = json.loads(checkpoint)
    saved = envelope['payload']
    assert sha(encoded(saved)) == envelope['sha256']
    assert row['complete'] and row['trace'] == saved['trace'] and row['final'] == saved['world']
    assert row['configuration'] == saved['contract']['configuration']
    assert row['configuration']['spec'] == spec and row['configuration']['retry'] is False
    for name, expected in row['configuration']['sources'].items():
        assert manifest['tools/' + name] == expected
    game = game_for(spec)
    world = game.start()
    actor = game.actor(game.context(world), learning='segmented')
    axes = character_key(game.context(world))
    rng = np.random.default_rng(spec.get('seed', 0))
    increments = []
    for entry in row['trace']:
        assert entry['before'] == world and entry['before_owner'] == digest(actor.record())
        context = game.context(world, actor.state)
        request = game.request(context)
        result = DecisionLoop.decide_batch([(actor, request)], False,
            _counterfactual_roots=(entry['action'],))[0]
        assert character_key(result['context']) == axes
        truth = law(world['tick'], spec['turns'], spec['regime']) if spec['mechanism'] == 'world' else {'0': 1.}
        response = sample(truth, rng.random())
        assert entry['truth'] == truth and entry['response'] == response
        increment = sum(p * (game.goal(game.transition(world, entry['action'], z)[0]).value
            - game.goal(world).value) for z, p in truth.items())
        assert increment == entry['conditional_increment']
        increments.append(increment)
        after, flow = game.transition(world, entry['action'], response)
        for tick, payout in spec.get('changes', []):
            if after['tick'] == tick:
                after['payout'] = payout
        resolve_feedback(actor, result['ticket'], game.feedback(world, after, entry['action'], context, flow))
        assert after == entry['after'] and digest(actor.record()) == entry['owner']
        world = after
    assert row['trace'] and world == row['final'] and game.terminal(world)
    assert actor.record() == saved['owner'] and rng.bit_generator.state == saved['rng']
    assert row['score'] == game.goal(world).value
    assert row['physical_replay'] == len(row['trace']) <= spec['turns']
    return dict(id=spec['id'], result_sha256=sha(data), checkpoint_sha256=sha(checkpoint),
        actual_decisions=len(row['trace']), conditional_total=sum(increments), runtime=row['runtime'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--expected-plan', required=True)
    parser.add_argument('--expected-source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    manifest = json.loads((root / 'source_snapshot.json').read_bytes())
    for name, value in manifest.items():
        assert sha((root / '_source' / name).read_bytes()) == value
    sys.path.insert(0, str(root / '_source'))
    from tools.owned_checkpoint import encoded
    assert sha(encoded(manifest)) == args.expected_source
    assert sha((root / 'plan.json').read_bytes()) == args.expected_plan
    plan = json.loads((root / 'plan.json').read_bytes())
    assert plan['source'] == args.expected_source
    assert len({s['id'] for s in plan['specs']}) == len(plan['specs']) > 0
    rows = [audit_control(root, spec, manifest) for spec in plan['specs']]
    out = dict(complete=True, controls=len(rows), source=plan['source'],
        plan_sha256=args.expected_plan, auditor_sha256=sha(Path(__file__).read_bytes()),
        actual_owner_replayed=True, source_verified=True, rows=rows,
        scope='Finished controls only. Candidate and paired-study completion remain separate.')
    Path(args.output).write_bytes(encoded(out))
    print(json.dumps(dict(controls=len(rows), actual_decisions=sum(r['actual_decisions'] for r in rows))))


if __name__ == '__main__':
    main()
