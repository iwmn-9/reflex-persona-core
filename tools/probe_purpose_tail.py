"""Frozen, same-owner comparison of the future-purpose connection at a known root."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(directory):
    root = Path(directory).resolve()
    manifest = json.loads((root / 'manifest.json').read_bytes())
    worker = root / 'source'
    assert not (root / 'comparison.json').exists()
    for name, expected in manifest['sources'].items():
        assert sha(worker / name) == expected, ('source changed', name)
    assert sha(root / 'input.json') == manifest['input_sha256']
    assert sha(root / 'original_probe.json') == manifest['original_probe_sha256']
    original = json.loads((root / 'input.json').read_bytes())
    probe = json.loads((root / 'original_probe.json').read_bytes())
    assert original['complete'] and original['configuration']['arm'] == 'transfer'
    assert original['configuration']['registration'] == manifest['original_registration']
    os.environ.update(manifest['arithmetic'])
    os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    sys.path.insert(0, str(worker))
    import numpy as np
    from reflex.core import digest, character_key
    from reflex.decision_loop import DecisionLoop
    from reflex.model_observer import resolve_feedback
    from tools.run_adverse_recovery import actor_for, decide, game_for
    from tools.run_owned_value import law, sample
    runtime = dict(python=sys.version, numpy=np.__version__, system=platform.system(), machine=platform.machine())
    assert runtime == probe['runtime'] == original['runtime']
    spec = original['configuration']['spec']; game = game_for(spec)
    world = game.start(); owner = actor_for(game, game.context(world), 'transfer')
    rng = np.random.default_rng(spec['seed'])
    for row in original['trace'][:manifest['tick']]:
        assert world == row['before'] and digest(owner.record()) == row['before_owner']
        context = game.context(world, owner.state)
        decision = DecisionLoop.decide_batch([(owner, game.request(context))], False,
                                            _counterfactual_roots=(row['action'],))[0]
        truth = law(world['tick'], spec['turns'], spec['regime'])
        assert truth == row['truth'] and sample(truth, rng.random()) == row['response']
        after, flow = game.transition(world, row['action'], row['response'])
        resolve_feedback(owner, decision['ticket'], game.feedback(world, after, row['action'], context, flow))
        assert after == row['after'] and digest(owner.record()) == row['owner']
        world = after
    assert digest(owner.record()) == probe['factual_owner']
    assert world == original['trace'][manifest['tick']]['before']
    source_owner = digest(owner.record()); persona = character_key(game.context(world)); rows = []
    for arm in ('transfer', 'transfer_purpose'):
        actor = copy.deepcopy(owner); started = time.monotonic()
        result, cap = decide(actor, game, copy.deepcopy(world), arm, seeds=tuple(manifest['seeds']))
        f = cap['forecast']
        assert character_key(result['context']) == persona
        if arm == 'transfer':
            assert result['decision'] == probe['decision']
            assert result['deliberation'] == probe['deliberation']
            assert f.purpose == probe['purpose_by_plan']
        row = dict(arm=arm, factual_owner=source_owner, decision=result['decision'],
            deliberation=result['deliberation'], purpose_by_plan=f.purpose, roots=f.roots,
            modeled=cap['modeled'], simulated_steps=cap['simulated_steps'],
            seconds=time.monotonic()-started)
        with (root / (arm + '.json')).open('x', encoding='utf-8') as stream:
            json.dump(row, stream, sort_keys=True, allow_nan=False)
        rows.append(row)
        print(json.dumps(dict(arm=arm, action=result['decision']['action_id'],
            purpose=result['deliberation']['selected_purpose'], seconds=row['seconds'])), flush=True)
    assert digest(owner.record()) == source_owner
    for name, expected in manifest['sources'].items():
        assert sha(worker / name) == expected
    receipt = dict(complete=True, adoption=False, runtime=runtime, source_ref=manifest['source_ref'],
        manifest_sha256=sha(root/'manifest.json'), factual_owner=source_owner,
        original_forecast_exact=True, persona_unchanged=True, source_unchanged=True,
        results={arm:sha(root/(arm+'.json')) for arm in ('transfer','transfer_purpose')},
        scope='Previously exposed factual root; neither new actual-world quality nor independent confirmation.')
    with (root / 'comparison.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, sort_keys=True, allow_nan=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    main(parser.parse_args().root)
