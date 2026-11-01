"""Explain a cross-platform replay mismatch; never replace the original audit."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reflex.core import character_key, digest
from reflex.decision_loop import DecisionLoop
from reflex.model_observer import resolve_feedback
from tools.probe_adverse_platform_replay import differences
from tools.run_adverse_recovery import game_for
from tools.run_owned_value import law, sample
from tools.owned_checkpoint import encoded
import numpy as np


def diagnose(history, checkpoints, identifier, candidate, restore_cadence=False):
    with gzip.open(history, 'rt') as stream:
        row = next(r for r in map(json.loads, stream)
                   if r['spec']['id'] == identifier and r['retry'] is candidate)
    arm = 'candidate' if candidate else 'baseline'
    with zipfile.ZipFile(checkpoints) as archive:
        saved = json.loads(archive.read(f'{identifier}/{arm}/checkpoint.json'))
    assert hashlib.sha256(encoded(saved['payload'])).hexdigest() == saved['sha256']
    saved = saved['payload']
    assert saved['trace'] == row['result']['trace']
    spec = row['spec']; game = game_for(spec); world = game.start()
    actor = game.actor(game.context(world), learning='segmented')
    axes = character_key(game.context(world)); rng = np.random.default_rng(spec['seed'])
    mismatches = []
    for t in row['result']['trace']:
        assert world == t['before'], ('public world before', t['tick'])
        context = game.context(world, actor.state)
        result = DecisionLoop.decide_batch([(actor, game.request(context))], False,
                                           _counterfactual_roots=(t['action'],))[0]
        truth = law(world['tick'], spec['turns'], spec['regime'])
        assert truth == t['truth'] and sample(truth, rng.random()) == t['response']
        after, flow = game.transition(world, t['action'], t['response'])
        resolve_feedback(actor, result['ticket'], game.feedback(world, after, t['action'], context, flow))
        if restore_cadence and world['tick'] % 23 == 0:
            record = json.loads(encoded(actor.record()))
            actor = DecisionLoop.from_record(context, record, predictor=actor.predictor)
            assert actor.record() == record
        assert after == t['after'], ('public world after', t['tick'])
        assert character_key(game.context(after)) == axes
        if digest(actor.record()) != t['owner']:
            mismatches.append(t['tick'])
        world = after
    assert world == saved['world'] and rng.bit_generator.state == saved['rng']
    delta = differences(actor.record(), saved['owner'])
    floats = [d for d in delta if type(d['local']) is float and type(d['saved']) is float]
    return dict(id=identifier, arm=arm, steps=len(row['result']['trace']),
        physical_rng_personality_exact=True,
        restore_cadence=23 if restore_cadence else None,
        exact_owner_hashes=len(row['result']['trace'])-len(mismatches),
        first_owner_hash_mismatch=mismatches[0] if mismatches else None,
        final_owner_equal=not delta, float_differences=len(floats),
        non_float_differences=[d for d in delta if d not in floats],
        max_absolute_float_difference=max((abs(d['local']-d['saved']) for d in floats), default=0.),
        float_fields=sorted({d['path'].split('/')[1] for d in floats}),
        largest_float_differences=sorted(floats,key=lambda d:abs(d['local']-d['saved']),reverse=True)[:5])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history', type=Path, required=True)
    p.add_argument('--checkpoints', type=Path, required=True)
    p.add_argument('--id', required=True)
    p.add_argument('--candidate', action='store_true')
    p.add_argument('--restore-cadence', action='store_true',
                   help='Replay the original JSON restore after ticks divisible by 23')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    manifest = {str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in (a.history, a.checkpoints)}
    with a.output.open('x', encoding='utf-8') as stream:
        result = dict(complete=False, adoption=False, original_exact_audit_passed=False,
            runtime=dict(python=platform.python_version(), system=platform.system(), numpy=np.__version__),
            input_sha256=manifest)
        try:
            result['comparison'] = diagnose(a.history, a.checkpoints, a.id, a.candidate, a.restore_cadence)
            assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest() == h for f,h in manifest.items())
            result['complete'] = True
        finally:
            json.dump(result, stream, ensure_ascii=False, allow_nan=False)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
