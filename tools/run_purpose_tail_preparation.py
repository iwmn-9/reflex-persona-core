"""Original twelve preparation cases, fresh paired controls and unchanged gates.

Only the candidate's future one-step purpose connection changes. This screen
does not replace the ten development worlds or forty reserved confirmations.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reflex.core import character_key, digest
from reflex.decision_loop import DecisionLoop
from reflex.model_observer import resolve_feedback
from tools.owned_checkpoint import encoded, sha
from tools.run_continuation_portfolio import execution, game_for


def specification(original, candidate):
    out = copy.deepcopy(original)
    if candidate:
        out['controller']['tail_purpose_reflex'] = True
    return out


def audit(folder, spec):
    envelope = json.loads((folder / 'checkpoint.json').read_bytes())
    saved = envelope['payload']
    assert sha(encoded(saved)) == envelope['sha256']
    result = json.loads((folder / 'result.json').read_bytes())
    assert result['complete'] and result['configuration'] == saved['contract']['configuration']
    assert result['configuration']['spec'] == spec and result['configuration']['retry']
    assert result['trace'] == saved['trace'] and len(result['trace']) == spec['turns']
    game = game_for(spec); world = game.start(); context = game.context(world)
    actor = game.actor(context, learning='segmented'); axes = character_key(context)
    rng = np.random.default_rng(spec.get('seed', 0))
    for row in result['trace']:
        assert row['before'] == world and row['before_owner'] == digest(actor.record())
        context = game.context(world, actor.state)
        chosen = DecisionLoop.decide_batch([(actor, game.request(context))], False,
                                           _counterfactual_roots=(row['action'],))[0]
        rng.random()
        assert row['truth'] == {'0': 1.} and row['response'] == '0'
        after, flow = game.transition(world, row['action'], '0')
        for tick, payout in spec['changes']:
            if after['tick'] == tick:
                after['payout'] = payout
        resolve_feedback(actor, chosen['ticket'], game.feedback(world, after, row['action'], context, flow))
        assert after == row['after'] and digest(actor.record()) == row['owner']
        assert character_key(game.context(after)) == axes
        world = after
    assert game.terminal(world) and world == saved['world'] == result['final']
    assert actor.record() == saved['owner'] and rng.bit_generator.state == saved['rng']
    assert result['score'] == game.goal(world).value
    assert result['checks']['owner_parity'] == result['checks']['fixed_axes'] == spec['turns']
    return dict(complete=True, actual_steps=len(result['trace']), score=result['score'],
        source_runtime=saved['contract']['runtime'], owner_replayed=True, persona=axes,
        restored=result['checks']['restored'],
        files={n:sha((folder/n).read_bytes()) for n in ('checkpoint.json', 'result.json')})


def save(path, value):
    temporary = path.with_suffix('.new')
    temporary.write_bytes(encoded(value))
    deadline = time.monotonic() + 2
    while True:
        try:
            temporary.replace(path)
            return
        except OSError as error:
            if getattr(error, 'winerror', None) not in (5, 32) or time.monotonic() >= deadline:
                raise
            time.sleep(.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--frozen', action='store_true')
    args = parser.parse_args(); root = Path(args.root).resolve()
    if not args.frozen:
        original = ROOT / 'evidence/long_horizon/continuation_fixed32_screen/plan.json'
        parent = json.loads(original.read_bytes()); assert len(parent['specs']) == 12
        root.mkdir(parents=True, exist_ok=True)
        if (root / 'plan.json').exists():
            raise FileExistsError('fresh paired preparation registration required')
        (root / 'original_plan.json').write_bytes(original.read_bytes())
        plan = dict(original_plan_sha256=sha(original.read_bytes()), specs=parent['specs'],
            modes=['baseline', 'purpose'], adoption=False,
            criterion='All twelve fresh pairs complete. Every final-score delta >= -1e-12. Exact actual owner and world replay, fixed persona, save restoration, and identical hidden-change prefixes. No independent quality claim.',
            arithmetic={k:os.environ.get(k) for k in ('OPENBLAS_CORETYPE','NPY_DISABLE_CPU_FEATURES','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
            runtime=dict(python=sys.version, numpy=np.__version__),
            source_relation='Same c3a4f10 purpose-tail implementation; only this preparation runner is added.')
        (root / 'plan.json').write_bytes(encoded(plan))
        from tools.freeze_experiment import dispatch
        dispatch(root, Path(__file__).name, [])
        return
    plan = json.loads((root / 'plan.json').read_bytes())
    assert sha((root / 'original_plan.json').read_bytes()) == plan['original_plan_sha256']
    assert plan['specs'] == json.loads((root / 'original_plan.json').read_bytes())['specs']
    assert plan['runtime'] == dict(python=sys.version, numpy=np.__version__)
    assert all(os.environ.get(k) == v for k,v in plan['arithmetic'].items())
    sources = json.loads((root / 'source_snapshot.json').read_bytes())
    assert all(sha((ROOT/n).read_bytes()) == h for n,h in sources.items())
    completed = []; outputs = {}
    for original in plan['specs']:
        pair = []
        for mode in plan['modes']:
            spec = specification(original, mode == 'purpose')
            folder = root / original['id'] / mode
            save(root/'status.json', dict(complete=False, state='running', pid=os.getpid(),
                case=spec['id'], mode=mode, pairs_complete=len(completed), expected_turns=spec['turns'],
                progress='Execution emits actual saved turns every 32 decisions; do not open the live checkpoint.'))
            if not (folder / 'result.json').exists():
                execution(folder, spec, True)
            proof = audit(folder, spec)
            save(folder / 'audit.json', proof)
            pair.append(proof)
            outputs[(spec['id'],mode)] = json.loads((folder/'result.json').read_bytes())
        assert pair[0]['source_runtime'] == pair[1]['source_runtime']
        completed.append(dict(id=original['id'], delta=pair[1]['score']-pair[0]['score'], arms=pair))
        print(encoded(completed[-1]).decode(), flush=True)
    prefixes = []
    for spec in plan['specs']:
        if not spec['id'].endswith('-loss'):
            continue
        normal = spec['id'].removesuffix('-loss') + '-stable'; shock = spec['changes'][0][0]
        for mode in plan['modes']:
            left = outputs[(normal,mode)]['trace']; right = outputs[(spec['id'],mode)]['trace']
            assert all(all(a[k] == b[k] for k in ('before','before_owner','action'))
                       for a,b in zip(left[:shock],right[:shock]))
            prefixes.append(dict(id=spec['id'],mode=mode,identical_pre_reveal_choices=shock))
    assert all(sha((ROOT/n).read_bytes()) == h for n,h in sources.items())
    result = dict(complete=True, pairs=completed, preparation_no_loss=all(p['delta'] >= -1e-12 for p in completed),
        prefixes=prefixes, actual_steps=sum(a['actual_steps'] for p in completed for a in p['arms']),
        source_snapshot_sha256=sha((root/'source_snapshot.json').read_bytes()),
        plan_sha256=sha((root/'plan.json').read_bytes()), adoption=False)
    save(root/'summary.json',result)
    save(root/'status.json',dict(complete=True,state='complete',pairs_complete=12,
        actual_steps=result['actual_steps'],preparation_no_loss=result['preparation_no_loss'],adoption=False))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if '--root' in sys.argv:
            root = Path(sys.argv[sys.argv.index('--root')+1]).resolve()
            if root.exists():
                status = root/'status.json'
                previous = json.loads(status.read_bytes()) if status.exists() else {}
                previous.update(complete=False,state='failed',error=repr(error))
                save(status, previous)
        raise
