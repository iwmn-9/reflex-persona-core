"""Collect only the complete, registered purpose-tail preparation experiment.

Replay actual learning using its frozen source, retain all raw executions and
the original criterion. This is a regression audit, never adoption evidence.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import uuid
import zipfile


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def collect(root, registration, output):
    root, registration, output = map(lambda p: Path(p).resolve(), (root, registration, output))
    if output.exists():
        raise FileExistsError('retain the existing audit')
    status, summary, plan, registered = [read(p) for p in
        (root/'status.json', root/'summary.json', root/'plan.json', registration)]
    assert status['complete'] and status['state'] == 'complete' and summary['complete']
    assert plan == registered['plan']
    assert sha(root/'plan.json') == registered['plan_sha256'] == summary['plan_sha256']
    assert sha(root/'source_snapshot.json') == registered['source_snapshot_sha256'] == summary['source_snapshot_sha256']
    assert sha(root/'original_plan.json') == plan['original_plan_sha256']
    assert plan['specs'] == read(root/'original_plan.json')['specs']
    assert len(plan['specs']) == len(summary['pairs']) == status['pairs_complete'] == 12
    assert plan['modes'] == ['baseline', 'purpose']
    sources = read(root/'source_snapshot.json')
    assert sources == registered['source_snapshot']
    frozen = root/'_source'
    assert all(sha(frozen/n) == h for n,h in sources.items())
    os.environ.update(plan['arithmetic'])
    sys.path.insert(0, str(frozen))
    spec = importlib.util.spec_from_file_location('frozen_preparation_audit', frozen/'tools/run_purpose_tail_preparation.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = ['plan.json', 'original_plan.json', 'source_snapshot.json', 'summary.json', 'status.json']
    outputs, proofs = {}, []
    for original, retained in zip(plan['specs'], summary['pairs']):
        assert retained['id'] == original['id']
        arms = []
        for mode, expected in zip(plan['modes'], retained['arms']):
            folder = root/original['id']/mode
            actual = module.audit(folder, module.specification(original, mode == 'purpose'))
            assert actual == expected == read(folder/'audit.json')
            assert actual['actual_steps'] == 128 and actual['restored'] == 6
            arms.append(actual)
            outputs[(original['id'], mode)] = read(folder/'result.json')
            files += [f"{original['id']}/{mode}/{n}" for n in ('checkpoint.json', 'result.json', 'audit.json')]
        assert len(arms) == len(retained['arms']) == 2
        assert arms[0]['persona'] == arms[1]['persona']
        assert arms[0]['source_runtime'] == arms[1]['source_runtime']
        delta = arms[1]['score'] - arms[0]['score']
        assert delta == retained['delta']
        proofs.append(dict(id=original['id'], delta=delta, arms=arms))
    prefixes = []
    for original in plan['specs']:
        if not original['id'].endswith('-loss'):
            continue
        stable = original['id'].removesuffix('-loss') + '-stable'
        shock = original['changes'][0][0]
        for mode in plan['modes']:
            left, right = outputs[(stable,mode)]['trace'], outputs[(original['id'],mode)]['trace']
            assert all(all(a[k] == b[k] for k in ('before','before_owner','action'))
                       for a,b in zip(left[:shock], right[:shock]))
            prefixes.append(dict(id=original['id'], mode=mode, identical_pre_reveal_choices=shock))
    assert prefixes == summary['prefixes']
    passed = all(p['delta'] >= -1e-12 for p in proofs)
    assert passed == summary['preparation_no_loss'] == status['preparation_no_loss']
    assert sum(a['actual_steps'] for p in proofs for a in p['arms']) == 3072 == summary['actual_steps'] == status['actual_steps']
    assert all(sha(frozen/n) == h for n,h in sources.items())
    staging = output.with_name(output.name + '.partial-' + uuid.uuid4().hex)
    staging.mkdir(parents=True)
    with zipfile.ZipFile(staging/'executions.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for n in files:
            archive.writestr(n, (root/n).read_bytes())
    with zipfile.ZipFile(staging/'source.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for n in sources:
            archive.writestr(n, (frozen/n).read_bytes())
    (staging/'registration.json').write_bytes(registration.read_bytes())
    (staging/'collector.py').write_bytes(Path(__file__).read_bytes())
    (staging/'summary.json').write_bytes((root/'summary.json').read_bytes())
    hashes = {n:sha(root/n) for n in files}
    with zipfile.ZipFile(staging/'executions.zip') as archive:
        assert set(archive.namelist()) == set(files)
        assert all(hashlib.sha256(archive.read(n)).hexdigest() == h for n,h in hashes.items())
    with zipfile.ZipFile(staging/'source.zip') as archive:
        assert set(archive.namelist()) == set(sources)
        assert all(hashlib.sha256(archive.read(n)).hexdigest() == h for n,h in sources.items())
    proof = dict(complete=True, adoption=False, at=datetime.now(timezone.utc).isoformat(),
        preparation_no_loss=passed, pairs=12, executions=24, actual_steps=3072, restores=144,
        exact_actual_learning_replay=True, fixed_persona=True, hidden_change_prefixes=prefixes,
        plan_sha256=sha(root/'plan.json'), source_snapshot_sha256=sha(root/'source_snapshot.json'),
        sources=len(sources), original_files=hashes,
        artifacts={p.name:sha(p) for p in staging.iterdir()},
        scope='All twelve original preparation regressions, fresh paired controls. No ten-condition development or independent-forty qualification.')
    (staging/'verification.json').write_bytes(encoded(proof))
    staging.rename(output)
    return {k:proof[k] for k in ('complete','adoption','preparation_no_loss','pairs','actual_steps','restores')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','registration','output'):
        parser.add_argument('--'+name, required=True)
    print(json.dumps(collect(**vars(parser.parse_args()))))
