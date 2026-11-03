"""Compare registered stable controls in separate, complete source trees.

The child process imports only its declared tree. This catches changes hidden
by loading an old runner into a process that already imported a new core.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def unpack(archive, target):
    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist()) != len(set(z.namelist())):
            raise ValueError('duplicate archive entries')
        for name in z.namelist():
            if not (target / name).resolve().is_relative_to(target):
                raise ValueError('unsafe archive path')
        z.extractall(target)
    return target


def child(source, output, binding=None):
    sys.path.insert(0, str(source))
    from tools.run_adverse_recovery import run, verify_registration
    plan_path = source / 'registered/plan.json'
    registration = verify_registration(plan_path)
    if binding:
        importlib.import_module(binding).activate(plan_path)
    plan = json.loads(plan_path.read_bytes())
    rows = {}
    for family in ('auction', 'combat', 'resources', 'gambling'):
        spec = next(s for s in sorted(plan['specs'], key=lambda s: s['id']) if s['game'] == family)
        folder = output / spec['id']
        run(folder, spec, 'stable', max_decisions=3, registration=registration)
        checkpoint = json.loads((folder / 'checkpoint.json').read_bytes())
        payload = checkpoint['payload']
        assert sha(encoded(payload)) == checkpoint['sha256']
        assert len(payload['trace']) == 3
        rows[family] = {k: payload[k] for k in ('world', 'owner', 'rng', 'trace', 'checks')}
        rows[family]['runtime'] = payload['contract']['runtime']
    verify_registration(plan_path)
    (output / 'summary.json').write_bytes(encoded(rows))


def compare(parent, candidate, output, binding):
    output.mkdir(parents=True, exist_ok=False)
    roots = {label: unpack(path, output / label) for label, path in
             (('parent', parent), ('candidate', candidate))}
    plans = {label: json.loads((root / 'registered/plan.json').read_bytes()) for label, root in roots.items()}
    specs = {label: {s['id']: s for s in plan['specs']} for label, plan in plans.items()}
    assert all(len(specs[label]) == len(plan['specs']) for label, plan in plans.items())
    assert specs['parent'] == specs['candidate']
    results = {}
    for label, root in roots.items():
        target = output / (label + '-results')
        target.mkdir()
        command = [sys.executable, str(Path(__file__).resolve()), '--child', str(root),
                   '--output', str(target)]
        if label == 'candidate':
            command += ['--binding', binding]
        with (output / (label + '.log')).open('w') as log:
            subprocess.run(command, cwd=root, env=dict(os.environ, PYTHONPATH=''),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        results[label] = json.loads((target / 'summary.json').read_bytes())
    assert results['parent'] == results['candidate'], 'registered stable controls changed'
    receipt = dict(returncode=0, complete=True, families=4, actual_steps_per_side=12,
        parent_source_sha256=sha(parent.read_bytes()), candidate_source_sha256=sha(candidate.read_bytes()),
        parent_trace_sha256=sha(encoded(results['parent'])),
        candidate_trace_sha256=sha(encoded(results['candidate'])),
        isolated_complete_sources=True, binding=binding, adoption=False)
    (output / 'receipt.json').write_bytes(encoded(receipt))
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--child', type=Path)
    p.add_argument('--parent', type=Path)
    p.add_argument('--candidate', type=Path)
    p.add_argument('--binding')
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if a.child:
        child(a.child.resolve(), a.output.resolve(), a.binding)
    else:
        if not all((a.parent, a.candidate, a.binding)):
            p.error('parent, candidate and binding required')
        print(json.dumps(compare(a.parent.resolve(), a.candidate.resolve(), a.output.resolve(), a.binding)))
