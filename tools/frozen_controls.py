"""Reuse an entire audited development control cohort without relabeling it.

The prospective plan pins the archive and original proof before new candidate
data. Every old byte, registration and source is retained. This is not fresh or
independent evidence; no candidate execution is read or selected here.
"""
import ast
import io
import json
from pathlib import Path
import zipfile

from tools.owned_checkpoint import encoded, sha

ROOT = Path(__file__).resolve().parents[1]


def function_tree(data, name):
    tree = ast.parse(data.decode('utf-8-sig'))
    return ast.dump(next(n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == name), include_attributes=False)


def load_controls(plan, root=ROOT):
    config = plan.get('frozen_controls')
    if config is None:
        return None
    reference = plan['reference_arm']
    assert reference == config['arm'] == 'stable'
    path = Path(root) / config['archive']
    raw = path.read_bytes()
    assert sha(raw) == config['archive_sha256'], 'control bank changed'
    with zipfile.ZipFile(io.BytesIO(raw)) as bank:
        proof_raw = bank.read('original_cohort/verification.json')
        assert sha(proof_raw) == config['original_proof_sha256']
        proof = json.loads(proof_raw)
        assert proof['complete'] and proof['pairs'] == 10 and proof['actual_steps'] == 2560
        original = {}
        for name in ('plan.json', 'registration.json', 'source.zip'):
            data = bank.read('original_cohort/' + name)
            assert sha(data) == proof['artifacts'][name]
            original[name] = data
        old_plan = json.loads(original['plan.json'])
        registration = json.loads(original['registration.json'])
        assert sha(original['plan.json']) == registration['plan_sha256']
        expected = dict(plan_sha256=registration['plan_sha256'],
            source_sha256=sha(encoded(registration['sources'])))
        assert expected == proof['registration'] == config['registration']
        assert proof['source_ref'] == config['source_ref']
        # Require the whole original cohort, unchanged; no outcome-selected subset.
        assert len(plan['specs']) == 10
        assert sorted(plan['specs'], key=lambda s:s['id']) == sorted(old_plan['specs'], key=lambda s:s['id'])
        with zipfile.ZipFile(io.BytesIO(original['source.zip'])) as sources:
            for name, checksum in registration['sources'].items():
                assert sha(sources.read(name)) == checksum
            for name in ('tools/continuing_games.py', 'tools/preparation_world.py'):
                assert sources.read(name) == (Path(root)/name).read_bytes(), ('different physical rules', name)
            for name, function in (('tools/run_continuation_portfolio.py','game_for'),
                    ('tools/run_owned_value.py','law'), ('tools/run_owned_value.py','cumulative_basis')):
                assert function_tree(sources.read(name),function) == function_tree((Path(root)/name).read_bytes(),function)
        # Preserve the original precision contract; do not invent unrecorded
        # thread settings or claim an old run used the new candidate source.
        assert all(plan['arithmetic'][k] == v for k,v in old_plan['arithmetic'].items())
        files = {}
        for spec in plan['specs']:
            worker = next(w for w in proof['workers'] if w['id'] == spec['id'] and w['arm'] == reference)
            assert worker['steps'] == worker['worker_exact_owner_replay'] == spec['turns'] == 128
            assert worker['physical_and_rng_replayed'] and worker['owner_hash_chain']
            prefix = spec['id'] + '/' + reference + '/'
            for name in ('checkpoint.json', 'result.json', 'status.json'):
                value = bank.read(prefix+name)
                assert sha(value) == worker['hashes'][name]
                files[(spec['id'], name)] = value
            result = json.loads(files[(spec['id'],'result.json')])
            assert result['complete'] and result['configuration']['spec'] == spec
            assert result['configuration']['registration'] == expected
            assert result['runtime'] == config['runtime']
    return dict(registration=expected, proof=registration, source_ref=proof['source_ref'],
        files=files, archive=raw, original_verification=proof_raw, runtime=config['runtime'])


def materialize_controls(plan, destination, root=ROOT):
    """Copy exact old bytes once; a different existing result is an error."""
    controls = load_controls(plan, root)
    if controls is None:
        raise ValueError('explicit prospectively pinned control cohort required')
    for (identity, name), raw in controls['files'].items():
        target = Path(destination)/f'adverse-{identity}-{plan["reference_arm"]}'/name
        if target.exists():
            assert target.read_bytes() == raw, ('existing control differs', str(target))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as output:
                output.write(raw)
    return controls
