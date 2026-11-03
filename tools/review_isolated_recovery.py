"""Replay each cohort with its own frozen core, then compare unchanged gates.

Use when a research candidate changes a common implementation file. Such changes
must be enumerated in the plan; they never waive parent controls or replay.
"""
import argparse
from collections import defaultdict
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from check_registered_parent import encoded, sha, unpack


def source(archive):
    with zipfile.ZipFile(archive) as z:
        raw = z.read('registered/plan.json')
        plan = json.loads(raw)
        manifest = json.loads(z.read('registered/registration.json'))
        assert sha(raw) == manifest['plan_sha256']
        for name, value in manifest['sources'].items():
            assert sha(z.read(name)) == value
    return plan, manifest


def contract(parent, candidate, parity):
    pp, pm = source(parent)
    cp, cm = source(candidate)
    assert sha(parent.read_bytes()) == cp['parent_source_sha256']
    assert pm['plan_sha256'] == cp['parent_plan_sha256']
    assert sha(encoded(pm)) == cp['parent_registration_sha256']
    assert {s['id']: s for s in pp['specs']} == {s['id']: s for s in cp['specs']}
    assert len(pp['specs']) == len(cp['specs']) == 10
    for key in ('criteria', 'independent_confirmation', 'concordant_adverse',
                'arithmetic', 'completion_contract_sha256'):
        assert pp[key] == cp[key]
    changed = cp.get('candidate_source_changes', {})
    common = cp['common_sources']
    # The original common set is every core file plus the declared common tools.
    required = {n for n in pm['sources'] if n.startswith('reflex/')}
    required |= {'tools/run_owned_value.py', 'tools/continuing_games.py', 'tools/owned_checkpoint.py',
                 'tools/run_continuation_portfolio.py', 'tools/preparation_world.py'}
    assert set(common).isdisjoint(changed)
    assert set(common) | set(changed) == required
    for name, value in common.items():
        assert pm['sources'][name] == cm['sources'][name] == value
    for name, values in changed.items():
        assert values == dict(parent_sha256=pm['sources'][name], candidate_sha256=cm['sources'][name])
        assert values['parent_sha256'] != values['candidate_sha256']
    assert parity['returncode'] == 0 and parity['complete']
    assert parity['isolated_complete_sources'] and parity['families'] == 4
    assert parity['actual_steps_per_side'] == 12
    assert parity['parent_source_sha256'] == sha(parent.read_bytes())
    assert parity['candidate_source_sha256'] == sha(candidate.read_bytes())
    assert parity['parent_trace_sha256'] == parity['candidate_trace_sha256']
    return cp


def child(root, cohort, output, role, reviewer):
    sys.path.insert(0, str(root))
    # Imports resolve against this complete source tree, including factual replay.
    spec = importlib.util.spec_from_file_location('isolated_world_reviewer', reviewer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = root
    raw = (root / 'registered/plan.json').read_bytes()
    plan = json.loads(raw)
    manifest = json.loads((root / 'registered/registration.json').read_bytes())
    assert sha(raw) == manifest['plan_sha256']
    for name, value in manifest['sources'].items():
        assert sha((root / name).read_bytes()) == value
    registration = dict(plan_sha256=manifest['plan_sha256'], source_sha256=sha(encoded(manifest['sources'])))
    if role == 'candidate' and 'selector' in plan['candidate']:
        controller = json.loads((cohort / 'controller.json').read_bytes())
        assert controller == dict(registration=registration, selector=plan['candidate']['selector'],
                                  episodes_and_branches_share_chooser=True, adoption=False)
    rows = {}
    for case in plan['specs']:
        for arm in (('stable', 'expected') if role == 'parent' else ('expected',)):
            directory = cohort / ('adverse-' + case['id'] + '-' + arm) if role == 'parent' else cohort / 'episodes' / case['id']
            result = module.execution(directory, case, arm, 'root' if role == 'parent' else 'shared', manifest, registration)
            if result is not None:
                rows[case['id'] + '|' + arm] = dict(score=result['score'],
                    conditional_increment=result['conditional_increment'], runtime=result['runtime'],
                    responses=[r['response'] for r in result['trace']],
                    result_sha256=sha((directory / 'result.json').read_bytes()))
    output.write_bytes(encoded(rows))


def compare(parent_source, candidate_source, parent_root, candidate_root, parity_receipt, output):
    parity = json.loads(parity_receipt.read_bytes())
    plan = contract(parent_source, candidate_source, parity)
    output.mkdir(parents=True, exist_ok=False)
    rows = {}
    reviewer = Path(__file__).with_name('review_recovery_worlds.py').resolve()
    for role, archive, cohort in (('parent', parent_source, parent_root), ('candidate', candidate_source, candidate_root)):
        root = unpack(archive, output / role)
        target = output / (role + '-audit.json')
        command = [sys.executable, str(Path(__file__).resolve()), '--child', str(root),
                   '--cohort', str(cohort), '--output', str(target), '--role', role, '--reviewer', str(reviewer)]
        with (output / (role + '.log')).open('w') as log:
            subprocess.run(command, cwd=root, env=dict(os.environ, PYTHONPATH=''),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        rows[role] = json.loads(target.read_bytes())
    pairs = []
    pending = []
    for spec in plan['specs']:
        key = spec['id']
        stable = rows['parent'].get(key + '|stable')
        old = rows['parent'].get(key + '|expected')
        new = rows['candidate'].get(key + '|expected')
        if stable is None or new is None:
            pending.append(key)
            continue
        assert stable['runtime'] == new['runtime']
        count = min(len(stable['responses']), len(new['responses']))
        assert stable['responses'][:count] == new['responses'][:count]
        pairs.append(dict(id=key, game=spec['game'], profile=spec['profile'],
            score_delta=new['score'] - stable['score'],
            conditional_delta=new['conditional_increment'] - stable['conditional_increment'],
            score=new['score'], stable_score=stable['score'],
            root_candidate_score=old['score'] if old else None))
    groups = defaultdict(list)
    for pair in pairs:
        groups[pair['game']].append(pair)
    families = {g: {k: sum(p[k] for p in members) / len(members) for k in ('score_delta', 'conditional_delta')}
                for g, members in groups.items()}
    gates = dict(complete=len(pairs) == len(plan['specs']),
        parent_root_audit_complete=sum(k.endswith('|expected') for k in rows['parent']) == len(plan['specs']),
        all_families=set(families) == {'auction', 'combat', 'resources', 'gambling'},
        no_family_loss=all(v >= -1e-12 for f in families.values() for v in f.values()),
        concordant_adverse_improvement=any(p['id'] in plan['concordant_adverse'] and p['score_delta'] > 1e-12
                                         and p['conditional_delta'] > 1e-12 for p in pairs))
    result = dict(complete=gates['complete'], pairs=pairs, pending=pending, families=families,
        gate_checks=gates, development_gate_passed=all(gates.values()), adoption=False,
        independent_confirmation='Requires C2 reference, C3 benefit and individual-loss review as well.',
        provenance=dict(parent_source=sha(parent_source.read_bytes()), candidate_source=sha(candidate_source.read_bytes()),
                        parity_receipt=sha(parity_receipt.read_bytes()), reviewer=sha(reviewer.read_bytes()),
                        isolated_reviewer=sha(Path(__file__).read_bytes())))
    (output / 'summary.json').write_bytes(encoded(result))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('child', 'cohort', 'reviewer', 'parent-source', 'candidate-source', 'parent-root', 'candidate-root', 'parity-receipt', 'output'):
        p.add_argument('--' + name, type=Path, required=name == 'output')
    p.add_argument('--role', choices=('parent', 'candidate'))
    a = p.parse_args()
    if a.child:
        child(a.child.resolve(), a.cohort.resolve(), a.output.resolve(), a.role, a.reviewer.resolve())
    else:
        result = compare(*(getattr(a, n).resolve() for n in
            ('parent_source', 'candidate_source', 'parent_root', 'candidate_root', 'parity_receipt', 'output')))
        print(json.dumps({k: result[k] for k in ('complete', 'development_gate_passed', 'pending')}))
