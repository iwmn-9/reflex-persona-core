"""Archive every resolved candidate-root diagnostic without qualifying adoption.

The complete study, registered root rule, preparation manifest and finished
execution must agree. Inputs are read from the complete study archive; no live
scientific checkpoint is opened and no forecast is executed by this collector.
"""
import argparse
from datetime import datetime, timezone
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import uuid
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.run_registered_loss_roots import read, sha, validate


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def archive(execution, study, root_plan, prepared_manifest, resolver, output):
    execution, study = Path(execution), Path(study)
    root_plan, prepared_manifest, resolver = map(Path, (root_plan, prepared_manifest, resolver))
    output = Path(output)
    if output.exists():
        raise FileExistsError('retain the existing audit; never replace it')
    plan, manifest = read(root_plan), read(prepared_manifest)
    assert read(execution / 'plan.json') == plan
    assert read(execution / 'manifest.json') == manifest
    assert sha(root_plan) == sha(execution / 'plan.json') == manifest['plan_sha256']
    assert sha(resolver) == plan['resolver_sha256']
    # Resolve again from the completed study so missing losses, altered roots,
    # substituted input histories or reduced diagnostic banks cannot pass.
    spec = importlib.util.spec_from_file_location('registered_root_resolver', resolver)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.prepare(study, None) == plan
    whole = read(study / 'verification.json')
    assert whole['complete'] and whole['registration'] == plan['registration']
    assert whole['source_ref'] == plan['source_ref'] == manifest['source_ref']
    for name, checksum in plan['inputs'].items():
        assert sha(study / name) == checksum == whole['artifacts'][name]
    status, receipt = read(execution / 'status.json'), read(execution / 'verification.json')
    jobs = plan['jobs']
    assert status['state'] == 'complete' and status['complete']
    assert status['evaluations_complete'] == status['evaluations_required'] == len(jobs)
    assert receipt['complete'] and not receipt['adoption'] and receipt['exact_factual_roots']
    assert receipt['evaluations'] == len(jobs) and receipt['sources'] == manifest
    assert receipt['root_plan_sha256'] == sha(root_plan)
    assert [r['job'] for r in receipt['rows']] == jobs
    assert sha(execution / 'driver.py') == manifest['driver_sha256']
    assert sha(execution / 'probe_registered_owner.py') == manifest['probe_sha256']
    rows, files, controls = [], [], 0
    with zipfile.ZipFile(study / 'checkpoints.zip') as inputs:
        for index, record in enumerate(receipt['rows']):
            filename = record['file']
            assert filename == f'root-{index:03}.json'
            path = execution / filename
            assert sha(path) == record['sha256']
            row, job = read(path), record['job']
            original = json.loads(inputs.read(job['input_archive_member']))
            controls += validate(job, row, original, plan['registration'], manifest['probe_sha256'])
            assert not row['adoption']
            assert row['personal_ranking']['exact_selection_reproduced']
            assert set(row['personal_ranking']['plans']) == set(row['purpose_by_plan'])
            rows.append(row)
            files.append(filename)
    roots = {(j['id'], j['tick']) for j in jobs}
    assert len(jobs) == 6 * len(roots)
    assert controls == receipt['exact_original_forecast_controls'] == len(roots)
    reviews = []
    for loss in plan['losses']:
        for root in loss['roots']:
            group = sorted((r for r in rows if (r['id'], r['tick']) == (loss['id'], root['tick'])),
                           key=lambda r: (r['seeds'][0], len(r['seeds'])))
            choices = {str(n): [r['action'] for r in group if len(r['seeds']) == n] for n in (3, 8)}
            reviews.append(dict(id=loss['id'], tick=root['tick'], reasons=root['reasons'],
                losses=loss['losses'], factual_owner=root['factual_owner'],
                actions_by_sample_count=choices,
                unique_actions_by_sample_count={n: len(set(a)) for n, a in choices.items()},
                selections=[{k: r[k] for k in ('seeds', 'action', 'selected_purpose',
                    'estimated_purpose_by_root', 'personal_ranking')} for r in group]))
    staging = output.with_name(output.name + '.partial-' + uuid.uuid4().hex)
    staging.mkdir(parents=True)
    names = ['manifest.json', 'plan.json', 'verification.json', 'status.json',
             'driver.py', 'probe_registered_owner.py', *files]
    with zipfile.ZipFile(staging / 'execution.zip', 'x', zipfile.ZIP_DEFLATED) as packed:
        for name in names:
            packed.writestr(name, (execution / name).read_bytes())
    (staging / 'rows.json.gz').write_bytes(gzip.compress(encoded(rows), mtime=0))
    (staging / 'review.json').write_bytes(encoded(dict(complete=True, adoption=False, roots=reviews,
        limits=['Known factual roots only, not independent long-play evidence or an adoption gate.',
                'Three predeclared banks measure sensitivity, not a calibrated confidence interval.',
                'The actual persona choice is distinct from the highest raw purpose estimate.',
                'Stable local choices do not establish correctness of the unknown future world.'])))
    (staging / 'plan.json').write_bytes(root_plan.read_bytes())
    (staging / 'resolver.py').write_bytes(resolver.read_bytes())
    (staging / 'collector.py').write_bytes(Path(__file__).read_bytes())
    proof = dict(complete=True, adoption=False, at=datetime.now(timezone.utc).isoformat(),
        evaluations=len(rows), exact_factual_roots=len(roots), exact_original_forecast_controls=controls,
        source_ref=plan['source_ref'], registration=plan['registration'],
        whole_study_verification_sha256=sha(study / 'verification.json'),
        preparation_manifest_sha256=sha(prepared_manifest), collector_sha256=sha(__file__),
        artifacts={p.name: sha(p) for p in staging.iterdir()})
    (staging / 'verification.json').write_bytes(encoded(proof))
    # Verify the actual retained bytes before exposing the completed directory.
    assert all(sha(staging / name) == value for name, value in proof['artifacts'].items())
    assert json.loads(gzip.decompress((staging / 'rows.json.gz').read_bytes())) == rows
    staging.rename(output)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('execution', 'study', 'root-plan', 'prepared-manifest', 'resolver', 'output'):
        parser.add_argument('--' + name, required=True)
    args = vars(parser.parse_args())
    print(json.dumps(archive(**args)))
