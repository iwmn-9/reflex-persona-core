"""Verify all predeclared roots, reusing the original gambling results exactly."""
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile
from vmctl import run

V = Path(__file__).resolve().parent
U = V.parents[1]
R = U.parent / 'reflex-persona-core'
E = R / 'evidence/long_horizon/adverse_sample_budget'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def main():
    target = E / 'root_diagnosis_all'
    if target.exists():
        raise FileExistsError('never replace an existing root audit')
    status = json.loads(run('cat /home/ubuntu/recovery/sample8_loss_roots/status.json\n', capture=True).stdout)
    assert status['complete'] and status['state'] == 'complete' and status['evaluations_complete'] == 12
    raw = run('cd /home/ubuntu/recovery/sample8_loss_roots\ntar -czf - manifest.json plan.json verification.json status.json driver.py probe_frozen_sample_budget.py root-*.json\n', capture=True).stdout
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        assert all(m.isfile() and '/' not in m.name and '\\' not in m.name for m in tar.getmembers())
        files = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
        assert len(files) == len(tar.getmembers())
    plan_raw = (E / 'loss_root_plan.json').read_bytes()
    plan = json.loads(plan_raw)
    assert files['plan.json'] == plan_raw
    manifest = json.loads(files['manifest.json'])
    assert manifest == json.loads((V / 'http/sample8-root-manifest.json').read_bytes())
    assert manifest['plan_sha256'] == sha(plan_raw)
    assert manifest['driver_sha256'] == sha(files['driver.py'])
    assert manifest['probe_sha256'] == sha(files['probe_frozen_sample_budget.py'])
    assert sha((E / 'loss_root_resolver.py').read_bytes()) == plan['resolver_sha256']
    remote = json.loads(files['verification.json'])
    assert remote['complete'] and remote['exact_factual_roots'] and remote['evaluations'] == 12
    assert remote['sources'] == manifest and remote['root_plan_sha256'] == sha(plan_raw)
    pending = [job for job in plan['jobs'] if not job['reuse']]
    assert [r['job'] for r in remote['rows']] == pending
    original = {}
    archive = E / 'completed_local'
    full = json.loads((archive / 'verification.json').read_bytes())
    assert full['complete'] and full['actual_steps'] == 2560 and full['pairs'] == 10
    assert full['source_ref'] == plan['source_ref'] == manifest['source_ref']
    assert full['registration'] == plan['registration']
    for name, checksum in plan['inputs'].items():
        assert sha((archive / name).read_bytes()) == checksum == full['artifacts'][name]
    with zipfile.ZipFile(archive / 'checkpoints.zip') as checkpoints:
        for job in plan['jobs']:
            raw_input = checkpoints.read(job['input_archive_member'])
            assert sha(raw_input) == job['input_sha256']
            original[job['id']] = json.loads(raw_input)
    prior = E / 'root_diagnosis'
    prior_raw = (prior / 'rows.json.gz').read_bytes()
    prior_proof = json.loads((prior / 'verification.json').read_bytes())
    assert prior_proof['complete'] and prior_proof['evaluations'] == 12
    assert sha(prior_raw) == prior_proof['rows_sha256'] == plan['prior_diagnosis']['rows_sha256']
    assert sha((prior / 'registration.json').read_bytes()) == plan['prior_diagnosis']['registration_sha256']
    old_rows = json.loads(gzip.decompress(prior_raw))
    new_rows = []
    for record in remote['rows']:
        data = files[record['file']]
        assert sha(data) == record['sha256']
        new_rows.append(json.loads(data))
    rows = old_rows + new_rows
    key = lambda r: (r['id'], r['tick'], r['seeds'][0], len(r['seeds']))
    by_key = {key(row): row for row in rows}
    expected = {(j['id'], j['tick'], j['start'], j['samples']) for j in plan['jobs']}
    assert len(by_key) == len(rows) == len(expected) == 24 and set(by_key) == expected
    controls = 0
    for job in plan['jobs']:
        row = by_key[job['id'], job['tick'], job['start'], job['samples']]
        saved = original[job['id']]['trace'][job['tick']]
        assert row['seeds'] == list(range(job['start'], job['start'] + job['samples']))
        assert row['registration'] == plan['registration'] and row['input_sha256'] == job['input_sha256']
        assert row['factual_owner'] == job['factual_owner'] == saved['before_owner']
        assert row['diagnostic_source_sha256'] == manifest['probe_sha256']
        if (job['start'], job['samples']) == (701, 8):
            assert row['exact_registered_forecast_reproduced']
            assert (row['action'], row['selected_purpose']) == (saved['action'], saved['selected_purpose'])
            controls += 1
    assert controls == 4 and remote['exact_original_forecast_controls'] == 2
    review = []
    for loss in plan['losses']:
        for root in loss['roots']:
            group = [r for r in rows if (r['id'], r['tick']) == (loss['id'], root['tick'])]
            choices = {str(n): [r['action'] for r in sorted(group, key=key) if len(r['seeds']) == n] for n in (3, 8)}
            review.append(dict(id=loss['id'], tick=root['tick'], factual_owner=root['factual_owner'],
                reasons=root['reasons'], actions_by_sample_count=choices,
                unique_actions_by_sample_count={n: len(set(a)) for n, a in choices.items()},
                selected_purpose_and_root_values=[{k: r[k] for k in ('seeds','action','selected_purpose','estimated_purpose_by_root')} for r in sorted(group,key=key)]))
    target.mkdir()
    # Keep the complete new execution outputs and their original worker proof.
    with zipfile.ZipFile(target / 'execution.zip', 'x', zipfile.ZIP_DEFLATED) as packed:
        for name, data in files.items():
            packed.writestr(name, data)
    (target / 'rows.json.gz').write_bytes(gzip.compress(encoded(sorted(rows,key=key)), mtime=0))
    (target / 'review.json').write_bytes(encoded(dict(complete=True, adoption=False, roots=review,
        limits=['Known factual roots, not independent long-play worlds or an adoption gate.',
                'Variation among three predeclared banks is sensitivity, not a calibrated confidence interval.',
                'Stable local choices cannot establish correctness of the unknown response law or the whole future-world model.'])))
    (target / 'plan.json').write_bytes(plan_raw)
    (target / 'collector.py').write_bytes(Path(__file__).read_bytes())
    proof = dict(complete=True, adoption=False, at=datetime.now(timezone.utc).isoformat(),
        evaluations=24, reused_evaluations=12, new_evaluations=12, exact_factual_roots=4,
        exact_original_forecast_controls=controls, source_ref=plan['source_ref'], registration=plan['registration'],
        whole_study_verification_sha256=sha((archive / 'verification.json').read_bytes()),
        prior_rows_sha256=sha(prior_raw), remote_bundle_sha256=sha(raw),
        collector_sha256=sha(Path(__file__).read_bytes()), artifacts={p.name: sha(p.read_bytes()) for p in target.iterdir()})
    (target / 'verification.json').write_bytes(encoded(proof))
    print(json.dumps(dict(evaluations=24, roots=review, adoption=False)))


if __name__ == '__main__':
    main()
