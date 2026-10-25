"""Replay every registered terminal fork from the published frozen inputs.

No forecasts are rerun and no new games are generated. This independently
reconstructs factual owner updates, world RNG and diagnostic accounts, including
the four archived resume controls. The full forecast parity of those controls
was checked by the original runner and is retained in its receipts.
"""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import zipfile


def main():
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    read = lambda path: json.loads(path.read_bytes())
    proof = read(here / 'results_verification.json')
    for name, expected in proof['artifacts'].items():
        assert sha((here / name).read_bytes()) == expected, name
    registration = read(here / 'registration_proof.json')
    for name, expected in registration['artifacts'].items():
        assert sha((here / name).read_bytes()) == expected, name
    plan = read(here / 'plan.json')
    assert sha((here / 'plan.json').read_bytes()) == proof['plan_sha256']
    source = repo / registration['source_zip']
    assert sha(source.read_bytes()) == registration['source_zip_sha256']
    runner = here / 'frozen_runner.py'
    assert sha(runner.read_bytes()) == plan['runner_sha256']
    spec = importlib.util.spec_from_file_location('sampled_root_auditor', runner)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = json.loads(gzip.decompress((here / 'results.json.gz').read_bytes()))
    rows = {row['job']: row for row in expected['rows']}
    jobs = plan['controls'] + plan['branches']
    assert len(rows) == len(jobs) == 9
    assert set(rows) == {job['id'] for job in jobs}
    cases = {case['id']: case for case in plan['cases']}
    with tempfile.TemporaryDirectory(prefix='sampled-root-results-') as temporary:
        root = Path(temporary)
        for archive, target in ((source, root / '_source'),
                                (here / 'inputs.zip', root),
                                (here / 'results.zip', root)):
            with zipfile.ZipFile(archive) as bundle:
                for name in bundle.namelist():
                    assert (target / name).resolve().is_relative_to(target.resolve())
                bundle.extractall(target)
        manifest = read(root / 'source_snapshot.json')
        assert module.sha(module.encoded(manifest)) == plan['source']
        for name, value in manifest.items():
            assert sha((root / '_source' / name).read_bytes()) == value, name
        sys.path.insert(0, str(root / '_source'))
        for job in jobs:
            actual = module.audit(root, cases[job['case']], job, terminal=not job['control'])
            assert actual == rows[job['id']], job['id']
            receipt = read(root / 'runs' / job['id'] / 'verification.json')
            assert actual == receipt
            print(json.dumps({'job': job['id'], 'actual_replay_exact': True}), flush=True)
    print(json.dumps(dict(complete=True, terminal_forks=5, resume_controls=4,
        terminal_trajectory_steps=640, control_prefix_steps=242,
        new_full_planning_decisions=sum(row.get('new_full_planning_decisions', 0)
                                       for row in rows.values()),
        exact_reproduction=True, independent_worlds_added=0, adoption=False)), flush=True)


if __name__ == '__main__':
    main()
