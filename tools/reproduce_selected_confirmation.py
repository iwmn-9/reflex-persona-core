"""Re-audit the published forty-pair evidence in a fresh isolated directory.

This replays factual actions and learning; it does not generate new games or
rerun expensive forecasts. Complete checkpoints and frozen code are required.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    folder = Path(args.evidence).resolve()
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    read = lambda path: json.loads(path.read_bytes())
    package = read(folder / 'reproduction_package.json')
    for name, expected in package['artifacts'].items():
        assert sha((folder / name).read_bytes()) == expected, name
    proof = read(folder / 'verification.json')
    for name, expected in proof['artifacts'].items():
        assert sha((folder / name).read_bytes()) == expected, name
    assert proof['complete'] and proof['pairs'] == 40 and proof['replayed_actual_decisions'] == 10240
    expected_quality = read(folder / 'quality_review.json')
    expected_results = {row['id']: row for row in expected_quality['results']}
    runs = [json.loads(line) for line in gzip.decompress((folder / 'runs.jsonl.gz').read_bytes()).splitlines()]
    assert len(runs) == 80
    with tempfile.TemporaryDirectory(prefix='selected-confirmation-replay-') as temporary:
        root = Path(temporary)
        for archive, target in ((folder / 'source.zip', root / '_source'),
                                (folder / 'checkpoints.zip', root)):
            with zipfile.ZipFile(archive) as bundle:
                for name in bundle.namelist():
                    assert (target / name).resolve().is_relative_to(target.resolve())
                bundle.extractall(target)
        for name in ('plan.json', 'summary.json', 'source_snapshot.json'):
            (root / name).write_bytes((folder / name).read_bytes())
        for row in runs:
            identifier = row['spec']['id']
            side = int(row['retry'])
            raw = encoded(row['result'])
            receipt = expected_results[identifier]['results'][side]
            assert sha(raw) == receipt['sha256']
            path = root / receipt['path']
            assert path.resolve().is_relative_to(root)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        audit = subprocess.run([sys.executable, '-I', str(folder / 'auditor.py'),
            '--root', str(root)], text=True, capture_output=True)
        if audit.returncode:
            print(audit.stderr, file=sys.stderr)
            audit.check_returncode()
        actual = read(root / 'verification.json')
        assert actual == {k: v for k, v in proof.items() if k != 'artifacts'}
        review = subprocess.run([sys.executable, '-I', str(folder / 'quality_reviewer.py'),
            '--root', str(root), '--output', str(root / 'quality_review.json')],
            text=True, capture_output=True)
        if review.returncode:
            print(review.stderr, file=sys.stderr)
            review.check_returncode()
        assert read(root / 'quality_review.json') == expected_quality
        result = dict(complete=True, audited_pairs=40, actual_decisions=10240,
            exact_factual_audit=True, exact_quality_review=True, new_games=0,
            forecasts_recomputed=0, independent_evidence_added=0, adoption=False,
            source=expected_quality['source'],
            reproduction_package_sha256=sha((folder / 'reproduction_package.json').read_bytes()),
            reproducer_sha256=sha(Path(__file__).read_bytes()),
            audit_output=json.loads(audit.stdout), quality_output=json.loads(review.stdout))
    Path(args.receipt).write_bytes(encoded(result))
    print(json.dumps({k: v for k, v in result.items() if k not in ('audit_output', 'quality_output')}), flush=True)


if __name__ == '__main__':
    main()
