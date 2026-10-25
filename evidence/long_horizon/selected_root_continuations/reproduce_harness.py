"""Re-run the short diagnostic-harness test in a fresh isolated directory."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

here = Path(__file__).resolve().parent
repo = here.parents[2]
proof = json.loads((here / 'registration_proof.json').read_bytes())
sha = lambda raw: hashlib.sha256(raw).hexdigest()
for name, expected in proof['artifacts'].items():
    assert sha((here / name).read_bytes()) == expected, name
runner = repo / proof['runner']
assert sha(runner.read_bytes()) == proof['runner_sha256']
source_zip = repo / proof['source_zip']
assert sha(source_zip.read_bytes()) == proof['source_zip_sha256']
with tempfile.TemporaryDirectory(prefix='sampled-root-harness-') as temporary:
    isolated = Path(temporary)
    test = isolated / 'results' / 'check'
    source = test / '_source'
    source.mkdir(parents=True)
    (isolated / 'tools').mkdir()
    (isolated / proof['runner']).write_bytes(runner.read_bytes())
    (test / 'check.py').write_bytes((here / 'harness_test.py').read_bytes())
    with zipfile.ZipFile(source_zip) as archive:
        for item in archive.infolist():
            assert (source / item.filename).resolve().is_relative_to(source.resolve())
        archive.extractall(source)
    subprocess.run([sys.executable, '-I', str(test / 'check.py')], cwd=isolated, check=True)
    result = json.loads((test / 'verification.json').read_bytes())
    original = json.loads((here / 'harness_test.json').read_bytes())
    assert result['checks'] == original['checks']
    assert result['runner_sha256'] == original['runner_sha256']
    for key in original['control']:
        if key != 'checkpoint_sha256':
            assert result['control'][key] == original['control'][key], key
    # Result/checkpoint file hashes contain measured wall time; compare outcomes
    # and factual replay properties, not that nondeterministic metadata.
    for key in ('intervention_action', 'actual_decisions_replayed', 'unchanged_prefix',
                'factual_owner_replayed', 'world_rng_replayed', 'personality_preserved',
                'complete', 'adoption', 'score_delta', 'conditional_delta', 'residual_delta',
                'scores', 'new_full_planning_decisions', 'accounts'):
        assert result['branch'][key] == original['branch'][key], key
print(json.dumps({'complete': True, 'scope': 'fresh six-turn harness replay only',
                  'checks': len(result['checks']), 'quality_evidence': False}))
