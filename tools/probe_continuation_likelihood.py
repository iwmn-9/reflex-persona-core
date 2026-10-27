"""Score existing world-continuation assumptions on retained observation streams.

Diagnostic only: no controller, personality, source stream or acceptance gate
is changed. The chain-rule likelihood conditions on the subsequently revealed
prefix; it is a joint forecast score, not an unconditional marginal forecast
at every horizon, nor evidence that a policy improves.
"""
import argparse
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import platform
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reflex.segmented_prediction import segment_prediction, persistent_prediction
from tools.owned_checkpoint import encoded, sha


def main(output):
    out = Path(output)
    if out.exists():
        raise FileExistsError('fresh diagnostic directory required')
    source = ROOT/'evidence/long_horizon/adverse_recovery/prediction_probe/rows.json.gz'
    inputs = source.read_bytes()
    streams = json.loads(gzip.decompress(inputs))
    assert len(streams) == 16 and all(len(s['trace']) == 128 for s in streams)
    assert len({(s['regime'], s['seed']) for s in streams}) == 16
    files = list((ROOT/'reflex').glob('*.py')) + [Path(__file__), ROOT/'tools/owned_checkpoint.py']
    sources = {p.relative_to(ROOT).as_posix(): sha(p.read_bytes()) for p in files}
    registration = dict(at=datetime.now(timezone.utc).isoformat(), inputs_sha256=sha(inputs),
        input_source=source.relative_to(ROOT).as_posix(), sources=sources,
        roots=list(range(0, 97, 16)), horizon=32, models=['posterior', 'persistent'],
        supplied_laws=[[.8, .1, .1], [.1, .8, .1]], alphabet=['0', '1', '2'],
        conditions=[dict(seed=s['seed'], regime=s['regime']) for s in streams],
        interpretation=__doc__, adoption=False)
    out.mkdir(parents=True)
    (out/'registration.json').write_bytes(encoded(registration))
    (out/'inputs.json.gz').write_bytes(inputs)
    with zipfile.ZipFile(out/'source.zip', 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
    laws = tuple(tuple(row) for row in registration['supplied_laws'])
    rows = []
    for stream in streams:
        sequence = tuple(int(t['response']) for t in stream['trace'])
        for root in registration['roots']:
            prefix = sequence[:root]
            forecasts = []
            for offset in range(registration['horizon']):
                tick = root+offset
                future = sequence[root:tick]
                posterior = segment_prediction(sequence[:tick], (laws,)*tick, laws, 3)[0]
                persistent = persistent_prediction(prefix, (laws,)*root, future, (laws,)*offset, laws, 3)[0]
                if offset == 0:
                    # Mathematical equivalence at the factual root, not a
                    # relaxation of any execution/owner replay requirement.
                    assert max(abs(a-b) for a, b in zip(posterior, persistent)) < 1e-12
                actual = stream['trace'][tick]
                predictions = dict(posterior=posterior, persistent=persistent)
                losses = {name: dict(log_loss=-math.log(p[sequence[tick]]),
                    brier_excess=sum((p[i]-actual['truth'][str(i)])**2 for i in range(3)))
                    for name, p in predictions.items()}
                forecasts.append(dict(offset=offset, tick=tick, response=actual['response'],
                    predictions=predictions, losses=losses))
            metrics = {name: {k: sum(t['losses'][name][k] for t in forecasts)/len(forecasts)
                for k in ('log_loss', 'brier_excess')} for name in registration['models']}
            rows.append(dict(seed=stream['seed'], regime=stream['regime'], root=root,
                metrics=metrics, persistent_minus_posterior={k: metrics['persistent'][k]-metrics['posterior'][k]
                    for k in ('log_loss', 'brier_excess')}, forecasts=forecasts))
        print(json.dumps(dict(regime=stream['regime'], seed=stream['seed'], roots_completed=7)), flush=True)
    assert len(rows) == 112
    assert all(sha((ROOT/n).read_bytes()) == h for n, h in sources.items())
    regimes = sorted({r['regime'] for r in rows})
    means = {regime: {k: sum(r['persistent_minus_posterior'][k] for r in rows if r['regime'] == regime)/28
        for k in ('log_loss', 'brier_excess')} for regime in regimes}
    summary = dict(complete=True, streams=16, roots=112, steps_per_root=32,
        predictions_per_model=3584, lower_is_better=True, persistent_minus_posterior=means,
        source_unchanged=True, controller_adoption=False,
        runtime=dict(python=platform.python_version(), numpy=np.__version__, system=platform.system()),
        limitations=[__doc__, 'Overlapping forecast windows are not independent samples.',
            'These retained diagnostic streams are not the reserved independent forty conditions.',
            'Better predictive likelihood need not imply better finite-sample personality decisions.'])
    (out/'rows.json.gz').write_bytes(gzip.compress(encoded(rows), mtime=0))
    (out/'summary.json').write_bytes(encoded(summary))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    main(p.parse_args().output)
