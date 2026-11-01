"""Prediction-only causal probe: supplied hypotheses versus unknown segments.

This cannot adopt a controller. Both observers see the same actual categorical
stream, before the next outcome is revealed. No action, reward or personality
is fitted. The hidden law is used only for evaluator Brier excess.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reflex.segmented_prediction import SegmentedEventLearner
from tools.owned_checkpoint import encoded,sha
from tools.run_owned_value import law,sample


def main(output):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    registration=dict(registered_at=datetime.now(timezone.utc).isoformat(),
        seeds=[2026102811,2026102812,2026102813,2026102814],
        regimes=['outside','shift-return','irregular','steady'],turns=128,
        arms=['supplied-plus-unknown','unknown-only'],
        gate='Diagnostic only: report all log-loss and expected Brier differences by regime. No controller adoption or changed acceptance.',
        source={n:sha((ROOT/n).read_bytes()) for n in ('reflex/segmented_prediction.py','tools/run_owned_value.py','tools/probe_adverse_prediction.py')})
    if (root/'registration.json').exists():raise FileExistsError('fresh diagnostic directory required')
    (root/'registration.json').write_bytes(encoded(registration))
    rows=[]
    labels=('0','1','2')
    models={f'known-{z}':{k:.8 if k==z else .1 for k in labels} for z in ('0','1')}
    for regime in registration['regimes']:
        for seed in registration['seeds']:
            rng=np.random.default_rng(seed)
            learners={arm:SegmentedEventLearner(owner='diagnostic',model_id='public-symbols-v1',names=tuple(models) if arm=='supplied-plus-unknown' else ()) for arm in registration['arms']}
            memories={k:v.start() for k,v in learners.items()};trace=[]
            for tick in range(registration['turns']):
                predictions={k:v.predict(memories[k],(),labels,models if k=='supplied-plus-unknown' else {}) for k,v in learners.items()}
                truth=law(tick,registration['turns'],regime);response=sample(truth,rng.random())
                for k,v in learners.items():memories[k]=v.observe(memories[k],(),labels,models if k=='supplied-plus-unknown' else {},response)
                trace.append(dict(tick=tick,truth=truth,response=response,predictions=predictions))
            metrics={arm:dict(log_loss=sum(-math.log(t['predictions'][arm][t['response']]) for t in trace)/len(trace),
                expected_brier_excess=sum(sum((t['predictions'][arm][k]-p)**2 for k,p in t['truth'].items()) for t in trace)/len(trace)) for arm in registration['arms']}
            row=dict(seed=seed,regime=regime,metrics=metrics,trace=trace,
                differences={k:metrics['unknown-only'][k]-metrics['supplied-plus-unknown'][k] for k in ('log_loss','expected_brier_excess')})
            rows.append(row);(root/'rows.json').write_bytes(encoded(rows))
            print(json.dumps({k:v for k,v in row.items() if k not in ('trace','metrics')}),flush=True)
    by_regime={regime:{metric:sum(r['differences'][metric] for r in rows if r['regime']==regime)/len(registration['seeds']) for metric in ('log_loss','expected_brier_excess')} for regime in registration['regimes']}
    result=dict(complete=True,conditions=len(rows),observations_per_arm=sum(len(r['trace']) for r in rows),
        lower_is_better=True,unknown_only_minus_supplied=by_regime,controller_adoption=False,
        source_unchanged=all(sha((ROOT/n).read_bytes())==h for n,h in registration['source'].items()))
    assert result['source_unchanged']
    (root/'summary.json').write_bytes(encoded(result));print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    main(p.parse_args().output)
