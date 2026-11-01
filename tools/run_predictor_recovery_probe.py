"""Prediction-only diagnosis on exogenous events; not a live-agent recovery gate."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from reflex.segmented_prediction import SegmentedEventLearner
from tools.completion_metrics import summarize_trace
from tools.owned_checkpoint import encoded,sha
from tools.run_owned_value import law,sample


def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    specs=[dict(seed=s,regime=r,turns=128) for s in (10061201,10061202)
           for r in ('outside','shift-return','irregular','steady')]
    sources={p.relative_to(ROOT).as_posix():sha(p.read_bytes())
             for d in ('reflex','tools') for p in (ROOT/d).rglob('*.py')}
    plan=dict(specs=specs,sources=sources,adoption=False,
        completion_contract_sha256=sha((ROOT/'docs/COMPLETION.md').read_bytes()),
        criteria='Fixed C3 prediction metric, all streams retained. No physical-agent death censoring; component diagnosis only. No behavioral or C3 completion claim.')
    with (out/'plan.json').open('xb') as f:f.write(encoded(plan))
    rows=[]
    models={str(i):{str(z):(.8 if z==i else .1) for z in range(3)} for i in (0,1)}
    for spec in specs:
        learner=SegmentedEventLearner(factors=(),names=('0','1'),owner='recovery-probe',model_id='public-library-v1')
        memory=learner.start();rng=np.random.default_rng(spec['seed']);trace=[]
        schedule=[law(t,spec['turns'],spec['regime']) for t in range(spec['turns'])]
        for t in range(spec['turns']):
            p=learner.predict(memory,(),('0','1','2'),models)
            # Only the world samples q. Learner receives the realized public symbol.
            response=sample(schedule[t],rng.random())
            trace.append(dict(tick=t,before={'tick':t},after={'tick':t+1},
                response_prediction=p,truth=schedule[t],response=response,conditional_increment=0.))
            memory=learner.observe(memory,(),('0','1','2'),models,response)
            saved=learner.record(memory);memory=learner.restore(json.loads(encoded(saved)))
            assert learner.record(memory)==saved
        metrics=summarize_trace(trace,scheduled_turns=128,terminal=True,truth_schedule=schedule)
        row=dict(spec=spec,trace=trace,metrics=metrics,observations=memory.observations)
        rows.append(row)
        (out/f"{spec['regime']}-{spec['seed']}.json").write_bytes(encoded(row))
    changes=sum(r['metrics']['qualifying_change_segments'] for r in rows)
    recovered=sum(r['metrics']['recovered_within_16'] for r in rows)
    assert all(sha((ROOT/n).read_bytes())==h for n,h in sources.items())
    result=dict(complete=True,streams=len(rows),public_observations=sum(r['observations'] for r in rows),
        qualifying_changes=changes,recovered_within_16=recovered,
        rate=recovered/changes if changes else None,component_gate=recovered>=.9*changes if changes else None,
        regimes={regime:[dict(seed=r['spec']['seed'],segments=r['metrics']['regimes'],mean_brier=r['metrics']['all_mean_brier_excess'])
                          for r in rows if r['spec']['regime']==regime] for regime in ('outside','shift-return','irregular','steady')},
        source_unchanged=True,adoption=False,
        C3='unmeasured: prediction component only; live-agent censoring and learned/frozen benefit comparison absent')
    (out/'summary.json').write_bytes(encoded(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    main(p.parse_args().output)
