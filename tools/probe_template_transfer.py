"""Independent prediction-only check beyond the original three-symbol games.

No measured law or realized future is passed to either predictor. This is not
long-play acceptance and cannot authorize independent forty-game confirmation.
"""
import argparse
import gzip
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reflex.segmented_prediction import SegmentedEventLearner
from tools.symmetric_prediction import SymmetricEventLearner
from tools.template_transfer_prediction import TemplateTransferLearner
from tools.owned_checkpoint import encoded, sha


def run(plan_path, output):
    plan_path, output = Path(plan_path), Path(output)
    plan=json.loads(plan_path.read_bytes())
    if output.exists():raise FileExistsError('retain the original screen')
    for name,checksum in plan['sources'].items():
        assert sha((ROOT/name).read_bytes())==checksum, 'screen source changed'
    rows=[]
    for alphabet in plan['alphabets']:
        legal=tuple(map(str,range(alphabet)))
        template_rng=np.random.default_rng(42000+alphabet)
        models={f'template-{i}':dict(zip(legal,map(float,template_rng.dirichlet(np.full(alphabet,.7))))) for i in range(2)}
        for mode in plan['patterns']:
            for seed in plan['seeds']:
                law_rng=np.random.default_rng(np.random.SeedSequence([seed,alphabet,991]))
                laws=[list(models['template-0'].values()),list(map(float,law_rng.dirichlet(np.full(alphabet,1.3)))),
                      list(np.roll(list(models['template-0'].values()),1)),list(models['template-1'].values())]
                candidate = TemplateTransferLearner if plan.get('learner') == 'hierarchical-transfer' else SymmetricEventLearner
                learners={name:cls(owner='screen',model_id='public-categories',names=tuple(models))
                    for name,cls in [('baseline',SegmentedEventLearner),('transfer',candidate)]}
                memories={name:l.start() for name,l in learners.items()}
                metrics={name:dict(log_loss=0.,expected_log_regret=0.,expected_brier_excess=0.) for name in learners}
                rng=np.random.default_rng(seed);trace=[]
                for tick in range(plan['turns']):
                    index={'known':0,'novel':1,'permuted':2}.get(mode,min(3,4*tick//plan['turns']))
                    truth=dict(zip(legal,laws[index]))
                    predictions={name:l.predict(memories[name],(),legal,models) for name,l in learners.items()}
                    response=str(rng.choice(alphabet,p=laws[index]))
                    for name,l in learners.items():
                        p=predictions[name]
                        metrics[name]['log_loss']-=math.log(p[response])/plan['turns']
                        metrics[name]['expected_log_regret']+=sum(q*math.log(q/p[z]) for z,q in truth.items())/plan['turns']
                        metrics[name]['expected_brier_excess']+=sum((p[z]-q)**2 for z,q in truth.items())/plan['turns']
                        memories[name]=l.observe(memories[name],(),legal,models,response)
                    trace.append(dict(tick=tick,truth=truth,response=response,predictions=predictions))
                rows.append(dict(alphabet=alphabet,pattern=mode,seed=seed,metrics=metrics,trace=trace,
                    delta={k:metrics['transfer'][k]-v for k,v in metrics['baseline'].items()}))
    groups=[]
    for alphabet in plan['alphabets']:
        for pattern in plan['patterns']:
            group=[r for r in rows if (r['alphabet'],r['pattern'])==(alphabet,pattern)]
            groups.append(dict(alphabet=alphabet,pattern=pattern,streams=len(group),
                mean_delta={k:math.fsum(r['delta'][k] for r in group)/len(group) for k in group[0]['delta']}))
    output.mkdir(parents=True)
    (output/'rows.json.gz').write_bytes(gzip.compress(encoded(rows),mtime=0))
    summary=dict(complete=True,streams=len(rows),observations=sum(len(r['trace']) for r in rows),
        plan_sha256=sha(plan_path.read_bytes()),rows_sha256=sha((output/'rows.json.gz').read_bytes()),
        groups=groups,adoption=False,independent_game_confirmation=False)
    (output/'summary.json').write_bytes(encoded(summary))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.plan,a.output)
