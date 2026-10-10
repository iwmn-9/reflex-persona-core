"""Score already-frozen factual models on new baseline learning series."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.probe_outcome_value import extract, predict, metrics


def score(root,models):
    root=Path(root);models=Path(models);development=json.loads((models/'development.json').read_text(encoding='utf-8'))
    registration=json.loads((models/'preregister.json').read_text(encoding='utf-8'))
    tool=Path(__file__).with_name('probe_outcome_value.py')
    assert hashlib.sha256(tool.read_bytes()).hexdigest()==registration['implementation_sha256']
    reports=[];rng=np.random.default_rng(90213)
    for game in ('goofspiel','no_thanks'):
        frozen=models/f'{game}-frozen.npz';info=next(r for r in development['results'] if r['game']==game)
        assert hashlib.sha256(frozen.read_bytes()).hexdigest()==info['frozen_model_sha256']
        with np.load(frozen,allow_pickle=False) as f:model={k:f[k] for k in f.files}
        pieces=[]
        for i in range(4):pieces.append(extract(root/f'shard-{i}'/'baseline')[game])
        d={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
        assert sorted(set(d['key'][:,0]))==list(range(6700,6716))
        p=predict(model,d['x']);mc=np.clip(d['mc'],.02,.98);y=d['y'];w=d['weights']
        unique,first=np.unique(d['key'],axis=0,return_index=True)
        scopes={}
        for label,ids in (('all_weighted_choices',np.arange(len(y))),('first_choice_per_npc_episode',first)):
            weights=w[ids] if label=='all_weighted_choices' else np.ones(len(ids))
            series=[]
            for seed in range(6700,6716):
                mask=d['key'][ids,0]==seed
                diff=(mc[ids][mask]-y[ids][mask])**2-(p[ids][mask]-y[ids][mask])**2
                series.append(float(np.average(diff,weights=weights[mask])))
            val=np.array(series);sample=val[rng.integers(len(val),size=(10000,len(val)))].mean(1)
            scopes[label]=dict(ridge=metrics(y[ids],p[ids],weights),mc=metrics(y[ids],d['mc'][ids],weights),
                constant=metrics(y[ids],np.full(len(ids),.25),weights),
                brier_improvement=float(val.mean()),paired_series_bootstrap_95=list(map(float,np.quantile(sample,[.025,.975]))),
                independent_series=16)
        reports.append(dict(game=game,model_sha256=info['frozen_model_sha256'],scopes=scopes))
    result=dict(training_registration=registration,prospective_seeds=list(range(6700,6716)),reports=reports,
        adopted_as_controller=False,
        limitations=['factual selected-action prediction; no support test or counterfactual action guarantee',
                     'different future continuation targets: MC models a plan, ridge actual historical behaviour',
                     'baseline only; same known games/profiles, new seeds, not unknown-game generalization'])
    (root/'value_probe_prospective.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--models',required=True)
    a=p.parse_args();score(a.root,a.models)
