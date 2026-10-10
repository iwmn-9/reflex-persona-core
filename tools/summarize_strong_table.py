"""Merge predeclared independent series and compute paired series uncertainty."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.strong_table import summarize


def merge(root):
    root=Path(root);registered=json.loads((root/'analysis_preregister.json').read_text(encoding='utf-8'))
    parts=[root.with_name('strong_table_part'+str(seeds[0])) for seeds in registered['shards']]
    runs=[];plans=[];elapsed=[]
    for part in parts:
        data=json.loads((part/'evaluation.json').read_text(encoding='utf-8'));plan=data['plan']
        assert plan['seeds'] in registered['shards']
        for key in ('encounters','games','modes','strong','persona','sources'):assert plan[key]==registered[key],key
        assert hashlib.sha256((part/'trajectories.jsonl').read_bytes()).hexdigest()==data['trajectory_sha256']
        runs.extend(data['matches']);plans.append(plan);elapsed.append(data['elapsed_seconds'])
    assert sorted(r['seed'] for r in runs[::registered['encounters']]) # nonempty; exact uniqueness below
    expected={(g,s,m,e) for g in registered['games'] for s in registered['seeds'] for m in registered['modes'] for e in range(registered['encounters'])}
    indexed={(r['game'],r['seed'],r['mode'],r['encounter']):r for r in runs}
    assert len(indexed)==len(runs) and set(indexed)==expected
    plan=dict(plans[0]);plan['seeds']=registered['seeds'];plan['analysis_registration_sha256']=hashlib.sha256((root/'analysis_preregister.json').read_bytes()).hexdigest()
    with (root/'trajectories.jsonl').open('wb') as target:
        for part in parts:
            with (part/'trajectories.jsonl').open('rb') as source:shutil.copyfileobj(source,target)
    contrasts=[];rng=np.random.default_rng(90210)
    for game in registered['games']:
        for baseline in ('planned','reflex'):
            differences=[];better=same=worse=0
            for seed in registered['seeds']:
                total=0.
                for encounter in range(registered['encounters']):
                    a=indexed[(game,seed,'adaptive',encounter)];b=indexed[(game,seed,baseline,encounter)]
                    diff=b['credits'][b['benchmark']]-a['credits'][a['benchmark']]
                    total+=diff;better+=diff>0;same+=diff==0;worse+=diff<0
                differences.append(total/registered['encounters'])
            values=np.asarray(differences);samples=values[rng.integers(len(values),size=(10000,len(values)))].mean(1)
            contrasts.append(dict(game=game,baseline=baseline,series=len(values),episodes=len(values)*registered['encounters'],
                npc_credit_rate_difference=float(values.mean()),cluster_bootstrap_95=list(map(float,np.quantile(samples,[.025,.975]))),
                episode_better=better,episode_same=same,episode_worse=worse,series_differences=list(map(float,values)),
                interpretation='three-NPC aggregate; read+decision+multi-agent feedback, not isolated individual learning effect'))
    data=dict(plan=plan,matches=runs,summary=summarize(runs,registered['encounters']),contrasts=contrasts,
        elapsed_seconds=max(elapsed),summed_worker_seconds=sum(elapsed),shard_seconds=elapsed,
        trajectory_sha256=hashlib.sha256((root/'trajectories.jsonl').read_bytes()).hexdigest())
    (root/'preregister.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    (root/'evaluation.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    return dict(matches=len(runs),contrasts=contrasts,summary=data['summary'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    print(json.dumps(merge(args.root),ensure_ascii=False,indent=2))
