"""Paired registered budget/feedback contrasts; no outcome-selected seeds."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(a,b,label):
    a=json.loads((Path(a)/'evaluation.json').read_text(encoding='utf-8'))
    b=json.loads((Path(b)/'evaluation.json').read_text(encoding='utf-8'))
    for key in ('seeds','encounters','games','modes','persona'):assert a['plan'][key]==b['plan'][key],key
    index=lambda x:{(r['game'],r['seed'],r['mode'],r['encounter']):r for r in x['matches']}
    aa=index(a);bb=index(b);assert set(aa)==set(bb)
    rows=[];rng=np.random.default_rng(90211)
    for game in a['plan']['games']:
        series=[];better=same=worse=0;ca=cb=0.
        for seed in a['plan']['seeds']:
            diffs=[]
            for mode in a['plan']['modes']:
                for encounter in range(a['plan']['encounters']):
                    key=(game,seed,mode,encounter);u=aa[key];v=bb[key]
                    assert u['benchmark']==v['benchmark'] and u['profiles']==v['profiles']
                    public_world='prizes' if game=='goofspiel' else 'seen'
                    assert u['final'][public_world]==v['final'][public_world]
                    x=u['credits'][u['benchmark']];y=v['credits'][v['benchmark']];ca+=x;cb+=y
                    diff=x-y;diffs.append(diff);better+=diff>0;same+=diff==0;worse+=diff<0
            series.append(float(np.mean(diffs)))
        val=np.array(series);samples=val[rng.integers(len(val),size=(10000,len(val)))].mean(1)
        n=sum(k[0]==game for k in aa)
        rows.append(dict(comparison=label,game=game,matches_per_version=n,independent_series=len(series),
            benchmark_credit_before=ca,benchmark_credit_after=cb,npc_rate_difference=float(val.mean()),
            cluster_bootstrap_95=list(map(float,np.quantile(samples,[.025,.975]))),
            npc_episode_better=better,npc_episode_same=same,npc_episode_worse=worse,series_differences=series))
    return dict(label=label,before_plan=a['plan'],after_plan=b['plan'],contrasts=rows,
        interpretation='paired series, multi-agent feedback; four series do not establish global strength')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--before',required=True);p.add_argument('--after',required=True)
    p.add_argument('--label',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    result=compare(args.before,args.after,args.label)
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result['contrasts'],indent=2))
