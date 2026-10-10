"""Matched terminal outcomes and intervention exposure for the expiry phase."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def analyze(root):
    root=Path(root);matches=[];runs=[];plans=[];exposure=[];worlds={}
    for shard in range(4):
        folder=root/f'shard-{shard}';plan=load(folder/'analysis_preregister.json');plans.append(plan)
        base=plans[0]['seeds'][0]
        assert plan['seeds']==list(range(base+4*shard,base+4*shard+4))
        for variant in plan['variants']:
            run=folder/variant;e=load(run/'evaluation.json');p=e['plan']
            assert p==load(run/'preregister.json')
            for k in ('seeds','encounters','games','strong','persona','sources'):assert p[k]==plan[k],k
            assert p['variant']==variant and p['modes']==['adaptive']
            runs.append(dict(folder=str(run.relative_to(root)).replace('\\','/'),matches=len(e['matches']),
                evaluation_sha256=sha(run/'evaluation.json'),preregister_sha256=sha(run/'preregister.json'),
                trajectory_sha256=e['trajectory_sha256']))
            for r in e['matches']:
                r={k:v for k,v in r.items() if k!='beliefs'};r['variant']=variant;matches.append(r)
            count=changed=expiry=0;ordered={};previous={}
            with (run/'trajectories.jsonl').open(encoding='utf-8') as f:
                for line in f:
                    t=json.loads(line);key=(t['seed'],t['encounter']);b=t['before']
                    if previous.get(key)!=b['card']:
                        ordered.setdefault(key,[]).append(b['card']);previous[key]=b['card']
                    for a,d in t['decisions'].items():
                        if int(a)==t['benchmark']:continue
                        count+=1;facts=d['context']['facts']
                        if 'expired_proxies' in facts:
                            assert variant==plan['variants'][1] and b['remaining']==0
                            if variant=='certified_expiry':assert b['chips'][(int(a)+1)%4]==0 and 'expiry_certificate' in facts
                            expiry+=1;changed+=d['action']!=d['search']['frozen_action']
            for key,deck in ordered.items():
                assert len(deck)==len(set(deck))==24
                if key in worlds:assert deck==worlds[key]
                else:worlds[key]=deck
            exposure.append(dict(shard=shard,variant=variant,npc_choices=count,expired_proxy_choices=expiry,
                expiry_choices_different_from_unread_reflex=changed))
    assert all(p['sources']==plans[0]['sources'] for p in plans)
    index={(r['variant'],r['seed'],r['encounter']):r for r in matches};assert len(index)==len(matches)==128
    summary=[]
    for variant in plans[0]['variants']:
        rows=[r for r in matches if r['variant']==variant];profiles={}
        for pid in ('growth','steady','care','ego'):
            own=[(r,int(a)) for r in rows for a,p in r['profiles'].items() if p==pid]
            profiles[pid]=dict(appearances=len(own),credit=sum(r['credits'][a] for r,a in own),
                pairwise_benchmark_credit=sum(float(r['scores'][a]<r['scores'][r['benchmark']])+.5*float(r['scores'][a]==r['scores'][r['benchmark']]) for r,a in own),
                leader_deficit=sum(r['scores'][a]-min(r['scores']) for r,a in own))
        summary.append(dict(variant=variant,matches=len(rows),benchmark_credit=sum(r['credits'][r['benchmark']] for r in rows),
            npc_credit=sum(1-r['credits'][r['benchmark']] for r in rows),profiles=profiles,
            npc_failures=sum(sum(r['failures'])-r['failures'][r['benchmark']] for r in rows)))
    series=[];better=same=worse=0
    after=plans[0]['variants'][1]
    for seed in range(base,base+16):
        delta=[]
        for enc in range(4):
            a=index['progress',seed,enc];b=index[after,seed,enc]
            assert a['profiles']==b['profiles'] and a['benchmark']==b['benchmark']
            d=a['credits'][a['benchmark']]-b['credits'][b['benchmark']]
            delta.append(d);better+=d>0;same+=d==0;worse+=d<0
        series.append(float(np.mean(delta)))
    v=np.array(series);rng=np.random.default_rng(plans[0].get('bootstrap_seed',90213));boot=v[rng.integers(16,size=(10000,16))].mean(1)
    result=dict(plans=plans,runs=runs,matches=matches,summary=summary,exposure=exposure,
        ordered_worlds_checked=len(worlds),contrast=dict(npc_credit_rate_difference=float(v.mean()),
        paired_series_bootstrap_95=list(map(float,np.quantile(boot,[.025,.975]))),series_differences=series,
        independent_series=16,better=better,same=same,worse=worse),limitations=plans[0]['limitations'])
    (root/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(summary=summary,contrast=result['contrast'],exposure=exposure),indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();analyze(a.root)
