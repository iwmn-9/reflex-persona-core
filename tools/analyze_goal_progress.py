"""Merge fixed shards; score whole learning series, never individual turns.

Raw traces stay local. Public compact matches retain actual outcome/rule state;
the complete private shard evaluations and traces are pinned by SHA-256.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

VARIANTS=('baseline','progress','continuation','combined')


def load(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(root):
    root=Path(root);plan=load(root/'analysis_preregister.json');matches=[];runs=[]
    for shard in plan['shards']:
        for variant in VARIANTS:
            folder=root/shard['folder']/variant;e=load(folder/'evaluation.json');p=e['plan']
            assert p==load(folder/'preregister.json')
            assert p['seeds']==shard['seeds'] and p['variant']==variant
            for k in ('encounters','games','strong','persona','sources'):assert p[k]==plan[k],k
            assert p['modes']==['adaptive']
            runs.append(dict(folder=str(folder.relative_to(root)).replace('\\','/'),variant=variant,
                matches=len(e['matches']),elapsed_seconds=e['elapsed_seconds'],
                evaluation_sha256=sha(folder/'evaluation.json'),preregister_sha256=sha(folder/'preregister.json'),
                trajectory_sha256=e['trajectory_sha256']))
            for r in e['matches']:
                row={k:v for k,v in r.items() if k!='beliefs'};row['variant']=variant;matches.append(row)
    index={(r['variant'],r['game'],r['seed'],r['encounter']):r for r in matches}
    expected=len(VARIANTS)*len(plan['games'])*len(plan['seeds'])*plan['encounters']
    assert len(index)==len(matches)==expected
    for game in plan['games']:
        for seed in plan['seeds']:
            for encounter in range(plan['encounters']):
                rows=[index[v,game,seed,encounter] for v in VARIANTS]
                for r in rows:
                    assert r['benchmark']==rows[0]['benchmark'] and r['profiles']==rows[0]['profiles']
                    key='prizes' if game=='goofspiel' else 'seen';assert r['final'][key]==rows[0]['final'][key]
                if game=='goofspiel':
                    # The own-continuation intervention is deliberately absent:
                    # exact match of all outcomes is a negative control.
                    for i,j in ((0,2),(1,3)):
                        assert {k:v for k,v in rows[i].items() if k!='variant'}=={k:v for k,v in rows[j].items() if k!='variant'}
    rng=np.random.default_rng(90212);contrasts=[]
    comparisons=(('baseline','progress'),('baseline','continuation'),('baseline','combined'),('continuation','combined'))
    for game in plan['games']:
        for before,after in comparisons:
            series=[];better=same=worse=0
            for seed in plan['seeds']:
                diffs=[]
                for encounter in range(plan['encounters']):
                    a=index[before,game,seed,encounter];b=index[after,game,seed,encounter]
                    d=a['credits'][a['benchmark']]-b['credits'][b['benchmark']]
                    diffs.append(d);better+=d>0;same+=d==0;worse+=d<0
                series.append(float(np.mean(diffs)))
            values=np.array(series);samples=values[rng.integers(len(values),size=(10000,len(values)))].mean(1)
            contrasts.append(dict(game=game,before=before,after=after,npc_credit_rate_difference=float(values.mean()),
                paired_series_bootstrap_95=list(map(float,np.quantile(samples,[.025,.975]))),
                independent_series=len(series),series_differences=series,better=better,same=same,worse=worse))
    summary=[]
    for game in plan['games']:
        for variant in VARIANTS:
            rows=[r for r in matches if r['game']==game and r['variant']==variant];n=len(rows)
            direction=1 if game=='goofspiel' else -1
            profiles={pid:dict(appearances=0,credit=0.,pairwise_benchmark_credit=0.,leader_deficit=0.)
                      for pid in ('growth','steady','care','ego')}
            for r in rows:
                scores=direction*np.array(r['scores']);bench=r['benchmark']
                for actor,pid in r['profiles'].items():
                    actor=int(actor)
                    if actor==bench:continue
                    p=profiles[pid];p['appearances']+=1;p['credit']+=r['credits'][actor]
                    p['pairwise_benchmark_credit']+=float(scores[actor]>scores[bench])+.5*float(scores[actor]==scores[bench])
                    p['leader_deficit']+=float(scores.max()-scores[actor])
            summary.append(dict(game=game,variant=variant,matches=n,
                benchmark_credit=sum(r['credits'][r['benchmark']] for r in rows),
                npc_credit=sum(1-r['credits'][r['benchmark']] for r in rows),
                npc_failures=sum(sum(r['failures'])-r['failures'][r['benchmark']] for r in rows),
                reading_changes=sum(sum(r['reading_changes']) for r in rows),
                guard_changes=sum(sum(r['guards']) for r in rows),profiles=profiles,
                early_benchmark_credit=sum(r['credits'][r['benchmark']] for r in rows if r['encounter']<plan['encounters']//2),
                late_benchmark_credit=sum(r['credits'][r['benchmark']] for r in rows if r['encounter']>=plan['encounters']//2)))
    result=dict(plan=plan,analysis_registration_sha256=sha(root/'analysis_preregister.json'),
                matches=matches,runs=runs,summary=summary,contrasts=contrasts,
                goof_continuation_negative_control_exact=True,
                limitations=plan['limitations']+['no multiplicity correction; 16 learning series, not 1024 independent samples'])
    (root/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(summary=summary,contrasts=contrasts),ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();analyze(a.root)
