"""Compare complete registered executions without selecting favorable shards.

Inputs are explicit study directories (each has registration and audit files).
Validation uses original-platform audits plus exact source/trajectory hashes;
it does not silently rerun Linux floating-point learning on another platform.
"""
import argparse,json,hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(studies,comparisons):
    groups=defaultdict(dict);sources=[];checks=Counter();total=0
    for study in map(Path,studies):
        audit=json.loads((study/'audit.json').read_text())
        registration=json.loads((study/'registration.json').read_text())
        manifest=json.loads((study/'source_snapshot.json').read_text())
        assert audit['complete'] and audit['source_freeze_verified']
        assert all(sha(study/'_source'/name)==value for name,value in manifest.items())
        assert sha(study/'trajectories.jsonl')==audit['trajectory_sha256']
        rows=[json.loads(line) for line in (study/'trajectories.jsonl').read_text().splitlines()]
        assert len(rows)==len(registration['jobs'])*len(registration['arms'])==audit['completed_runs']
        total+=len(rows);checks.update(audit['checks'])
        environment=json.loads((study/'environment.json').read_text())
        for row in rows:
            key=tuple(row[k] for k in ('game','profile','seed','regime','turns','tail_steps','offset'))
            arm=row['arm']+'@'+str(row['interval'])
            assert arm not in groups[key],('duplicate execution',key,arm)
            groups[key][arm]=(row,environment)
        sources.append(dict(study=str(study),trajectory_sha256=audit['trajectory_sha256'],
            source_manifest_sha256=sha(study/'source_snapshot.json'),stage=registration['stage'],runs=len(rows)))
    pairs=[]
    for condition,arms in sorted(groups.items()):
        for baseline,candidate in comparisons:
            if baseline not in arms or candidate not in arms:continue
            (base,be),(new,ne)=arms[baseline],arms[candidate]
            assert all(be[k]==ne[k] for k in ('python','numpy','system','machine')),'matched execution platform required'
            n=min(base['executed'],new['executed'])
            assert [t['response'] for t in base['trace'][:n]]==[t['response'] for t in new['trace'][:n]]
            delta=new['score']-base['score']
            pairs.append(dict(zip(('game','profile','seed','regime','turns','tail_steps','offset'),condition),
                baseline=baseline,candidate=candidate,baseline_score=base['score'],candidate_score=new['score'],delta=delta,
                better=int(delta>1e-12),same=int(abs(delta)<=1e-12),worse=int(delta< -1e-12)))
    summaries=[]
    for baseline,candidate in comparisons:
        matched=[p for p in pairs if (p['baseline'],p['candidate'])==(baseline,candidate)]
        for game in sorted({p['game'] for p in matched}):
            gs=[p for p in matched if p['game']==game]
            for field,values in [('all',['all']),('profile',sorted({p['profile'] for p in gs})),('regime',sorted({p['regime'] for p in gs}))]:
                for value in values:
                    ps=[p for p in gs if field=='all' or p[field]==value]
                    clusters=[[p['delta'] for p in ps if p['seed']==s] for s in sorted({p['seed'] for p in ps})]
                    assert len({len(c) for c in clusters})==1,'balanced seed clusters required'
                    means=np.array([np.mean(c) for c in clusters])
                    ci=None
                    if len(means)>1:
                        rng=np.random.default_rng(73019)
                        boots=means[rng.integers(0,len(means),(20000,len(means)))].mean(axis=1)
                        ci=np.quantile(boots,[.025,.975]).tolist()
                    summaries.append(dict(baseline=baseline,candidate=candidate,game=game,group_by=field,group=value,
                        pairs=len(ps),world_seeds=len(means),mean_delta=float(np.mean([p['delta'] for p in ps])),
                        worst=min(p['delta'] for p in ps),**{k:sum(p[k] for p in ps) for k in ('better','same','worse')},
                        seed_cluster_percentile_95=ci))
    return dict(runs=total,conditions=len(groups),sources=sources,checks=dict(checks),pairs=pairs,summaries=summaries,
        caveats=['Only supplied complete studies are included; the caller must verify the full registered study matrix separately.',
                 'Scores are game-defined purpose, not win rate. No pooled cross-game strength score.',
                 'Seed-cluster intervals are descriptive, conditional on these synthetic rules/regimes; few clusters remain weak evidence.',
                 'Unchanged personality axes do not by themselves establish believable behavior or human-level intelligence.'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--studies',nargs='+',required=True)
    p.add_argument('--compare',nargs=2,action='append',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=compare(a.studies,a.compare)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(runs=result['runs'],conditions=result['conditions'],
        summaries=[s for s in result['summaries'] if s['group_by']=='all']),indent=2))
