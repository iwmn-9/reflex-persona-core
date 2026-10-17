"""Report every persona x regime cell, without replacing observed returns.

World seeds are the resampling unit. The auction oracle accounting is a
post-hoc companion diagnostic, not the registered acceptance measure.
"""
import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np


def stats(rows, field):
    clusters = defaultdict(list)
    for row in rows:
        clusters[row['seed']].append(float(row[field]))
    assert len({len(v) for v in clusters.values()}) == 1
    means = np.array([np.mean(clusters[s]) for s in sorted(clusters)])
    ci = None
    if len(means) > 1:
        rng = np.random.default_rng(73019)
        boots = means[rng.integers(0, len(means), (20000, len(means)))].mean(axis=1)
        ci = np.quantile(boots, [.025, .975]).tolist()
    values = [float(row[field]) for row in rows]
    return dict(pairs=len(rows), world_seeds=len(means), mean=float(np.mean(values)),
        worst=min(values), best=max(values), better=sum(v>1e-12 for v in values),
        same=sum(abs(v)<=1e-12 for v in values), worse=sum(v< -1e-12 for v in values),
        seed_means=[dict(seed=s, mean=float(np.mean(clusters[s]))) for s in sorted(clusters)],
        seed_cluster_percentile_95=ci)


def main():
    p=argparse.ArgumentParser();p.add_argument('--primary',required=True)
    p.add_argument('--auction-accounting',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();primary=json.loads(Path(a.primary).read_text())
    auction=json.loads(Path(a.auction_accounting).read_text())
    assert primary['complete'] and primary['registered_runs']==560
    assert len(primary['pairs'])==448 and len(auction['pairs'])==256
    keys=('profile','seed','regime','baseline')
    indexed={tuple(r[k] for k in keys):r for r in auction['pairs']}
    assert len(indexed)==256
    rows=[]
    for original in primary['pairs']:
        row=dict(original)
        if row['game']=='auction':
            d=indexed[tuple(row[k] for k in keys)]
            assert abs(row['delta']-d['score_delta'])<1e-12
            row.update({k:d[k] for k in ('oracle_adjusted_return_delta','innovation_delta')})
        rows.append(row)
    groups=[]
    for dimensions in (('game','baseline'),('game','profile','baseline'),
                       ('game','profile','regime','baseline')):
        grouped=defaultdict(list)
        for row in rows:grouped[tuple(row[k] for k in dimensions)].append(row)
        for key,rs in sorted(grouped.items()):
            item=dict(zip(dimensions,key));item['observed_score']=stats(rs,'delta')
            assert item['observed_score']['world_seeds']==(8 if item['game']=='auction' else 2)
            if item['game']=='auction':
                item['oracle_adjusted_return']=stats(rs,'oracle_adjusted_return_delta')
                item['innovation']=stats(rs,'innovation_delta')
            groups.append(item)
    result=dict(complete_primary_matrix=True,groups=groups,
        sources={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest()
                 for p in (a.primary,a.auction_accounting)},
        caveats=['All persona-regime cells are retained, including losses.',
          'Small seed-cluster intervals are descriptive, not a human-level intelligence certificate.',
          'Do not pool scores across games; repeated comparator arms are not independent worlds.',
          'Oracle accounting uses inaccessible true dynamics only after execution and cannot attribute personality costs.'])
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(groups=len(groups),negative_observed_groups=sum(g['observed_score']['mean']< -1e-12 for g in groups))))


if __name__=='__main__':main()
