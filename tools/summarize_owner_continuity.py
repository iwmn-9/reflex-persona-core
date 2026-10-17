"""Matched control-cadence comparison, not a comparison of prediction alone."""
import argparse,json,hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np


def summarize(root):
    paths=sorted(Path(root).glob('**/summary.json'))
    groups=defaultdict(dict);sources=[]
    for path in paths:
        if '_source' in path.parts:continue
        rows=json.loads(path.read_text())
        if not isinstance(rows,list):continue
        for r in rows:
            if not all(k in r for k in ('game','profile','seed','regime','arm','interval','score')):continue
            key=tuple(r[k] for k in ('game','profile','seed','regime','turns','tail_steps'))
            arm=({'receding_posterior':'endpoint_posterior','receding':'endpoint'}.get(r['arm'],r['arm']),r['interval'])
            if arm in groups[key]:raise ValueError('duplicate experimental condition')
            groups[key][arm]=r
        sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    pairs=[]
    for condition,arms in groups.items():
        for baseline,candidate in [(('endpoint',7),('endpoint_posterior',1)),
                (('endpoint_posterior',7),('endpoint_posterior',1)),
                (('endpoint',7),('endpoint',1)),(('endpoint',1),('endpoint_posterior',1))]:
            if baseline not in arms or candidate not in arms:continue
            delta=arms[candidate]['score']-arms[baseline]['score']
            pairs.append(dict(zip(('game','profile','seed','regime','turns','tail_steps'),condition),
                baseline=list(baseline),candidate=list(candidate),delta=delta))
    summaries=[]
    for base,candidate in sorted({(tuple(p['baseline']),tuple(p['candidate'])) for p in pairs}):
        for game in sorted({p['game'] for p in pairs}):
            ps=[p for p in pairs if tuple(p['baseline'])==base and tuple(p['candidate'])==candidate and p['game']==game]
            if not ps:continue
            deltas=[p['delta'] for p in ps]
            summaries.append(dict(baseline=base,candidate=candidate,game=game,pairs=len(ps),
                better=sum(d>1e-12 for d in deltas),same=sum(abs(d)<=1e-12 for d in deltas),
                worse=sum(d< -1e-12 for d in deltas),mean=float(np.mean(deltas)),worst=min(deltas)))
    out=dict(semantics=__doc__,sources=sources,conditions=len(groups),pairs=pairs,summaries=summaries)
    (Path(root)/'evaluation.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(conditions=len(groups),summaries=summaries),indent=2))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();summarize(a.root)
