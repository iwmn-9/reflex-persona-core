"""Pre-reveal prediction comparisons on the same actual public action paths.

Learning series, not observer copies or turns, are independent units. This
isolates forecast quality from action-selection changes in the paired matches.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from reflex.strong_search import PublicMemory
from reflex.supported_memory import SupportedPublicMemory
from tools.intervene_goal_progress import position


def run(root):
    root=Path(root);files=[root/f'shard-{i}'/'global'/'trajectories.jsonl' for i in range(4)]
    record=dict(kind='prequential matched-path diagnostic',conditions=['global','supported'],
        source_traces={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        target='next publicly revealed opponent action, scored before update; not terminal win prediction',
        weighting='average observer copies for each public reveal, mean per episode, then mean per learning series',
        partitions=['future opportunity','current-card only','Goof single-bank negative control'],
        uncertainty='10000 paired learning-series bootstrap samples; seed 73118; no multiplicity correction')
    path=root/'prediction_analysis_registration.json';assert not path.exists();path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    groups=defaultdict(lambda:defaultdict(list));count=defaultdict(int)
    for trace in files:
        series=None
        with trace.open(encoding='utf-8') as f:
            for line in f:
                t=json.loads(line);game=t['game'];seed=t['seed'];enc=t['encounter'];bench=t['benchmark']
                if series!=(game,seed):
                    series=(game,seed);memories={kind:[cls(game,a) for a in range(4)] for kind,cls in (('global',PublicMemory),('supported',SupportedPublicMemory))}
                s=position(game,t['before']);actors=range(4) if game=='goofspiel' else (s.turn,)
                phase='bidding' if game=='goofspiel' else ('future_draws' if s.remaining>0 else 'current_card_only')
                for actor in actors:
                    revealed=t['moves'][str(actor)];losses={kind:[] for kind in memories}
                    for observer in range(4):
                        if observer==actor:continue
                        forecasts={kind:memories[kind][observer].predict(s,actor) for kind in memories}
                        if game=='goofspiel':assert forecasts['global']==forecasts['supported']
                        for kind in memories:
                            forecast=forecasts[kind]
                            if len(forecast)>1 and observer!=bench:losses[kind].append(-math.log(forecast[revealed]))
                            memories[kind][observer].observe(s,actor,revealed,f'encounter-{enc}-tick-{t["tick"]}-actor-{actor}')
                    if losses['global']:
                        count[game,phase]+=1
                        for kind in memories:groups[game,phase,kind][seed,enc].append(float(np.mean(losses[kind])))
    rng=np.random.default_rng(73118);summary=[]
    for game,phase in sorted(count):
        per_kind={kind:{seed:[] for seed,enc in groups[game,phase,kind]} for kind in memories}
        for kind in memories:
            for (seed,enc),values in groups[game,phase,kind].items():per_kind[kind][seed].append(float(np.mean(values)))
        seeds=sorted(per_kind['global']);baseline=np.array([np.mean(per_kind['global'][s]) for s in seeds]);candidate=np.array([np.mean(per_kind['supported'][s]) for s in seeds])
        diff=baseline-candidate;sample=diff[rng.integers(len(diff),size=(10000,len(diff)))].mean(1)
        summary.append(dict(game=game,phase=phase,public_nonforced_reveals=count[game,phase],independent_series=len(seeds),
            mean_log_loss={'global':float(baseline.mean()),'supported':float(candidate.mean())},
            log_loss_improvement=float(diff.mean()),paired_series_bootstrap_95=list(map(float,np.quantile(sample,[.025,.975]))),series_improvements=diff.tolist()))
    result=dict(registration=record,summary=summary,limitations=['matched global-policy paths, not actions from the supported controller',
        'better response prediction does not alone prove better move selection','terminal opportunity classes require adapter knowledge'])
    (root/'prediction_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
