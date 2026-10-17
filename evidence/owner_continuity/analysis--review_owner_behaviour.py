"""Post-hoc, descriptive phase and action review; no policy selection or tuning."""
import argparse,json,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from tools.compare_owned_studies import compare
from reflex.laboratory import PROFILES
from tools.continuing_games import ContinuingGame
p=argparse.ArgumentParser();p.add_argument('--studies',nargs='+',required=True);p.add_argument('--output',required=True)
a=p.parse_args();validated=compare(a.studies,[])
runs=[]
for study in map(Path,a.studies):
    for row in map(json.loads,(study/'trajectories.jsonl').read_text().splitlines()):
        key=tuple(row[k] for k in ('game','profile','seed','regime','turns'))
        arm=row['arm']+'@'+str(row['interval'])
        phases=[];trace=row['trace'];start=0
        game=ContinuingGame(row['game'],next(q for q in PROFILES if q['id']==row['profile']),row['turns'],row['seed']%2)
        while start<len(trace):
            stop=start+1
            while stop<len(trace) and trace[stop]['truth']==trace[start]['truth']:stop+=1
            # Report both the first and second half of each actual regime.
            middle=start+(stop-start)//2
            for label,lo,hi in [('early',start,middle),('late',middle,stop)]:
                if hi==lo:continue
                ts=trace[lo:hi];initial=trace[lo-1]['goal'] if lo else game.goal(trace[0]['before']).value
                phases.append(dict(start=lo,end=hi,phase_start=start,half=label,
                    gain=ts[-1]['goal']-initial,brier=float(np.mean([t['brier'] for t in ts])),
                    actions=dict(Counter(t['action'] for t in ts)),planned=sum(t['planned'] for t in ts)))
            start=stop
        runs.append(dict(zip(('game','profile','seed','regime','turns'),key),arm=arm,score=row['score'],phases=phases,
            actions=dict(Counter(t['action'] for t in trace)),mode_changes=sum(x['mode']!=y['mode'] for x,y in zip(trace,trace[1:]))))
diversity=[];personas=defaultdict(list)
for row in runs:
    personas[(row['game'],row['seed'],row['regime'],row['turns'],row['arm'])].append(row)
for key,rows in personas.items():
    for i,left in enumerate(rows):
        for right in rows[i+1:]:
            if left['profile']==right['profile']:continue
            ln=sum(left['actions'].values());rn=sum(right['actions'].values())
            keys=set(left['actions'])|set(right['actions'])
            tv=.5*sum(abs(left['actions'].get(k,0)/ln-right['actions'].get(k,0)/rn) for k in keys)
            diversity.append(dict(zip(('game','seed','regime','turns','arm'),key),
                profiles=[left['profile'],right['profile']],action_frequency_distance=tv))
result=dict(sources=validated['sources'],runs=runs,personality_action_differences=diversity,
    limitations=['Post-hoc phase review, not an independent acceptance test or causal estimate of learning.',
                'Personality runs share exogenous responses but can visit different states. Action frequency difference is not psychological validity.',
                'A regime can span the entire run. Improvement between its halves also reflects resource changes and public time.'])
Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(runs=len(runs),personality_comparisons=len(diversity))))
