"""Public-action learning under context-specific and abruptly changing policies.

Scripted opponent policies are evaluator-only. Learners see each public position
and a later legal action, never the generating margin or the switch schedule.
This is a prediction mechanism stress test, not a game strength result.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from reflex.board_models import ThanksPosition,card_points
from reflex.strong_search import PublicMemory
from reflex.supported_memory import SupportedPublicMemory


def run(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    plan=dict(version='public-supported-drift-v1',series=32,seed=74103,observations_per_series=128,
        paths='64 generated public positions repeated after policy switch; alternating future-draw/current-card-only support',
        policies_before={'future_draws':-2,'current_card_only':9},policies_after={'future_draws':9,'current_card_only':-2},
        target='pre-reveal legal action log loss; early 8 versus late 8 observations of each context after switch',
        sources={str(p.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((Path(__file__).resolve().parents[1]/'reflex').glob('*.py'))},
        limitations=['scripted switching policies, not natural opponents or win rates','same supplied public support classes'])
    path=root/'preregister.json';assert not path.exists();path.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    rng=np.random.default_rng(plan['seed']);rows=[]
    for series in range(plan['series']):
        positions=[]
        for j in range(64):
            remaining=8 if j%2==0 else 0;seen=tuple(sorted(map(int,rng.permutation(np.arange(3,36))[:24-remaining])));card=int(rng.choice(seen));cards=[[],[],[],[]]
            for c,a in zip([c for c in seen if c!=card],rng.integers(4,size=len(seen)-1)):cards[int(a)].append(c)
            pot=int(rng.integers(9));chips=list(map(int,rng.multinomial(44-pot,[.25]*4)))
            if chips[0]==0:other=max(range(1,4),key=lambda a:chips[a]);chips[other]-=1;chips[0]=1
            positions.append(ThanksPosition(tuple(tuple(c) for c in cards),tuple(chips),0,card,pot,seen,remaining,(0,)*4))
        memories={'global':PublicMemory('no_thanks',1),'supported':SupportedPublicMemory('no_thanks',1)};losses={k:[] for k in memories}
        for i,s in enumerate(positions*2):
            opportunity='future_draws' if s.remaining else 'current_card_only';margin=plan['policies_before' if i<64 else 'policies_after'][opportunity]
            added=card_points(s.cards[0]+(s.card,))-card_points(s.cards[0])-s.pot;revealed='TAKE' if added<=margin else 'PASS'
            for kind,memory in memories.items():
                prediction=memory.predict(s,0);losses[kind].append(-math.log(prediction[revealed]))
                memory.observe(s,0,revealed,f'public-{i}')
        rows.append(dict(series=series,losses=losses))
    summary=[];rng=np.random.default_rng(74104)
    for kind in ('global','supported'):
        for offset,opportunity in ((0,'future_draws'),(1,'current_card_only')):
            early=np.array([np.mean(r['losses'][kind][64+offset:80:2]) for r in rows]);late=np.array([np.mean(r['losses'][kind][112+offset:128:2]) for r in rows]);improvement=early-late
            samples=improvement[rng.integers(len(rows),size=(10000,len(rows)))].mean(1)
            summary.append(dict(kind=kind,opportunity=opportunity,new_policy_first_8_log_loss=float(early.mean()),new_policy_last_8_log_loss=float(late.mean()),
                recovery_improvement=float(improvement.mean()),paired_series_bootstrap_95=list(map(float,np.quantile(samples,[.025,.975])))))
    result=dict(plan=plan,series=rows,summary=summary,only_revealed_actions_learned=True)
    (root/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
