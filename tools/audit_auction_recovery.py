"""Audit registered integrated runs and separate prediction from control losses."""
import argparse,json,sys,hashlib
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.audit_owned_value import audit
from tools.continuing_games import ContinuingGame
from reflex.laboratory import PROFILES


def evaluate(root,reference=None,*,allow_abandoned=False):
    checked=audit(root,allow_abandoned=allow_abandoned)
    rows=[json.loads(l) for l in (root/'trajectories.jsonl').read_text().splitlines()]
    groups=defaultdict(dict)
    for r in rows:groups[tuple(r[k] for k in ('game','profile','seed','regime'))][r['arm']]=r
    if reference is not None:
        for r in [json.loads(l) for l in (reference/'trajectories.jsonl').read_text().splitlines()]:
            key=tuple(r[k] for k in ('game','profile','seed','regime'))
            if r['arm']=='endpoint' and key in groups:
                assert 'endpoint' not in groups[key],'ambiguous baseline'
                for candidate in groups[key].values():
                    assert all(candidate.get(k,0)==r.get(k,0) for k in ('turns','interval','tail_steps','offset'))
                groups[key]['endpoint']=r
    pairs=[]
    for key,arms in groups.items():
        base=arms['endpoint']
        for name,r in arms.items():
            if name=='endpoint':continue
            delta=r['score']-base['score']
            pairs.append(dict(zip(('game','profile','seed','regime'),key),candidate=name,delta=delta,
                better=int(delta>1e-12),same=int(abs(delta)<=1e-12),worse=int(delta< -1e-12)))
            for a,b in zip(base['trace'],r['trace']):
                assert a['response']==b['response'],'different actual information'
                if name in ('endpoint_stratified','endpoint_conditional'):assert a['law']==b['law'],'estimator changed factual predictor'
    summaries=[]
    for candidate in sorted({p['candidate'] for p in pairs}):
        for game in (*sorted({p['game'] for p in pairs}),'all'):
            ps=[p for p in pairs if p['candidate']==candidate and (game=='all' or p['game']==game)]
            if not ps:continue
            summaries.append(dict(candidate=candidate,game=game,pairs=len(ps),
                **{k:sum(p[k] for p in ps) for k in ('better','same','worse')},
                mean_delta=float(np.mean([p['delta'] for p in ps])),worst=min(p['delta'] for p in ps)))
    predictions=[]
    for r in rows:
        ts=r['trace'];switches=[i for i in range(1,len(ts)) if ts[i]['truth']!=ts[i-1]['truth']]
        game=ContinuingGame(r['game'],next(p for p in PROFILES if p['id']==r['profile']),r['turns'],r['seed']%2)
        initial=game.goal(ts[0]['before']).value
        segments=[]
        for start in [0,*switches]:
            segment=ts[start:min(start+16,len(ts))]
            segments.append(dict(start=start,goal_gain=segment[-1]['goal']-(ts[start-1]['goal'] if start else initial),
                brier=float(np.mean([t['brier'] for t in segment]))))
        preds=[t for t in ts if 'forecast_error' in t]
        predictions.append(dict(**{k:r[k] for k in ('game','profile','seed','regime','arm')},
            segments=segments,brier=float(np.mean([t['brier'] for t in ts])),
            forecast_mae=float(np.mean([abs(t['forecast_error']) for t in preds])) if preds else None,
            prediction_movement=float(np.mean([sum(abs(b['law'][k]-a['law'][k]) for k in a['law'])/2 for a,b in zip(ts,ts[1:])])),
            actions=dict(Counter(t['action'] for t in ts))))
    result=dict(checks=checked['checks'],complete=checked['complete'],early_stop=checked['early_stop'],pairs=pairs,summaries=summaries,predictions=predictions,
        caveat='Paired condition counts share real seeds. Endpoint proxy and finite policy continuation remain assumptions.')
    if reference is not None:
        result['reference']=dict(study=reference.name,sha256=hashlib.sha256((reference/'trajectories.jsonl').read_bytes()).hexdigest())
    (root/'evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(summaries,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--reference');p.add_argument('--allow-abandoned',action='store_true');a=p.parse_args()
    evaluate(Path(a.root),Path(a.reference) if a.reference else None,allow_abandoned=a.allow_abandoned)
