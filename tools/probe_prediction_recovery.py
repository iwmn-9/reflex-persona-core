"""Prequential non-game controls: change, return, isolated shock, and noise.

Truth is diagnostic only. Identical streams reach both original forecasts and
the existing EventLearner. No utility/score/character or game labels are inputs.
"""
import argparse, copy, json, math, random, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.adaptive_dynamics import EventLearner
from reflex.opponent_beliefs import HypothesisTracker
from collections import Counter,deque,defaultdict


def experiment(segmented=False):
    rows=[]
    for seed in (211,307,401,503,601,701,809,907):
        for size in (3,5):
            legal=tuple(str(i) for i in range(size))
            library={str(k):{z:.94 if int(z)==k else .06/(size-1) for z in legal} for k in (0,1)}
            for regime in ('stable','shift-return','outside','outlier','noise','cue-reversal'):
                learner=EventLearner(owner='probe',model_id='opaque-symbols-v1',names=tuple(library),factors=('cue',))
                memory=learner.start();tracker=HypothesisTracker(tuple(library));recent=deque(maxlen=4)
                if segmented:
                    from reflex.segmented_prediction import SegmentedEventLearner
                    alternative=SegmentedEventLearner(owner='probe',model_id='opaque-symbols-v1',names=tuple(library),factors=('cue',))
                    alternate_memory=alternative.start()
                rng=random.Random(seed);trace=[]
                for tick in range(512):
                    stage=tick//128;cue=str((tick//3)%2);main=0
                    if regime=='shift-return':main=(0,size-1,1,0)[stage]
                    elif regime=='outside':main=size-1
                    elif regime=='cue-reversal':main=size-1 if int(cue)==int(stage>=2) else 0
                    p={z:1/size if regime=='noise' else (.85 if int(z)==main else .15/(size-1)) for z in legal}
                    predictions=dict(prior={z:(Counter(recent)[z]+1)/(len(recent)+size) if len(recent)>=3 else 1/size for z in legal},
                        candidate=tracker.snapshot().predict(library),recovery=learner.predict(memory,(cue,),legal,library))
                    if segmented:predictions['segmented']=alternative.predict(alternate_memory,(cue,),legal,library)
                    u=rng.random();total=0.;outcome=legal[-1]
                    for z,v in p.items():
                        total+=v
                        if u<total:outcome=z;break
                    if regime=='outlier' and tick==256:outcome=legal[-1]
                    trace.append(dict(tick=tick,outcome=outcome,truth=p,predictions=predictions,
                        errors={name:sum((row[z]-p[z])**2 for z in legal)/size for name,row in predictions.items()}))
                    memory=learner.observe(memory,(cue,),legal,library,outcome)
                    if segmented:alternate_memory=alternative.observe(alternate_memory,(cue,),legal,library,outcome)
                    tracker.observe(library,outcome,str(tick));recent.append(outcome)
                rows.append(dict(seed=seed,size=size,regime=regime,trace=trace))
    grouped=defaultdict(list)
    for r in rows:
        for t in r['trace']:
            for name,e in t['errors'].items():grouped[(r['regime'],r['size'],name,t['tick']//128)].append(e)
    return dict(streams=len(rows),observations=sum(len(r['trace']) for r in rows),
        phases=[dict(regime=k[0],size=k[1],arm=k[2],phase=k[3],error=sum(v)/len(v)) for k,v in sorted(grouped.items())],rows=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--frozen',action='store_true')
    p.add_argument('--segmented',action='store_true');a=p.parse_args();root=Path(a.root)
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        dispatch(root,Path(__file__).name,['--segmented'] if a.segmented else [])
    else:
        root.mkdir(parents=True,exist_ok=True);r=experiment(a.segmented)
        (root/'streams.json').write_text(json.dumps(r,separators=(',',':'))+'\n',encoding='utf-8')
        (root/'summary.json').write_text(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2)+'\n',encoding='utf-8')
        print(json.dumps(dict(streams=r['streams'],observations=r['observations'])))
