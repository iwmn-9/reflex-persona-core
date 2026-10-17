"""Prequential prediction test before action selection; hidden laws score only."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.adaptive_dynamics import EventLearner


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--frozen',action='store_true')
    p.add_argument('--segmented',action='store_true');a=p.parse_args()
    root=Path(a.root)
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        dispatch(root,Path(__file__).name,['--segmented'] if a.segmented else []);return
    legal=('0','1','2');library={str(i):{str(z):.8 if z==i else .1 for z in range(3)} for i in (0,1)}
    rows=[]
    for regime in ('steady','outside','shift','noise'):
        for seed in range(31,47):
            rng=np.random.default_rng(seed);models=[EventLearner(owner='npc',model_id='shared',names=tuple(library),renewal=v) for v in (False,True)]
            labels=['old','renewal']
            if a.segmented:
                from reflex.segmented_prediction import SegmentedEventLearner
                models.append(SegmentedEventLearner(owner='npc',model_id='shared',names=tuple(library)));labels.append('segmented')
            states=[m.start() for m in models];errors=[[] for m in models];logs=[[] for m in models]
            for tick in range(256):
                top=0 if regime=='steady' else 2 if regime=='outside' else (0,2,1,0)[tick//64]
                truth=np.full(3,1/3) if regime=='noise' else np.array([.8 if z==top else .1 for z in range(3)])
                outcome=legal[rng.choice(3,p=truth)]
                for i,m in enumerate(models):
                    q=m.predict(states[i],(),legal,library)
                    errors[i].append(float(sum((q[z]-truth[j])**2 for j,z in enumerate(legal))))
                    logs[i].append(float(-np.log(q[outcome])))
                    states[i]=m.observe(states[i],(),legal,library,outcome)
            rows.append(dict(regime=regime,seed=seed,brier=errors,logloss=logs,models=labels))
        print(regime,json.dumps({label:float(np.mean([r['brier'][i] for r in rows if r['regime']==regime])) for i,label in enumerate(labels)}),flush=True)
    (root/'evaluation.json').write_text(json.dumps(rows),encoding='utf8')

if __name__=='__main__':main()
