"""Post-hoc diagnosis of persistent-world choices, never a control input.

Compare the selected root and its alternatives under the same learning reflex
continuation. Fresh model seeds separate selection noise from model-law error.
The oracle law is visible only to this offline evaluator. This is NOT the value
of repeatedly replanning for the rest of a full game.
"""
import argparse,copy,json,sys,time,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import digest
from reflex.laboratory import PROFILES
from reflex.decision_loop import DecisionLoop
from reflex.model_observer import resolve_feedback
from tools.continuing_games import ContinuingGame
from tools.run_owned_value import sample,law


def probe(r,tick):
    profile=next(p for p in PROFILES if p['id']==r['profile'])
    g=ContinuingGame(r['game'],profile,r['turns'],r['seed']%2)
    t=r['trace'][tick];w=t['before'];c=g.context(w)
    own=DecisionLoop.from_record(c,t['checkpoint'],predictor=g.actor(c).predictor)
    assert digest(own.record())==t['before_owner']
    req=g.request(c);effective,base=own.preview(req)
    predict=own.predictor.frozen_forecaster();roots={};started=time.monotonic()
    seeds=tuple(range(9101,9133))
    for key in g.keys(w):
        roots[key]={}
        for kind in ('learned','oracle'):
            samples=[]
            for seed in seeds:
                person=copy.deepcopy(own);state=copy.deepcopy(w)
                for step in range(min(16,r['turns']-tick)):
                    if g.terminal(state):break
                    cc=g.context(state,person.state);request=g.request(cc)
                    result=DecisionLoop.decide_batch([(person,request)],False,
                        _counterfactual_roots=(key,) if step==0 else None)[0]
                    action=result['decision']['action_id']
                    p=predict(result['context']) if kind=='learned' else law(state['tick'],r['turns'],r['regime'])
                    u=np.random.default_rng(np.random.SeedSequence([seed,state['tick']])).random()
                    end,flow=g.transition(state,action,sample(p,u))
                    resolve_feedback(person,result['ticket'],g.feedback(state,end,action,result['context'],flow));state=end
                samples.append(g.goal(state).value)
            roots[key][kind]=dict(mean=float(np.mean(samples)),values=samples)
    selected=t['action'];baseline=base['action_id'];contrasts={}
    for kind in ('learned','oracle'):
        delta=np.array(roots[selected][kind]['values'])-np.array(roots[baseline][kind]['values'])
        contrasts[kind]=dict(selected_minus_reflex=float(np.mean(delta)),paired_se=float(np.std(delta,ddof=1)/np.sqrt(len(delta))),
            best_root=max(roots,key=lambda k:roots[k][kind]['mean']))
    return dict(game=r['game'],profile=r['profile'],regime=r['regime'],seed=r['seed'],tick=tick,
        selected=selected,reflex=baseline,selection_forecast=t['forecast'],owner=t['before_owner'],
        roots=roots,contrasts=contrasts,independent_seeds=seeds,seconds=time.monotonic()-started)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',required=True)
    p.add_argument('--seed',type=int,default=170)
    p.add_argument('--frozen',action='store_true');a=p.parse_args();root=Path(a.root);source=Path(a.source).resolve()
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        dispatch(root,Path(__file__).name,['--source',str(source),'--seed',str(a.seed)]);return
    rows=[json.loads(l) for l in source.read_text().splitlines()]
    cases=[('auction','steady','outside',32),('auction','steady','outside',64),('resources','growth','shift-return',40)]
    jobs=[(next(r for r in rows if (r['game'],r['profile'],r['regime'],r['arm'],r['seed'])==(g,p,s,'endpoint',a.seed)),t) for g,p,s,t in cases]
    (root/'registration.json').write_text(json.dumps(dict(cases=cases,seed=a.seed,semantics=__doc__,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()),indent=2),encoding='utf8')
    results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(probe,*j) for j in jobs]):
            result=f.result();results.append(result)
            print(json.dumps({k:result[k] for k in ('game','tick','selected','reflex','contrasts')}),flush=True)
    (root/'evaluation.json').write_text(json.dumps(results,indent=2),encoding='utf8')


if __name__=='__main__':main()
