"""Post-hoc same-owner diagnosis; oracle probabilities never enter live play."""
import argparse,copy,json,sys,time,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import digest,compile_batch
from reflex.laboratory import PROFILES
from reflex.decision_loop import DecisionLoop
from reflex.closed_loop import owned_rollout
from tools.continuing_games import ContinuingGame
from tools.run_owned_value import law


CASES=(('auction','steady',170,'outside',32),('auction','steady',170,'outside',64),
       ('auction','ego',203,'outside',32),('auction','ego',203,'outside',64),
       ('auction','growth',170,'shift-return',40),('auction','growth',170,'shift-return',72),
       ('resources','growth',170,'shift-return',40),('gambling','care',203,'outside',64))

def probe(row,tick):
    profile=next(p for p in PROFILES if p['id']==row['profile'])
    g=ContinuingGame(row['game'],profile,row['turns'],row['seed']%2)
    t=row['trace'][tick];w=t['before'];c=g.context(w)
    own=DecisionLoop.from_record(c,t['checkpoint'],predictor=g.actor(c).predictor)
    assert digest(own.record())==t['before_owner']
    effective,base=own.preview(g.request(c));before=digest(own.record())
    args=dict(observe=g.context,events=g.events,terminal=g.terminal,assess=lambda w:g.goal(w).record(),
        record=lambda w:w,request_for=g.request,feedback=g.feedback,model_id=g.model_id,target=g.target,tail_steps=15)
    results={};started=time.monotonic()
    banks=[('original',(701,702,703)),('independent',(1101,1102,1103)),
           ('expanded',tuple(range(1101,1113))),('truth_current',tuple(range(1101,1113)))]
    for label,seeds in banks:
        actor=copy.deepcopy(own)
        if label=='truth_current':
            true=law(tick,row['turns'],row['regime'])
            actor.predictor.frozen_forecaster=lambda:lambda c:copy.deepcopy(true)
        ctrl=owned_rollout(actor,**args,seeds=seeds);cap={};request=g.request(c)
        result=DecisionLoop.decide_batch([(actor,request)],False,ctrl.planner(actor,request,w,captured=cap))[0]
        f=cap['node']['forecast'];b=compile_batch(f.contexts,outcome_limit=f.outcome_limit)
        d=own.policy.decide(b,False,return_frame=f.return_frame)
        results[label]=dict(action=result['decision']['action_id'],purpose={f.roots[k][0]:v for k,v in f.purpose.items()},
            scores={f.roots[k][0]:float(v) for k,v in zip(b.ids[0],d.scores[0])},
            tier=[f.roots[k][0] for k in f.roots],band=f.max_regret,
            law=own.predictor.predict(c) if label!='truth_current' else true,
            simulated_steps=ctrl.frontier.simulated_steps)
        assert digest(own.record())==before
    assert results['original']['action']==t['action']
    return dict(game=row['game'],profile=row['profile'],seed=row['seed'],regime=row['regime'],tick=tick,
        original=t['action'],reflex=base['action_id'],owner=before,results=results,seconds=time.monotonic()-started)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--sources',nargs='+',required=True)
    p.add_argument('--workers',type=int,default=3);p.add_argument('--frozen',action='store_true');a=p.parse_args()
    root=Path(a.root);sources=[Path(s).resolve() for s in a.sources]
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        dispatch(root,Path(__file__).name,['--sources',*map(str,sources),'--workers',str(a.workers)]);return
    rows=[json.loads(l) for s in sources for l in s.read_text().splitlines()]
    jobs=[]
    for game,profile,seed,regime,tick in CASES:
        r=next(r for r in rows if (r['game'],r['profile'],r['seed'],r['regime'],r['arm'])==(game,profile,seed,regime,'endpoint'))
        jobs.append((r,tick))
    (root/'registration.json').write_text(json.dumps(dict(cases=CASES,source_sha256=[hashlib.sha256(s.read_bytes()).hexdigest() for s in sources],
        semantics=__doc__),indent=2),encoding='utf8')
    out=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool,(root/'diagnosis.jsonl').open('w',encoding='utf8') as f:
        for future in as_completed([pool.submit(probe,*j) for j in jobs]):
            r=future.result();out.append(r);f.write(json.dumps(r)+'\n');f.flush()
            print(json.dumps({k:r[k] for k in ('game','profile','tick','original','reflex')}|{'choices':{k:v['action'] for k,v in r['results'].items()}}),flush=True)
    (root/'evaluation.json').write_text(json.dumps(out,indent=2),encoding='utf8')

if __name__=='__main__':main()
