"""Same-state Monte Carlo variability, separate from long-run competence."""
import argparse,copy,json,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.closed_loop import owned_rollout
from reflex.decision_loop import DecisionLoop
from reflex.core import digest
from reflex.laboratory import PROFILES
from tools.continuing_games import ContinuingGame


def probe(row,tick,bank,mode):
    g=ContinuingGame(row['game'],next(p for p in PROFILES if p['id']==row['profile']),row['turns'],row['seed']%2)
    t=row['trace'][tick];w=t['before'];c=g.context(w)
    own=DecisionLoop.from_record(c,t['checkpoint'],predictor=g.actor(c).predictor)
    before=digest(own.record());seeds=tuple(range(2101+3*bank,2104+3*bank));req=g.request(c);cap={}
    ctrl=owned_rollout(own,observe=g.context,events=g.events,terminal=g.terminal,
        assess=lambda w:g.goal(w).record(),record=lambda w:w,request_for=g.request,
        feedback=g.feedback,model_id=g.model_id,target=g.target,tail_steps=15,seeds=seeds,
        sampling='stratified' if mode=='stratified' else 'independent',
        purpose_estimator='conditional' if mode=='conditional' else 'sampled')
    effective,_=own.preview(req);ctrl.planner(own,req,w,captured=cap)([effective],[])
    assert digest(own.record())==before
    node=cap['node'];f=node['forecast']
    return dict(profile=row['profile'],tick=tick,bank=bank,mode=mode,action=node['action'],owner=before,
        purpose={f.roots[k][0]:v for k,v in f.purpose.items()})


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',required=True)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--frozen',action='store_true')
    p.add_argument('--modes',nargs='+',default=['independent','stratified']);a=p.parse_args();root=Path(a.root)
    source=Path(a.source).resolve()
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        dispatch(root,Path(__file__).name,['--source',str(source),'--workers',str(a.workers),'--modes',*a.modes]);return
    rows=[json.loads(l) for l in source.read_text().splitlines()]
    cases=[(next(r for r in rows if (r['game'],r['profile'],r['arm'],r['regime'])==('auction',profile,'endpoint',regime)),tick)
        for profile,regime,tick in [('steady','outside',64),('growth','shift-return',40)]]
    output=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool,(root/'replicates.jsonl').open('w',encoding='utf8') as f:
        jobs=[pool.submit(probe,r,t,b,m) for r,t in cases for b in range(12) for m in a.modes]
        for future in as_completed(jobs):
            v=future.result();output.append(v);f.write(json.dumps(v)+'\n');f.flush()
            if len(output)%8==0:print(json.dumps(dict(done=len(output),total=len(jobs))),flush=True)
    summary=[]
    for r,t in cases:
        for mode in a.modes:
            subset=[x for x in output if x['profile']==r['profile'] and x['mode']==mode]
            keys=set.intersection(*(set(x['purpose']) for x in subset))
            # Purpose differences, not common reward shifts, determine ranking.
            contrasts={f'{left}/{right}':float(np.var([x['purpose'][left]-x['purpose'][right] for x in subset],ddof=1))
                for left in keys for right in keys if left<right}
            summary.append(dict(profile=r['profile'],tick=t,mode=mode,banks=len(subset),contrast_variance=contrasts,
                choices={k:sum(x['action']==k for x in subset) for k in sorted({x['action'] for x in subset})}))
    (root/'evaluation.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
