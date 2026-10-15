"""Revisit the previous lifecycle failure matrix without changing its protocol.

Seed 29 was already examined: this is regression evidence, NOT new holdout.
The original all-response hypothesis library and frozen-root rollout law remain.
"""
import argparse,json,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.event_reader import EventReader
from tools.run_lifecycle_soak import Adapter,public_context,run_arm,GAMES
from tools.run_lifecycle_choice import PersonaChoiceAdapter
from reflex.laboratory import PROFILES


class BridgeReader(EventReader):
    # Compatibility only for the historical experimental recorder.
    @property
    def tracker(self):return self
    def forecasts(self,c):
        p=self.predict(c);return p,p


class BridgeAdapter(Adapter):
    def legal(self,c):return tuple(sorted(self.rules.symbols(public_context(c)['world'])))
    def supported_models(self,c):
        legal=self.legal(c)
        return {name:{z:sum(p for symbol,p in row.items() if str(min(int(symbol),len(legal)-1))==z) for z in legal}
                for name,row in self.models(c).items()}
    def actor(self,c):
        actor=super().actor(c)
        actor.predictor=BridgeReader(c['scope'],self.symbols(),self.supported_models,self.outcome,
            legal_for=self.legal,model_id='bridge-v1:'+self.rules.model_id,known_conditionals=True)
        return actor


def evaluate(job):
    rows=[]
    for name,cls in (('choice',PersonaChoiceAdapter),('recovery',BridgeAdapter)):
        row=run_arm(job,'channels',adapter_factory=cls);row['arm']=name;rows.append(row)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--frozen',action='store_true');p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();root=Path(a.root)
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        dispatch(root,Path(__file__).name,['--workers',str(a.workers)])
    else:
        jobs=[(g,p,29,r,128,'censored',12) for g in GAMES for p in range(4) for r in ('steady','shift-return')]
        (root/'registration.json').write_text(json.dumps(dict(jobs=jobs,semantics=__doc__),indent=2)+'\n',encoding='utf-8')
        rows=[];started=time.monotonic()
        with (root/'trajectories.jsonl').open('w',encoding='utf-8') as f,ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures={pool.submit(evaluate,j):j for j in jobs}
            for i,fu in enumerate(as_completed(futures)):
                for row in fu.result():rows.append(row);f.write(json.dumps(row,separators=(',',':'))+'\n')
                f.flush();print(json.dumps(dict(done=i+1,total=len(jobs),seconds=round(time.monotonic()-started,1))),flush=True)
        pairs=[]
        for g,p,s,r,n,v,pe in jobs:
            by={row['arm']:row for row in rows if (row['game'],row['profile'],row['schedule'])==(g,PROFILES[p]['id'],r)}
            pairs.append(dict(game=g,profile=PROFILES[p]['id'],regime=r,
                scores={k:row['score'] for k,row in by.items()},delta=by['recovery']['score']-by['choice']['score']))
        result=dict(runs=len(rows),decisions=sum(r['turns'] for r in rows),modeled_decisions=sum(r['modeled_decisions'] for r in rows),pairs=pairs,
            checks={k:sum(r['checks'].get(k,0) for r in rows) for k in rows[0]['checks']})
        (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
