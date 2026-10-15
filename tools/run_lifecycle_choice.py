"""Separate probability validation from frozen personal choice validation.

No probability/decision threshold, personality weight, rule or action changes.
The joint-persona arm isolates the metric; choice-persona isolates the gate.
"""
import argparse,json,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.decision_loop import DecisionLoop
from reflex.laboratory import PROFILES
from tools.run_lifecycle_soak import run_arm,GAMES
from tools.run_lifecycle_support import AvailabilityAdapter,LegacyAvailabilityAdapter


class PersonaJointAdapter(AvailabilityAdapter):
    adoption='joint'
    def actor(self,c):
        base=super().actor(c)
        return DecisionLoop(c,policy=base.policy,predictor=base.predictor,read_control=base.read_control,
            pressure=base.pressure,progress=base.progress,decision_metric='persona',response_adoption=self.adoption)


class PersonaChoiceAdapter(PersonaJointAdapter):adoption='persona_choice'


def evaluate(job,holdout):
    variants=[('persona_joint',PersonaJointAdapter),('persona_choice',PersonaChoiceAdapter)]
    if holdout:variants=[('purpose_joint',AvailabilityAdapter),('legacy_support',LegacyAvailabilityAdapter),*variants]
    rows=[]
    for name,factory in variants:
        row=run_arm(job,'channels',adapter_factory=factory);row['arm']=name;rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--frozen',action='store_true')
    p.add_argument('--turns',type=int,default=128);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--holdout',action='store_true');a=p.parse_args();root=Path(a.root)
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        args=['--turns',str(a.turns),'--workers',str(a.workers)]
        if a.holdout:args.append('--holdout')
        dispatch(root,Path(__file__).name,args);return
    root.mkdir(parents=True,exist_ok=True);seed=29 if a.holdout else 17
    jobs=[(g,p,seed,mode,a.turns,'censored',12) for g in GAMES for p in range(4) for mode in ('steady','shift-return')]
    (root/'registration.json').write_text(json.dumps(dict(jobs=jobs,holdout=a.holdout,semantics=__doc__),indent=2)+'\n',encoding='utf-8')
    started=time.monotonic();rows=[]
    with (root/'trajectories.jsonl').open('w',encoding='utf-8') as f,ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(evaluate,j,a.holdout):j for j in jobs}
        for i,fu in enumerate(as_completed(futures)):
            for row in fu.result():rows.append(row);f.write(json.dumps(row,separators=(',',':'))+'\n')
            f.flush();print(json.dumps(dict(done=i+1,total=len(jobs),job=futures[fu],seconds=round(time.monotonic()-started,1))),flush=True)
    summary=[]
    for job in jobs:
        by={r['arm']:r for r in rows if (r['game'],r['profile'],r['schedule'])==(job[0],PROFILES[job[1]]['id'],job[3])}
        summary.append(dict(game=job[0],profile=PROFILES[job[1]]['id'],schedule=job[3],
            scores={k:v['score'] for k,v in by.items()},brier={k:v['response_brier'] for k,v in by.items()},
            choice_vs_joint=by['persona_choice']['score']-by['persona_joint']['score']))
    (root/'summary.json').write_text(json.dumps(dict(runs=len(rows),decisions=sum(r['turns'] for r in rows),
        modeled_decisions=sum(r['modeled_decisions'] for r in rows),pairs=summary,
        checks={k:sum(r['checks'].get(k,0) for r in rows) for k in rows[0]['checks']}),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
