"""Compose existing support banks with the public availability of responses.

Three arms isolate global history, the former wrapped validation wiring, and
the corrected wrapper. No winning move or personality coefficient is changed.
Availability is the set allowed by the declared public rules, never a hidden
opponent type. All effects are delivered every fourth turn, all events arrive.
"""
import argparse,json,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import digest
from reflex.loop_predictor import GuardedSupportedCategoricalReader
from tools.run_lifecycle_soak import Adapter,public_context,run_arm,GAMES


class AvailabilityReader(GuardedSupportedCategoricalReader):
    @property
    def tracker(self):return self.shared.tracker


class LegacyAvailabilityReader(AvailabilityReader):
    def response_comparison(self,c):return None


class AvailabilityAdapter(Adapter):
    reader_type=AvailabilityReader
    def actor(self,c):
        actor=super().actor(c)
        classes=tuple(digest(self.symbols()[:i]) for i in range(1,len(self.symbols())+1))
        actor.predictor=self.reader_type(c['scope'],self.symbols(),self.models,self.outcome,known_conditionals=True,
            classes=classes,opportunity_for=lambda cc:digest(self.rules.symbols(public_context(cc)['world'])),
            support_key='public-response-availability-v1')
        return actor


class LegacyAvailabilityAdapter(AvailabilityAdapter):reader_type=LegacyAvailabilityReader


def evaluate(job):
    result=[]
    for name,factory in (('global',Adapter),('legacy-support',LegacyAvailabilityAdapter),('support',AvailabilityAdapter)):
        row=run_arm(job,'channels',adapter_factory=factory);row['arm']=name;result.append(row)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--frozen',action='store_true')
    p.add_argument('--turns',type=int,default=128);p.add_argument('--workers',type=int,default=3);p.add_argument('--pilot',action='store_true')
    a=p.parse_args();root=Path(a.root)
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        args=['--turns',str(a.turns),'--workers',str(a.workers)]
        if a.pilot:args.append('--pilot')
        dispatch(root,Path(__file__).name,args);return
    root.mkdir(parents=True,exist_ok=True)
    jobs=[(g,p,17,mode,a.turns,'censored',12) for g in GAMES for p in range(4) for mode in ('steady','shift-return')]
    if a.pilot:jobs=[(g,0,17,'shift-return',a.turns,'censored',12) for g in GAMES]
    (root/'registration.json').write_text(json.dumps(dict(jobs=jobs,semantics=__doc__),indent=2)+'\n',encoding='utf-8')
    started=time.monotonic();rows=[]
    with (root/'trajectories.jsonl').open('w',encoding='utf-8') as f,ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(evaluate,j):j for j in jobs}
        for i,fu in enumerate(as_completed(futures)):
            for row in fu.result():rows.append(row);f.write(json.dumps(row,separators=(',',':'))+'\n')
            f.flush();print(json.dumps(dict(done=i+1,total=len(jobs),job=futures[fu],seconds=round(time.monotonic()-started,1))),flush=True)
    summary=[]
    for job in jobs:
        from reflex.laboratory import PROFILES
        by={r['arm']:r for r in rows if (r['game'],r['profile'],r['schedule'])==(job[0],PROFILES[job[1]]['id'],job[3])}
        summary.append(dict(game=job[0],profile=PROFILES[job[1]]['id'],schedule=job[3],
            scores={k:v['score'] for k,v in by.items()},brier={k:v['response_brier'] for k,v in by.items()},
            versus_global=by['support']['score']-by['global']['score'],
            versus_legacy_support=by['support']['score']-by['legacy-support']['score']))
    (root/'summary.json').write_text(json.dumps(dict(runs=len(rows),decisions=sum(r['turns'] for r in rows),
        modeled_decisions=sum(r['modeled_decisions'] for r in rows),pairs=summary,
        checks={k:sum(r['checks'].get(k,0) for r in rows) for k in rows[0]['checks']}),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
