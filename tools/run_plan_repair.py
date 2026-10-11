"""Compare fresh search, retained plans and local repairs under the same purpose."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.observed_transfer import run as episode
from reflex.laboratory import profiles

MODES=('fresh','retained','repair')
SPECS={
 'diagnostic':[
  dict(genre='delivery',limit=10,capacity=3,supply=2),dict(genre='resources',route='mixed',limit=14),
  dict(genre='auction',prizes=[9,14,6],budget=11,rival_shift=True),dict(genre='combat',hp=7,limit=7,distance=4)],
 'confirmation':[
  dict(genre='delivery',limit=12,capacity=2,supply=3),dict(genre='resources',route='mixed',limit=16),
  dict(genre='auction',prizes=[8,15,7,11],budget=12,rival_shift=True),dict(genre='combat',hp=5,limit=12,distance=4),
  dict(genre='delivery',limit=9,capacity=3,supply=1),dict(genre='resources',route='science',opening='conversion',limit=7),
  dict(genre='auction',prizes=[6,16,10,4],budget=7,rival_shift=True),dict(genre='combat',hp=4,limit=9,distance=4)]}


def run_game(job):
    stage,spec,index,seed,mode=job
    r=episode(spec,profiles()[index],'verified',seed,6,search=True,continuity=mode!='fresh',
        repairs=24 if mode=='repair' else 0,principle_priority='finite',settlement_weight='absolute',
        validation_seeds=(920,921,922,923) if spec['genre']=='combat' else None)
    r.update(stage=stage,profile_index=index,mode=mode)
    return r


def worker(packet):return [run_game(j) for j in packet]


def run(a):
    root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
    indices=range(2) if a.stage=='diagnostic' else range(4)
    jobs=[(a.stage,s,p,seed,m) for s in SPECS[a.stage] for p in indices
        for seed in range(a.start,a.start+a.seeds) for m in MODES]
    pre=dict(format='plan-repair-v1',jobs=jobs,
        settings=dict(horizon=6,width=2,depth=2,repairs=24,principle_priority='finite',settlement_weight='absolute',
            actual_feedback='same progress and need pressure in every arm',combat_discovery_seeds=[820,821,822,823],
            combat_validation_seeds=[920,921,922,923]),
        primary='paired actual game-owned terminal goal, survival, shortages, reachability; no standard adoption on regressions',
        secondary='selected one-step patches vs fresh/retained proposals, actual head deviations, goal identity/checkpoint retention, shared-branch schedule fallbacks',
        intervention='same fresh candidates plus the owned remaining plan and bounded single-step patches; tie preference only for equal purpose AND subjective scores',
        constraints=['stable owner/preferences/target/time unit','only actual executed root consumed','same schedule in every model branch',
            'no real future seed or hypothetical empirical learning','fresh/full replacement remains available'],
        limits=['known authored games, not unseen rule transfer','single-owner schedules; team commitments not connected in this study',
            'publicly unavailable future schedule uses existing reflex fallback','model replanning is not recursive outer search',
            'deterministic genres repeat across seeds, not independent samples','repairs add compute and candidates'])
    if (root/'preregister.json').exists():raise FileExistsError('fresh study directory required')
    (root/'preregister.json').write_text(json.dumps(pre,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    packets=[jobs[i:i+len(MODES)] for i in range(0,len(jobs),len(MODES))]
    with ProcessPoolExecutor(max_workers=a.workers) as pool,(root/'trajectories.jsonl').open('w',encoding='utf-8') as stream:
        futures=[pool.submit(worker,p) for p in packets]
        for count,f in enumerate(as_completed(futures),1):
            rows=f.result()
            for row in rows:stream.write(json.dumps(row,separators=(',',':'))+'\n')
            stream.flush();r=rows[-1]
            print(f'{count}/{len(packets)} {r["spec"]["genre"]}/{r["profile"]}/{r["seed"]}: '+
                str([(r['mode'],r['status'],r['goal_value']) for r in rows]),flush=True)
    (root/'completed.json').write_text(json.dumps(dict(games=len(jobs),complete=True))+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--stage',choices=tuple(SPECS),required=True)
    p.add_argument('--start',type=int,required=True);p.add_argument('--seeds',type=int,default=1);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--frozen',action='store_true');a=p.parse_args()
    if not 1<=a.seeds<=32 or not 1<=a.workers<=8:p.error('bounded seeds and workers required')
    if a.frozen:run(a)
    else:
        from tools.freeze_experiment import dispatch
        dispatch(a.root,Path(__file__).name,['--stage',a.stage,'--start',str(a.start),'--seeds',str(a.seeds),'--workers',str(a.workers)])
