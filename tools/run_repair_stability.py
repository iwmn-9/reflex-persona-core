"""Registered cross-genre comparisons; no action/profile-specific repair rule."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.observed_transfer import run as episode
from reflex.laboratory import profiles

MODES=('retained','repair','guard')
SPECS={
 'diagnostic':[
  dict(genre='delivery',limit=12,capacity=2,supply=3),dict(genre='resources',route='mixed',limit=16),
  dict(genre='auction',prizes=[8,15,7,11],budget=12,rival_shift=True),dict(genre='combat',hp=5,limit=12,distance=4)],
 'confirmation':[
  dict(genre='delivery',limit=11,capacity=3,supply=2),dict(genre='resources',route='culture',opening='conversion',limit=8),
  dict(genre='auction',prizes=[7,13,5,17],budget=10,rival_shift=True),dict(genre='combat',hp=6,limit=11,distance=4)]}


def run_game(job):
    stage,spec,index,seed,mode=job
    r=episode(spec,profiles()[index],'verified',seed,6,search=True,continuity=True,
        repairs=0 if mode in ('retained','complete-retained') else 24,repair_guard=mode in ('guard','feedback','complete-feedback'),replan_check=mode in ('feedback','complete-feedback'),
        complete_to=16 if mode.startswith('complete-') else 0,principle_priority='finite',
        settlement_weight='absolute',validation_seeds=tuple(range(940,948)) if spec['genre']=='combat' else None)
    r.update(stage=stage,profile_index=index,mode=mode);return r


def worker(packet):return [run_game(j) for j in packet]


def run(a):
    root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
    modes=MODES if a.angle=='paired' else ('retained','repair','feedback') if a.angle=='feedback' else ('retained','complete-retained','complete-feedback')
    jobs=[(a.stage,s,p,seed,m) for s in SPECS[a.stage] for p in range(4)
        for seed in range(a.start,a.start+a.seeds) for m in modes]
    pre=dict(format='repair-stability-v1',angle=a.angle,modes=modes,jobs=jobs,settings=dict(horizon=6,width=2,depth=2,
        repairs=24,principle_priority='finite',settlement_weight='absolute',validation_seeds=list(range(940,948))),
        hypothesis='branchwise endpoint dominance over independently selected retained bank reduces repair regressions' if a.angle=='paired' else
            'one future replanning event with selected owner state and branch-local progress/needs reduces optimistic repair errors' if a.angle=='feedback' else
            'extend all modeled endpoints to settlement or cap16 instead of relying on cutoff proxies; compare completion alone and completion plus repair protection',
        primary='paired actual terminal game-owned goal, survival, shortages and reachable-route loss',
        secondary='guard acceptance/rejection/unprotected tier changes, actual actions, selected repairs',
        adoption='no default adoption if any actual regression; no adoption for unchanged actions or wholesale fallback alone',
        constraints=['same target/personality/actual feedback','same candidate bank; all viable current roots kept',
            'same disjoint validation bank in every arm','no actual future seed in forecasts','actual progress masks in both guard votes'],
        limits=['branch dominance is not real-world safety or calibrated confidence','only eight stochastic model branches',
            'known authored games; confirmation changes conditions, not unknown-rule transfer',
            'deterministic seed repeats are not independent','value-tier changes remain explicitly unprotected',
            'future outer replanning remains approximated by schedules/reflex'])
    if a.angle in ('feedback','completion'):
        pre['settings']['feedback_check_seeds']=[960,961,962,963]
        pre['limits'].append('future controller omits this check; one next replan then shared tail/reflex, not exact recursive current controller')
    if a.angle=='completion':pre['settings']['completion_cap']=16
    if (root/'preregister.json').exists():raise FileExistsError('fresh study required')
    (root/'preregister.json').write_text(json.dumps(pre,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    packets=[jobs[i:i+len(modes)] for i in range(0,len(jobs),len(modes))]
    with ProcessPoolExecutor(max_workers=a.workers) as pool,(root/'trajectories.jsonl').open('w',encoding='utf-8') as stream:
        for count,f in enumerate(as_completed([pool.submit(worker,p) for p in packets]),1):
            rows=f.result()
            for r in rows:stream.write(json.dumps(r,separators=(',',':'))+'\n')
            stream.flush();print(f'{count}/{len(packets)} '+str([(r['spec']['genre'],r['profile'],r['mode'],r['status'],r['goal_value']) for r in rows]),flush=True)
    (root/'completed.json').write_text(json.dumps(dict(games=len(jobs),complete=True))+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--stage',choices=tuple(SPECS),required=True)
    p.add_argument('--start',type=int,required=True);p.add_argument('--seeds',type=int,default=1);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--angle',choices=('paired','feedback','completion'),default='paired')
    p.add_argument('--frozen',action='store_true');a=p.parse_args()
    if not 1<=a.seeds<=32 or not 1<=a.workers<=8:p.error('bounded seeds/workers required')
    if a.frozen:run(a)
    else:
        from tools.freeze_experiment import dispatch
        dispatch(a.root,Path(__file__).name,['--stage',a.stage,'--start',str(a.start),'--seeds',str(a.seeds),'--workers',str(a.workers),'--angle',a.angle])
