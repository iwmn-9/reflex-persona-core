"""Finite native multi-NPC comparison and replay export. No cloud or scheduler."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
SCENARIOS=(('combat','cover','either'),('combat','open','both'),('combat','choke','eliminate'),
           ('hagetaka',None,None),('exchange',None,None))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,default=4)
    p.add_argument('--seed-start',type=int,default=2026110301)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--horizon',type=int,default=6)
    p.add_argument('--second-moves',type=int,default=8,choices=range(9))
    p.add_argument('--max-regret',type=float,default=.15)
    p.add_argument('--scenario-batches',type=int,default=1,choices=range(1,5))
    p.add_argument('--modes',nargs='+',choices=('reflex','planned','challenger'),default=('reflex','planned'))
    p.add_argument('--beliefs',choices=('joint','individual'),default='individual')
    p.add_argument('--coordination',choices=('none','consent'),default='consent')
    p.add_argument('--intent-sharing',choices=('none','proposals'),default='none')
    p.add_argument('--reference',choices=('frozen','matched'),default='frozen')
    p.add_argument('--planning-protocol',choices=('counterproposal','independent'),default='counterproposal')
    p.add_argument('--routes',choices=('none','persona'),default='none')
    p.add_argument('--forecast-guard',choices=('none','batch','alternatives'),default='none')
    p.add_argument('--relationships',choices=('off','memory'),default='off')
    p.add_argument('--games',nargs='+',choices=('combat','hagetaka','exchange'),default=('combat','hagetaka','exchange'))
    p.add_argument('--combat-maps',nargs='+',choices=('cover','open','choke'),default=('cover','open','choke'))
    a=p.parse_args()
    if not 1<=a.seeds<=16 or not 1<=a.workers<=4 or not 1<=a.horizon<=32: p.error('bounded run required')
    if not math.isfinite(a.max_regret) or not 0<=a.max_regret<=1:p.error('max-regret must be in 0..1')
    subprocess.run(['cargo','build','--release','--manifest-path',str(ROOT/'rust/Cargo.toml'),'-p','reflex-arena'],check=True)
    a.output.mkdir(parents=True,exist_ok=False)
    original=ROOT/'rust/target/release/reflex-arena.exe'
    if not original.exists(): original=original.with_suffix('')
    binary=a.output/original.name;shutil.copy2(original,binary)
    jobs=[]
    for game,board,goal in SCENARIOS:
        if game not in a.games:continue
        if game=='combat' and board not in a.combat_maps:continue
        for index in range(a.seeds):
            for mode in a.modes:
                seed=a.seed_start+index
                name=f'{game}-{board or "standard"}-{goal or "score"}-{seed}-{mode}'
                jobs.append(dict(id=name,game=game,map=board,goal=goal,seed=seed,mode=mode))
    manifest=dict(jobs=jobs,horizon=a.horizon,second_moves=a.second_moves,max_regret=a.max_regret,scenario_batches=a.scenario_batches,turns=32,beliefs=a.beliefs,coordination=a.coordination,intent_sharing=a.intent_sharing,reference=a.reference,planning_protocol=a.planning_protocol,routes=a.routes,forecast_guard=a.forecast_guard,relationships=a.relationships,opponent_models="adapter",binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        sources={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted((ROOT/'rust').glob('reflex-*/src/*.rs'))},
        purpose='Native mixed-personality actual games and behavior review; not automatic adoption',
        old_probability_trials='stopped; not a prerequisite',tiny_equivalence=0.005,severe_regression=-0.02)
    (a.output/'plan.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    def run(job):
        command=[str(binary.resolve()),'--game',job['game'],'--seed',str(job['seed']),
            '--mode',job['mode'],'--horizon',str(a.horizon),'--second-moves',str(a.second_moves),'--max-regret',str(a.max_regret),'--scenario-batches',str(a.scenario_batches),'--turns','32','--beliefs',a.beliefs,'--coordination',a.coordination,'--intent-sharing',a.intent_sharing,'--reference',a.reference,'--planning-protocol',a.planning_protocol,'--routes',a.routes,'--forecast-guard',a.forecast_guard,'--relationships',a.relationships,'--out',str((a.output/(job['id']+'.json')).resolve())]
        if job['map']:command+=['--map',job['map'],'--goal',job['goal']]
        result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf8',timeout=600)
        if result.returncode:raise RuntimeError(f"{job['id']}: {result.stderr[-2000:]}")
        report=json.loads((a.output/(job['id']+'.json')).read_bytes())
        if not report['complete']:raise RuntimeError(f"{job['id']}: incomplete")
        return dict(job=job,**{k:report[k] for k in ('ticks','factual_decisions','purpose','plan_events','seconds')})
    completed=[];failures=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(run,j):j for j in jobs}
        for future in as_completed(futures):
            try:completed.append(future.result())
            except Exception as error:failures.append(dict(job=futures[future],error=str(error)))
            (a.output/'progress.json').write_text(json.dumps(dict(completed=len(completed),expected=len(jobs),failures=failures)),encoding='utf8')
            print(json.dumps(dict(completed=len(completed),expected=len(jobs),failures=len(failures))),flush=True)
    comparisons=[]
    by_id={row['job']['id']:row for row in completed}
    for row in completed:
        job=row['job']
        if job['mode']!='challenger':continue
        baseline=by_id.get(job['id'].removesuffix('challenger')+'reflex')
        if baseline is None:continue
        report=json.loads((a.output/(job['id']+'.json')).read_bytes())
        focal=report['focal_players']
        delta=sum(row['purpose'][i]-baseline['purpose'][i] for i in focal)/len(focal)
        comparisons.append(dict(id=job['id'],game=job['game'],map=job['map'],goal=job['goal'],seed=job['seed'],
            focal_players=focal,delta=delta,classification='equivalent' if abs(delta)<=.005 else 'better' if delta>0 else 'worse',
            severe_regression=delta<=-.02))
    result=dict(complete=not failures and len(completed)==len(jobs),rows=completed,failures=failures,
        focal_comparisons=comparisons,adopted=False)
    (a.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
