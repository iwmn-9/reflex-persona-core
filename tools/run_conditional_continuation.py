"""Frozen cross-adapter comparisons of conditional owner continuation."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.run_team_cooperation import run_game as team_game,SCENARIOS
from reflex.conditional_forecast import forecast

MODES=('bargain_uncertain','bargain_personal','bargain_exchange','bargain_robust')


def solo_opening(genre,scenario,team,w):
    if genre=='projects':
        people=tuple(next(p for p in w.people if p.team==t) for t in (0,1))
        if scenario=='exhausted':people=tuple(replace(p,energy=2) for p in people)
        return replace(w,people=people,quota=8)
    people=tuple(w.units[1 if t==0 else 4] for t in (0,1))
    return replace(w,units=people)


def run_game(job):
    genre,scenario,roster,seed,arena,mode=job
    participation='partial' if arena=='partial' else 'all'
    continuation=mode.removeprefix('bargain_')
    def candidate(g,w,a,c,d):
        if continuation!='robust':return forecast(g,w,a,c,d,continuation=continuation)
        from reflex import team_combat as tc,team_projects as tp
        from reflex.forecast_ensemble import combine
        baseline=(tc if g=='combat' else tp).forecast(w,a,c,d,partner_model='uncertain')
        return combine((baseline,forecast(g,w,a,c,d)),purpose_rule='worst')
    callback=None if mode=='bargain_uncertain' else candidate
    row=team_game((genre,scenario,roster,seed,participation,mode),
        opening=solo_opening if arena=='solo' else None,forecast_fn=callback)
    row['arena']=arena
    return row


def worker(job):
    identity,modes=job
    return [run_game((*identity,mode)) for mode in modes]


def run(a):
    root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
    rosters=(0,1) if a.stage=='diagnostic' else tuple(range(4))
    jobs=[(g,s,r,seed,arena) for g in a.genres for s in (SCENARIOS[g][:2] if a.stage=='diagnostic' else SCENARIOS[g])
        for r in rosters for seed in range(a.start,a.start+a.seeds) for arena in a.arenas]
    protocol=dict(version='conditional-continuation-v1',stage=a.stage,jobs=jobs,modes=a.modes,
        primary='paired full-game terminal credit by adapter and solo/all/partial arena; seed-cluster intervals',
        settings=dict(outer_horizon=6,combat_samples=4,combat_root_plans=24,projects_root_plans=64,
            inner_horizon=1,inner_plans=8,unsigned_models='same hold/independent uniform hypotheses in all arms'),
        intervention='old capability continuation vs detached fixed owners refreshing needs/routes, optional one-turn future exchange, or worst-purpose ensemble',
        constraints=['same current candidate bank, purpose/concession/personality parameters','no real future intents/private unsigned memories',
            'solo reuses exactly the world rules and owner contract, with one person on each side','no coefficient tuning after results'],
        limits=['authored public worlds, not commercial games or humans','future one-turn exchange differs from actual six-turn exchange',
            'solo team objective becomes personal objective; not every individual game genre','roots updated from baseline owner mode, not outer forecast-derived future need mode',
            'inner controller computation is extra; this is an architecture comparison, not equal compute'])
    if (root/'preregister.json').exists():raise FileExistsError('fresh study directory required')
    (root/'preregister.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    with ProcessPoolExecutor(max_workers=a.workers) as pool,(root/'trajectories.jsonl').open('w',encoding='utf-8') as stream:
        tasks=[pool.submit(worker,(j,a.modes)) for j in jobs]
        for i,task in enumerate(as_completed(tasks),1):
            rows=task.result()
            for row in rows:stream.write(json.dumps(row,separators=(',',':'))+'\n')
            stream.flush();last=rows[-1]
            print(f'{i}/{len(jobs)} {last["genre"]}/{last["scenario"]}/{last["arena"]} roster{last["roster"]} seed{last["seed"]}: {[r["win_credit"] for r in rows]} seconds {[round(r["game_seconds"],1) for r in rows]}',flush=True)
    (root/'completed.json').write_text(json.dumps(dict(games=len(jobs)*len(a.modes),complete=True))+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--start',type=int,required=True)
    p.add_argument('--seeds',type=int,default=1);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--stage',choices=('diagnostic','confirmation'),default='diagnostic')
    p.add_argument('--genres',nargs='+',choices=tuple(SCENARIOS),default=list(SCENARIOS))
    p.add_argument('--arenas',nargs='+',choices=('solo','all','partial'),default=['solo','all','partial'])
    p.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES));p.add_argument('--frozen',action='store_true')
    a=p.parse_args()
    if a.frozen:run(a)
    else:
        from tools.freeze_experiment import dispatch
        dispatch(a.root,Path(__file__).name,['--start',str(a.start),'--seeds',str(a.seeds),'--workers',str(a.workers),
            '--stage',a.stage,'--genres',*a.genres,'--arenas',*a.arenas,'--modes',*a.modes])
