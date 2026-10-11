"""Frozen paired heterogeneous team games; strength and consent kept separate."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import Policy,digest,TRAITS,VALUES
from reflex.laboratory import PROFILES
from reflex.team_choice import select

ROSTERS=((0,1,2),(3,0,1),(2,3,0),(1,2,3))
SCENARIOS={'combat':('open','choke','rescue'),'projects':('balanced','exhausted','shock')}
MODES=('independent','sum','consent','bargain','sum_uncertain','bargain_uncertain','sum_observed','bargain_observed')


def run_game(job,members=3):
    genre,scenario,roster,seed,participation,mode=job
    from reflex import team_projects as tp,team_combat as tc
    from reflex.combat import alive,resolve,terminal,victory,battle_record,opponent
    team=seed%2;w=tc.opening(scenario,team) if genre=='combat' else tp.Workshop.start(scenario,members=members)
    initial=asdict(w);real_seed=int(digest(['team-cooperation-world',seed])[:16],16)
    people=alive(w,team) if genre=='combat' else tp.members(w,team)
    assigned={i:copy.deepcopy(PROFILES[ROSTERS[roster][j%3] if len(people)==3 else (j+roster)%4]) for j,i in enumerate(people)}
    before_profiles=copy.deepcopy(assigned);memories={};trace=[];counts=Counter();elapsed=0.
    book=None
    if mode.endswith('_observed') and participation=='partial':
        from reflex.team_observation import PartnerMemory
        book=PartnerMemory(genre,f'team-{seed}',f'team-{team}',(people[-1],))
    start=time.perf_counter()
    while not (terminal(w) if genre=='combat' else tp.terminal(w)):
        before=w;actors=alive(w,team) if genre=='combat' else tp.members(w,team)
        # All members exist in the team, but only the explicitly signed group
        # participates in collective choice. The unsigned last ally's current
        # choice is computed after negotiation and never enters the forecast.
        signed=actors if participation=='all' else tuple(i for i in actors if i!=people[-1])
        learning=None;partner_weights=None
        if book is not None:
            outsiders=tuple(i for i in actors if i not in signed);models={};rules=tc if genre=='combat' else tp
            from reflex.combat import legal as combat_legal
            for i in outsiders:
                names=combat_legal(w,i) if genre=='combat' else tp.legal(w,i)
                models[i]={label:{k:float(k==rules.outside_action(w,i,kind)) for k in names}
                    for kind,label in enumerate(book.names)}
            learning=dict(before=book.receipt(),models=models)
            weights=book.begin(book.scope,w.tick,models)
            if outsiders:partner_weights=weights[outsiders[0]]
        contexts=[(tc.make_person(w,i,assigned[i],seed,memories.get(i)) if genre=='combat'
            else tp.make_context(w,i,assigned[i],seed,memories.get(i))) for i in signed]
        ds=[Policy(principle_priority='finite').choose(c,False) for c in contexts]
        for i,c in zip(signed,contexts):
            assert c['personality']==dict(zip(TRAITS,assigned[i]['traits']))
            assert c['values']=={k:float(assigned[i]['values'].get(k,0)) for k in VALUES}
        audit=None;f=None;chosen=ds
        if signed and mode!='independent':
            model='uncertain' if mode.endswith(('_uncertain','_observed')) else 'cooperative'
            extra={} if partner_weights is None else {'partner_weights':partner_weights}
            t=time.perf_counter();f=(tc.forecast(w,signed,contexts,ds,partner_model=model,**extra) if genre=='combat' else tp.forecast(w,signed,contexts,ds,partner_model=model,**extra))
            selection='sum' if mode.startswith('sum') else 'consent'
            outside=tuple(d['action_id'] for d in ds) if mode.startswith('bargain') else None
            selected,audit=select(contexts,f,group=f'{genre}-team-{team}',mode=selection,outside=outside)
            elapsed+=time.perf_counter()-t;counts['model_nodes']+=f.audit['nodes']
            counts['adopted']+=selected is not None;counts['declined']+=selected is None
            if selected is not None:chosen=selected
        choices={i:d['action_id'] for i,d in zip(signed,chosen)}
        for i,c,d in zip(signed,contexts,chosen):
            assert d['context_hash']==digest(c)
            memories[i]=copy.deepcopy(d['next_state'])
            counts['changed_roots']+=d['action_id']!=ds[signed.index(i)]['action_id']
        # No actual unsigned ally or rival intention exists until signed offers
        # are fixed. They use independent controllers without reciprocal access.
        for i in actors:
            if i in signed:continue
            c=(tc.make_person(w,i,assigned[i],seed,memories.get(i)) if genre=='combat' else tp.make_context(w,i,assigned[i],seed,memories.get(i)))
            d=Policy(principle_priority='finite').choose(c,False);choices[i]=d['action_id'];memories[i]=d['next_state']
        if genre=='combat':
            rival='reference' if seed%3==0 else 'switch'
            for i in alive(w,1-team):choices[i]=opponent(w,i,seed,rival)
            w,world_audit=resolve(w,choices,real_seed)
            counts['movement_conflicts']+=world_audit['collisions']
            for i in actors:
                counts['failed_moves']+=choices[i].startswith('move:') and w.units[i].pos==before.units[i].pos
                counts['ally_heals']+=choices[i].startswith('heal:') and int(choices[i].split(':')[1])!=i
                counts['self_heals']+=choices[i]==f'heal:{i}'
            goal=team in victory(w);loss=1-team in victory(w)
            value=.5 if not goal and not loss else 1/(1+int(loss)) if goal else 0.
            health=sum(w.units[i].hp for i in people)
        else:
            choices.update(tp.greedy(w,1-team));w,world_audit=tp.resolve(w,choices)
            counts['material_conflicts']+=sum(i in people for i in world_audit['material_conflicts'])
            counts['ally_aids']+=sum(k.startswith('aid:') for i,k in choices.items() if i in people)
            counts['overflow']+=world_audit['overflow'];value=tp.credit(w,team);health=sum(w.people[i].energy for i in people)
            goal=team in tp.winners(w);loss=1-team in tp.winners(w)
        if book is not None:
            learning['updates']=book.observe(book.scope,before.tick,{i:choices[i] for i in models},witnessed=True)
            learning['after']=book.receipt()
        selected_plan=None if audit is None or not audit['adopted'] else audit['selected_plan']
        trace.append(dict(before=asdict(before),after=asdict(w),choices=choices,world_audit=world_audit,
            signed=list(signed),contexts=contexts,decisions=chosen,independent=ds,
            negotiation=audit,partner_learning=learning,
            selected_roots=None if selected_plan is None else list(f.roots[selected_plan]),
            selected_effects=None if selected_plan is None else [next(a for a in c['actions'] if a['id']==selected_plan)['outcomes'] for c in f.contexts]))
    assert assigned==before_profiles
    return dict(genre=genre,scenario=scenario,roster=roster,profiles={str(i):p['id'] for i,p in assigned.items()},
        seed=seed,team=team,participation=participation,mode=mode,members=len(people),initial=initial,final=asdict(w),
        win_credit=value,won=goal,lost=loss,health=health,ticks=w.tick,
        work=dict(counts),planner_seconds=elapsed,game_seconds=time.perf_counter()-start,trace=trace)


def worker(job):
    args,modes,members=job
    return [run_game((*args,mode),members=members) for mode in modes]


def run(args):
    root=Path(args.root);root.mkdir(parents=True,exist_ok=True)
    rosters=(0,1) if args.stage=='diagnostic' else tuple(range(4))
    scenarios={g:SCENARIOS[g][:2] if args.stage=='diagnostic' else SCENARIOS[g] for g in SCENARIOS}
    jobs=[(g,s,r,seed,p) for g in args.genres for s in scenarios[g] for r in rosters
          for seed in range(args.start,args.start+args.seeds) for p in args.participation]
    plan=dict(version='team-cooperation-v1',stage=args.stage,start=args.start,seeds=args.seeds,
        jobs=jobs,modes=args.modes,rosters=ROSTERS,workers=args.workers,project_members=args.members,
        primary='paired actual terminal credit, separately by genre and participation; seed-cluster intervals',
        secondary=['fixed mixed personalities, survival/energy, actual aid, resource/movement conflicts, per-member modeled concession'],
        settings=dict(combat_horizon=6,combat_samples=4,combat_plans=24,projects_horizon=6,projects_plans=64),
        selection='No personality coefficient or concession tuning on this matrix; failures retained. Confirmation uses fresh seeds.',
        limits=['authored public rule engines','explicit signed numeric bid exchange, not language negotiation',
                'future teammate tactics differ from actual re-negotiation','no trained team policy or performance target',
                'same seeds couple hit streams across arms but actual state divergence changes hit probabilities'])
    path=root/'preregister.json'
    if path.exists():raise FileExistsError('fresh study directory required')
    path.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    with ProcessPoolExecutor(max_workers=args.workers) as pool,(root/'trajectories.jsonl').open('w',encoding='utf-8') as stream:
        pending=[pool.submit(worker,(j,args.modes,args.members)) for j in jobs]
        for index,future in enumerate(as_completed(pending),1):
            rows=future.result()
            for row in rows:stream.write(json.dumps(row,separators=(',',':'))+'\n')
            stream.flush();last=rows[-1]
            print(f'{index}/{len(jobs)} {last["genre"]} {last["scenario"]} roster{last["roster"]} seed{last["seed"]} {last["participation"]}: credits {[r["win_credit"] for r in rows]} seconds {[round(r["game_seconds"],1) for r in rows]}',flush=True)
    (root/'completed.json').write_text(json.dumps(dict(games=len(jobs)*len(args.modes),complete=True))+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--start',required=True,type=int)
    p.add_argument('--seeds',type=int,default=2);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--stage',choices=('diagnostic','confirmation'),default='diagnostic')
    p.add_argument('--genres',nargs='+',choices=tuple(SCENARIOS),default=list(SCENARIOS))
    p.add_argument('--participation',nargs='+',choices=('all','partial'),default=['all','partial'])
    p.add_argument('--modes',nargs='+',choices=MODES,default=['independent','sum','consent'])
    p.add_argument('--members',type=int,choices=(2,3,4),default=3)
    p.add_argument('--frozen',action='store_true');a=p.parse_args()
    if a.frozen:run(a)
    else:
        from tools.freeze_experiment import dispatch
        dispatch(a.root,Path(__file__).name,['--start',str(a.start),'--seeds',str(a.seeds),
            '--workers',str(a.workers),'--stage',a.stage,'--members',str(a.members),'--genres',*a.genres,'--participation',*a.participation,'--modes',*a.modes])
