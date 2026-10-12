"""Freeze public-schema and actual strategic-instruction diagnostics."""
import argparse,copy,hashlib,json,shutil,subprocess,sys
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import Policy,character_key,digest
from reflex.laboratory import profiles
from reflex.execution_window import Directive,ExecutionWindow
from reflex.observation_plan import ObservationSchema,ObservationPlan,View
from reflex.contingent_games import make_bridge
from tools.run_contingent_coverage import cases

def hashes():
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('reflex','tools','tests') for p in sorted((ROOT/folder).glob('*.py'))}

def contexts():
    for case in cases('confirmation'):
        for p in profiles():
            if case['game']=='mahjong_closed':
                from reflex.contingent_mahjong import engine,candidates,HandState,HandModel
                env=engine().RiichiEnv(game_mode='4p-red-single',seed=1700);obs=env.reset();seat=next(iter(obs))
                v,cs=candidates(obs[seat]);c=HandModel(p,'conditional-mahjong').context(HandState(v,tuple(cs),0))
            elif case['game'].startswith('observed_'):
                from reflex.observed_transfer import make_probe
                b=make_probe(case['spec'],p);c=b.observe(b.start())
            else:
                b=make_bridge(p,case);c=b.observe(b.project(b.start()))
            yield case,p,c

def schemas():
    rows=[]
    for case,p,c in contexts():
        schema=ObservationSchema('adapter-facts-v1',tuple(sorted(c['facts'])))
        key=Policy().choose(c,False)['action_id'];v=View.make('',c,schema=schema)
        plan=ObservationPlan.make(c,'purpose','own-decisions',{v.node:key},schema=schema)
        changed=copy.deepcopy(c);changed['facts']['diagnostic_annotation']='irrelevant display label'
        assert plan.choose(View.make('',changed,schema=schema))==key
        changed['tick']+=1;assert plan.choose(View.make('',changed,schema=schema)) is None
        assert ObservationPlan.restore(plan.record())==plan
        rows.append(dict(game=case['game'],profile=p['id'],retained_facts=len(schema.facts),
            annotation_hit=True,clock_change_miss=True,root=key))
    return rows

def routes(profile,seed,route):
    from reflex.route_experiment import routed_context
    from reflex.resource_world import World,ROUTES,achieved,step,winners,terminal,economic_move,world_record,world_from_record
    w=World.start();seat=seed%4;memory=None;trace=[];tick=0;global_tick=0;window=None
    while not terminal(w):
        if w.turn==seat:
            c,_=routed_context(w,profile,42,tick,memory,route)
            if window is None:
                scope=copy.deepcopy(c['scope']);window=ExecutionWindow(scope,tuple(Directive(r,'actual-victory-'+r,12*len(w.empires),'global-turns') for r in ROUTES))
            window.issue(c['scope'],route,global_tick,opportunity=any(a['legal'] and not a['known_failure'] for a in c['actions']))
            p=Policy(principle_priority='personal',commitments={character_key(c):.5})
            d=p.choose(c,False);memory=d['next_state'];key=d['action_id']
        else:c=None;key=economic_move(w)
        before=w;w=step(w,key)
        audit=None
        if window is not None and window.active is not None:
            # Independent physical threshold criterion; not the objective proxy.
            met=route in achieved(w.empires[seat])
            opportunity=c is not None or w.empires[seat]!=before.empires[seat] or met or terminal(w)
            audit=window.observe(scope,global_tick,opportunity=opportunity,met=met,failed=terminal(w) and not met)
        if c is not None:tick+=1
        trace.append(dict(before=world_record(before),after=world_record(w),root=key,own=c is not None,execution=audit,global_tick=global_tick));global_tick+=1
    for t in trace:assert world_record(step(world_from_record(t['before']),t['root']))==t['after']
    return dict(profile=profile['id'],seed=seed,seat=seat,route=route,won=seat in winners(w),
        achieved=list(achieved(w.empires[seat])),actual_route_satisfied=route in achieved(w.empires[seat]),
        execution=window.record(),actions=[t['root'] for t in trace if t['own']],trace=trace,action_replay_verified=True)

def interventions():
    from reflex.resource_world import World,Empire,ROUTES,act,world_record
    from reflex.route_experiment import routed_context
    base=World.start();scenarios={'initial':base,
        'science_capacity':replace(base,empires=(replace(Empire(),tech=2,science=12,buildings=(1,1,1,0,2,0)),)+base.empires[1:]),
        'culture_capacity':replace(base,empires=(replace(Empire(),monuments=1,culture=12,buildings=(1,1,1,0,0,2)),)+base.empires[1:]),
        'military_capacity':replace(base,empires=(replace(Empire(),land=3,army=4),)+base.empires[1:])}
    rows=[]
    for name,w in scenarios.items():
        for p in profiles():
            choices=[]
            for route in ROUTES:
                c,_=routed_context(w,p,42,0,None,route);key=Policy(principle_priority='finite').choose(c,False)['action_id']
                choices.append(dict(route=route,root=key,after=world_record(act(w,key))))
            rows.append(dict(scenario=name,profile=p['id'],before=world_record(w),choices=choices,
                actions_changed=len({r['root'] for r in choices})>1,physical_outcomes_changed=len({digest(r['after']) for r in choices})>1))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--frozen',action='store_true');a=p.parse_args();out=Path(a.output).resolve()
    if not a.frozen:
        out.mkdir(parents=True,exist_ok=False);source=out/'source';source.mkdir()
        for folder in ('reflex','tools','tests'):shutil.copytree(ROOT/folder,source/folder,ignore=shutil.ignore_patterns('__pycache__'))
        return subprocess.call([sys.executable,str(source/'tools'/Path(__file__).name),'--output',str(out),'--frozen'],cwd=source)
    assert ROOT==out/'source';fixed=hashes()
    pre=dict(format='integrated-contracts-v1',source_hashes=fixed,schema_connections=30,schema_personas=4,
        route_jobs=[dict(profile=p['id'],seed=s,route=r) for p in profiles() for s in range(1700,1704) for r in ('science','culture','power')],
        monitor_contract='observe own actions, public passive changes and final settlement; 12 own opportunities measured in global turns; criterion failure distinct from overall game victory',
        limitations=['public-fact omission only, not learned abstraction or stronger play','route-specific evaluation is authored by resource game',
            'criterion success is separately measured from same-state causal action changes','fixed instructions not adaptive strategic intelligence'])
    (out/'preregister.json').write_text(json.dumps(pre,indent=2)+'\n',encoding='utf-8')
    starts=schemas();contrasts=interventions();runs=[]
    with (out/'routes.jsonl').open('x',encoding='utf-8') as stream:
        for profile in profiles():
            for seed in range(1700,1704):
                for route in ('science','culture','power'):
                    r=routes(profile,seed,route);stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush()
                    runs.append({k:v for k,v in r.items() if k!='trace'});print('routes',len(runs),r['profile'],route,r['won'],flush=True)
    assert hashes()==fixed
    result=dict(format=pre['format'],schema=starts,interventions=contrasts,routes=runs,source_hashes=fixed,preregister_sha256=digest(pre),
        schema_starts=len(starts),schema_connections=len({r['game'] for r in starts}),route_episodes=len(runs),
        causal_states=len(contrasts),causal_changed=sum(r['physical_outcomes_changed'] for r in contrasts))
    (out/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('completed',result['schema_starts'],result['route_episodes'],result['causal_changed'],flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
