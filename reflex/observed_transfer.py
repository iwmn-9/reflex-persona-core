"""Fixed-coefficient observer ablation plus transfer to new public logistics.

No outcome-dependent parameter tuning, empirical learning, hidden actual seed
or diagnostic oracle is supplied to any decision. Optional CPU workers execute
independent episodes, not NPCs sharing one simulated world.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
import copy
import json
import subprocess
from .core import digest,Policy
from .laboratory import profiles
from .purpose_recovery import GoalProbe
from .delivery_world import DeliveryProbe
from .purpose_plan import goal_forecast,PersonaProgressWatch
from .model_observer import ModelObserver,ModelFeedback
from .flow_rollout import rollout
from .pressure import NeedPressure,PressureConfig
from .progress import ProgressConfig
from .decision_loop import DecisionLoop,Request
from .judgment import Binding
from .purpose_experiment import source_hashes


def make_probe(spec,profile):
    return DeliveryProbe(spec,profile) if spec['genre']=='delivery' else GoalProbe(spec,profile)


def feedback(p,before,after,key,watched,pressured):
    fb=p.feedback(before,after,key)
    gain=max(-1.,min(1.,p.goal(after).value-p.goal(before).value))
    return ModelFeedback(fb if watched else None,{'growth':gain} if pressured else None,
        ('growth',) if pressured and fb.maintained else (),
        ('growth',) if pressured and p.goal(after).status=='success' else ())


@lru_cache(maxsize=100000)
def can_finish(w):
    """Exact public logistics reachability, diagnostic only, never a feature."""
    from .delivery_world import terminal,goal,step,legal
    if terminal(w):return goal(w).status=='success'
    return any(can_finish(step(w,k)) for k in legal(w))


def run(spec,profile,variant,seed,horizon=6,*,search=False,width=2,depth=2,continuity=False,principle_priority='lexicographic',max_regret=.15):
    watched=variant in ('watch','release','observed','verified')
    pressured=variant in ('pressure','release','observed','verified')
    if variant not in ('goal','watch','pressure','release','observed','verified'):raise ValueError('known ablation required')
    if continuity and not search:raise ValueError('continuity requires explicit search proposals')
    p=make_probe(spec,profile);w=p.start();c=p.observe(w)
    from .plan_continuity import PlanIntention
    intention=PlanIntention(c) if continuity else None
    watch=PersonaProgressWatch(c['scope'],ProgressConfig(grace=2,repeat_limit=2,
        proof_margin=0. if variant=='verified' else None)) if watched else None
    pressure=NeedPressure(c['scope'],PressureConfig(grace=2)) if pressured else None
    policy=Policy(principle_priority=principle_priority)
    loop=DecisionLoop(c,policy=policy,progress=watch,pressure=pressure);trace=[];previous=None
    while not p.terminal(w):
        c=p.observe(w,loop.state);pr=p.purpose(w,c,previous) if watched else None;captured={}
        def planner(cs,ds):
            observer=None
            if variant=='observed':
                observer=ModelObserver(c['scope'],purpose=p.purpose,
                    feedback=lambda a,b,k:feedback(p,a,b,k,watched,pressured),
                    progress=loop.progress,pressure=loop.pressure,previous=previous)
            saved=digest(loop.record())
            saved_intention=None if intention is None else intention.record()
            if search:
                from .continuation_search import search_forecast
                f=search_forecast(cs[0],w,observe=p.observe,advance=p.advance,terminal=p.terminal,
                    horizon=horizon,seeds=p.seeds,assess=lambda s:p.goal(s).record(),observer=observer,
                    width=width,depth=depth,target=p.target,policy=policy,max_regret=max_regret,
                    retained=() if intention is None else intention.offer(cs[0],target=p.target,unit='public-turns',horizon=horizon))
                assert digest(loop.record())==saved,'hypothetical state reached actual owner'
                assert intention is None or intention.record()==saved_intention,'planning changed actual intention'
                captured.update(forecast=f)
                return f
            paths,audit=rollout(cs[0],w,observe=p.observe,advance=p.advance,terminal=p.terminal,
                horizon=horizon,seeds=p.seeds,assess=lambda s:p.goal(s).record(),observer=observer,policy=policy)
            assert digest(loop.record())==saved,'hypothetical state reached actual owner'
            captured.update(audit=audit)
            return goal_forecast(cs[0],paths,audit,horizon=horizon,unit='public-turns',target=p.target,policy=policy,max_regret=max_regret)
        req=Request(c,{k:Binding(k,'public',()) for k in p.keys(w)},{k:() for k in p.keys(w)},purpose=pr)
        result=DecisionLoop.decide_batch([(loop,req)],False,planner)[0]
        key=result['decision']['action_id'];before=w;w,row=p.actual(w,key,seed)
        if search:
            f=captured['forecast'];proposals=f.audit['continuation_search']['proposals']
            selected=result['deliberation'].get('selected_plan')
            proposal=f.audit['plans'][selected]['proposal'] if selected else next(k for k,n in proposals.items() if n['root']==key and not n['schedule'])
            captured['audit']={key:proposals[proposal]['branches']}
        fb=feedback(p,before,w,key,watched,pressured)
        loop.abandon(result['ticket'],need_progress=fb.needs,maintained=fb.maintained,
            completed=fb.completed,purpose_feedback=fb.purpose)
        if intention is not None:intention.remember(c,captured['forecast'],result['deliberation'].get('selected_plan'),key)
        assert not loop.memory.entries and c['personality']==loop.personality and c['values']==loop.values
        trace.append(dict(before=asdict(before),after=asdict(w),root=key,flow=row,goal=p.goal(w).record(),
            purpose=None if pr is None else asdict(pr),feedback=asdict(fb),state=copy.deepcopy(loop.state),
            progress=result.get('progress'),watch=None if loop.progress is None else loop.progress.record(),
            pressure=None if loop.pressure is None else loop.pressure.record(),
            persona_hash=digest([c['personality'],c['values']]),endpoints=captured['audit'][key],
            deliberation=result['deliberation']))
        if intention is not None:trace[-1]['intention']=intention.record()
        previous=before
    reversals=0
    if p.genre=='combat':
        for i in range(len(trace)-2):
            a=trace[i]['after']['units'][0];b=trace[i+2]['after']['units'][0]
            reversals+=int((a['x'],a['y'])==(b['x'],b['y']) and trace[i+1]['root'].startswith('move:') and trace[i+2]['root'].startswith('move:'))
    lost=None
    if p.genre=='delivery':
        from .delivery_world import Depot
        lost=sum(can_finish(Depot(**t['before'])) and not can_finish(Depot(**t['after'])) for t in trace)
    return dict(spec=spec,profile=profile['id'],variant=variant,seed=seed,status=p.goal(w).status,
        goal_value=p.goal(w).value,actions=[t['root'] for t in trace],steps=len(trace),
        surviving=w.units[0].hp>0 if p.genre=='combat' else None,
        shortages=w.empires[0].shortages if p.genre=='resources' else None,
        move_reversals=reversals,solvable_route_lost=lost,
        delivered=w.delivered if p.genre=='delivery' else None,community=w.community if p.genre=='delivery' else None,
        trace=trace)


def jobs():
    diagnostic=[dict(genre='combat',hp=6,limit=9,distance=4)]
    confirmation=[dict(genre='resources',route='mixed',limit=12),
        dict(genre='auction',prizes=[6,13,9],budget=9,rival_shift=True),
        dict(genre='combat',hp=5,limit=10,distance=4),
        dict(genre='delivery',limit=10,capacity=3,supply=2),
        dict(genre='delivery',limit=8,capacity=2,supply=3)]
    holdout=[dict(genre='resources',route='mixed',limit=13),
        dict(genre='auction',prizes=[11,5,14],budget=10,rival_shift=True),
        dict(genre='combat',hp=4,limit=11,distance=4),
        dict(genre='delivery',limit=9,capacity=2,supply=2)]
    for split,specs,variants,seeds in (
        ('diagnostic',diagnostic,('goal','watch','pressure','release','observed','verified'),(210,211)),
        ('confirmation',confirmation,('goal','release','observed','verified'),(230,231)),
        ('holdout',holdout,('goal','release','verified'),(250,251))):
        for s in specs:
            for p in profiles():
                for seed in seeds if s['genre']=='combat' else seeds[:1]:
                    for v in variants:yield split,s,p,v,seed,6


def worker(args):
    split,spec,p,v,seed,h=args;r=run(spec,p,v,seed,h);r['split']=split
    return r


def aggregate(runs):
    result=[]
    for split,genre,v in sorted({(r['split'],r['spec']['genre'],r['variant']) for r in runs}):
        rs=[r for r in runs if (r['split'],r['spec']['genre'],r['variant'])==(split,genre,v)]
        result.append(dict(split=split,genre=genre,variant=v,episodes=len(rs),
            successes=sum(r['status']=='success' for r in rs),failures=sum(r['status']=='failure' for r in rs),
            goal_sum=sum(r['goal_value'] for r in rs),surviving=sum(r['surviving'] is True for r in rs),
            shortages=sum(r['shortages'] or 0 for r in rs),move_reversals=sum(r['move_reversals'] for r in rs),
            solvable_route_lost=sum(r['solvable_route_lost'] or 0 for r in rs),
            delivered=sum(r['delivered'] or 0 for r in rs),community=sum(r['community'] or 0 for r in rs)))
    return result


def experiment(output,progress=None,workers=1):
    if type(workers) is not int or not 1<=workers<=8:raise ValueError('bounded independent CPU workers required')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if any((output/n).exists() for n in ('preregister.json','trajectories.jsonl','evaluation.json')):
        raise FileExistsError('frozen transfer evidence exists; use a fresh output')
    fixed=source_hashes();tasks=list(jobs())
    try:base=subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):base=None
    pre=dict(format='observed-transfer-v1',base=base,source_hashes=fixed,jobs=tasks,workers=workers,
        parameters=dict(horizon=6,max_regret=.15,grace=2,repeat_limit=2,pressure_grace=2),
        model_combat_seeds=[820,821,822,823],
        status='prior 112-case observer-only batch exposed diagnostic/confirmation conditions; verified adds existing strict positive-purpose recovery proof with margin zero, not new scoring coefficients; holdout/source frozen before this batch; no tuning after holdout; not representative general strength',
        adoption='keep global defaults; recommend verified only if holdout has no per-persona settlement regression versus both goal/release, no increased shortages/reversals/solvable-route loss, and some reduction of unsupported motion; observer alignment alone requires separate benefit evidence')
    (output/'preregister.json').write_text(json.dumps(pre,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    runs=[]
    with (output/'trajectories.jsonl').open('w',encoding='utf-8',newline='\n') as stream:
        def consume(results):
            for r in results:
                stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush()
                runs.append({k:v for k,v in r.items() if k!='trace'})
                if progress and len(runs)%8==0:progress(f'{len(runs)}/{len(tasks)} fixed-source episodes',flush=True)
        if workers==1:consume(map(worker,tasks))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:consume(pool.map(worker,tasks))
    assert source_hashes()==fixed
    result=dict(format=pre['format'],episodes=len(runs),summary=aggregate(runs),runs=runs,
        source_hashes=fixed,preregister_digest=digest(pre),default_policy_changed=False,
        coefficients_changed=False,training_performed=False,cloud_runtime_used=False)
    (output/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    return result
