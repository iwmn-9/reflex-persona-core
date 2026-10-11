"""Whole-episode integration trials with one owned conditional policy memory."""
import copy
from .core import Policy,digest
from .decision_loop import DecisionLoop,Request
from .judgment import Binding
from .flow_rollout import rollout
from .purpose_plan import goal_forecast
from .contingent_search import search_forecast
from .observation_plan import PlanMemory
from .contingent_games import make_bridge


def play(case,profile,seed,method='conditional',horizon=3,revisions=2,patch_budget=4,validation_seeds=None,*,policy=None):
    if method not in ('goal-reflex','conditional'):raise ValueError('known conditional ablation required')
    game=make_bridge(profile,case);world=game.start();c=game.observe(game.project(world))
    policy=policy or Policy()
    loop=DecisionLoop(c,policy=policy);memory=PlanMemory(c,game.target,game.unit);trace=[]
    fixed=digest([c['scope'],c['seed'],c['personality'],c['values'],c['objective']])
    while not game.terminal(world):
        current=game.project(world);c=game.observe(current,loop.state);captured={}
        assert fixed==digest([c['scope'],c['seed'],c['personality'],c['values'],c['objective']])
        def planner(cs,ds):
            saved=digest(loop.record());saved_plan=memory.record()
            if method=='conditional':
                retained,prefix=memory.offer(cs[0])
                f=search_forecast(cs[0],current,observe=game.observe,advance=game.modeled_advance,
                    terminal=game.terminal,assess=lambda w:game.goal(w).record(),horizon=horizon,
                    seeds=game.seeds,width=1,revisions=revisions,target=game.target,unit=game.unit,
                    retained=retained,prefix=prefix,patch_budget=patch_budget,settlement_weight='absolute',validation_seeds=validation_seeds,policy=policy)
            else:
                paths,audit=rollout(cs[0],current,observe=game.observe,advance=game.modeled_advance,
                    terminal=game.terminal,assess=lambda w:game.goal(w).record(),horizon=horizon,
                    seeds=game.seeds if validation_seeds is None else validation_seeds,public_controller=lambda view:None,policy=policy)
                f=goal_forecast(cs[0],paths,audit,horizon=horizon,unit=game.unit,target=game.target,
                    settlement_weight='absolute',policy=policy)
                captured['baseline_audit']=audit
            assert saved==digest(loop.record()) and saved_plan==memory.record()
            captured['forecast']=f
            return f
        keys=[a['id'] for a in c['actions']]
        req=Request(c,{k:Binding(k,'public',()) for k in keys},{k:() for k in keys})
        result=DecisionLoop.decide_batch([(loop,req)],False,planner)[0]
        key=result['decision']['action_id'];f=captured['forecast'];selected=result['deliberation'].get('selected_plan')
        if method=='conditional':
            bank=f.audit['contingent_search'];proposal=f.audit['plans'][selected]['proposal'] if selected else next(k for k,p in bank['proposals'].items() if p['root']==key and not p['program']['entries'])
            branches=bank['proposals'][proposal]['branches'];program=bank['proposals'][proposal]['program']
            audit=dict(evaluated=bank['evaluated'],legal_roots=len(bank['all_viable_roots']),
                extra_slots=bank['extra_slots'],program=program)
            memory.remember(c,f,selected,key)
            assert PlanMemory.restore(c,memory.record()).record()==memory.record()
        else:branches=captured['baseline_audit'][key];audit=None
        before=game.record(world);world,flow=game.actual(world,key,seed)
        loop.abandon(result['ticket'])
        assert not loop.memory.entries
        trace.append(dict(before=before,after=game.record(world),root=key,flow=flow,
            goal=game.goal(world).record(),decision=result['decision'],deliberation=result['deliberation'],
            context_hash=digest(c),persona_hash=fixed,branches=branches,conditional=audit))
        if len(trace)>1000:raise AssertionError('whole episode did not settle')
    return dict(case=case,profile=profile['id'],seed=seed,method=method,horizon=horizon,
        goal=game.goal(world).record(),steps=len(trace),actions=[t['root'] for t in trace],trace=trace)
