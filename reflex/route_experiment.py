"""Game-authored route opportunities + actual resource games, no win-EV claim."""
from dataclasses import replace
from pathlib import Path
from collections import Counter
import copy
import json
import math
import time
import numpy as np
from .core import Policy, NEEDS, TRAITS
from .examples import action,context,effect
from .laboratory import profiles
from .routes import RouteState,choose_route
from .resource_world import (World, Empire, ROUTES, income, progress, achieved,
    legal, act, step, winners, terminal, economic_move, make_context, world_record,world_from_record)
from .cross_games import core_hashes
from .teachers import write_json


def opportunity(w,route,viewer):
    """Optimistic work/production proxy from public stocks, not turns-to-win EV.

    Science must pay 6/10/14 research then retain 14; culture spends two times
    8 then retains 20. No forecast of enemy interventions/event schedules.
    Military remains available through invasions even after frontier depletion.
    """
    if route not in ROUTES: raise ValueError('known route required')
    e=w.empires[viewer]; left=max(1,w.limit-w.round)
    if route=='science':
        missing=max(0,sum(6+4*t for t in range(e.tech,3))+14-e.science)
        work=(3-e.tech)+(e.buildings[4]==0)+math.ceil(missing/max(2,2*e.buildings[4]))
        pressure=(e.buildings[4]==0)*.3
    elif route=='culture':
        missing=max(0,8*(2-e.monuments)+20-e.culture)
        work=(2-e.monuments)+(e.buildings[5]==0)+math.ceil(missing/max(2,2*e.buildings[5]))
        pressure=(e.buildings[5]==0)*.3
    else:
        work=max(0,4-e.land)+math.ceil(max(0,6-e.army)/2)
        contested=max(0,4-e.land-w.frontier)
        work+=4*contested
        pressure=min(1,.2*max(0,4-e.land)+.2*contested)
    # Missing finance/ore implies work even when goal tokens are near complete.
    funding=(max(0,3-e.stock[3])+max(0,2-e.stock[2]))/6
    quality=float(np.clip(1-(work+funding)/left,0,1))
    return dict(work_proxy=float(work+funding),quality=quality,pressure=pressure)


def route_context(w,profile,seed,tick):
    viewer=w.turn; choices=[]; estimates={}
    for route in ROUTES:
        est=opportunity(w,route,viewer); estimates[route]=est
        # This mapping belongs to THIS game, not personality psychology. Science
        # provides self-direction; culture tradition; military power/pressure.
        quality=est['quality']; aligns={'science':{'self_direction':.65,'achievement':.3},
            'culture':{'tradition':.65,'achievement':.3},'power':{'power':.65,'achievement':.3}}[route]
        choices.append(action(route,effect(quality,needs={'growth':quality,'safety':-est['pressure']},
            values={**aligns,'security':-est['pressure']},cost=min(.5,est['work_proxy']/72)),legal=route in w.routes))
    c=context('victory-route',choices,{'growth':.4,'safety':.2},profile['values'],dict(zip(TRAITS,profile['traits'])),mode=None)
    for n in NEEDS:
        if n not in ('growth','safety'): c['needs'][n]=dict(supported=False,enabled=False,deficit=None)
    c.update(seed=seed,tick=tick,scope=dict(game='resource-world-routes-v1',episode=f'route-{seed}',npc=f'player-{viewer}'),
        objective='選べる勝利ルートへの本人の志向と現在の機会を評価する',
        facts=dict(opportunity='optimistic public work/production proxy; not calibrated win EV',
            mapping='game-owned: science=self-direction; culture=tradition; power=power',
            public_routes=str(w.routes),public_round=str(w.round)))
    return c,estimates


def route_potential(w,viewer,route):
    if route not in ROUTES: raise ValueError('known route required')
    e=w.empires[viewer]
    if set(achieved(e))&set(w.routes): return 1.
    if route=='science':
        completed=sum(6+4*t for t in range(e.tech))
        tokens=min(1,(completed+e.science)/44)
        projects=e.tech/3; production=min(e.buildings[4],2)/2
    elif route=='culture':
        tokens=min(1,(8*e.monuments+e.culture)/36)
        projects=e.monuments/2; production=min(e.buildings[5],2)/2
    else:
        tokens=min(1,e.land/4); projects=min(1,e.army/6)
        production=min(1,e.buildings[0]/2)
    return .6*(.55*tokens+.25*projects+.15*production+.05*sum(e.stock)/72)


def routed_context(w,profile,seed,tick,memory,route):
    if route not in w.routes: raise ValueError('available game route required')
    c,stats=make_context(w,profile,seed,tick,f'route-game-{seed}',memory,1)
    viewer=w.turn; before=route_potential(w,viewer,route)
    # Known prerequisites remain capability after spending them; winning ANY
    # allowed route has terminal credit even if it is not the intended route.
    for a in c['actions']:
        immediate=act(w,a['id']); e=income(immediate.empires[viewer],immediate)
        roster=list(immediate.empires); roster[viewer]=e
        projected=immediate if winners(immediate) else replace(immediate,empires=tuple(roster))
        exact=route_potential(immediate,viewer,route)-before
        g=exact+.65*(route_potential(projected,viewer,route)-route_potential(immediate,viewer,route))
        a['outcomes'][0]['objective']=g
        a['outcomes'][0]['values']['achievement']=g
    c['facts']['chosen_route']=route
    c['facts']['route_value']='completed investments + stock + own infrastructure; game-authored nonterminal proxy'
    return c,stats


def run(profile,seed,routed=True,record=True):
    w=World.start(); seat=seed%4; memory=None; route=RouteState(); trace=[]; timings=[]; route_counts=Counter()
    while not terminal(w):
        before=w; c=d=proposal=None
        if w.turn==seat:
            t=time.perf_counter()
            if routed:
                rc,estimates=route_context(w,profile,seed,len(trace))
                route,proposal=choose_route(rc,route,uncertainty=.35)
                proposal['estimates']=estimates; route_counts[route.chosen]+=1
                c,_=routed_context(w,profile,seed,len(trace),memory,route.chosen)
            else: c,_=make_context(w,profile,seed,len(trace),f'route-game-{seed}',memory,1)
            d=Policy().choose(c); memory=d['next_state']; key=d['action_id']
            timings.append((time.perf_counter()-t)*1000)
        else: key=economic_move(w)
        w=step(w,key)
        trace.append(dict(before=world_record(before),after=world_record(w),context=c,decision=d,proposal=proposal,action=key) if record else None)
    return dict(profile=profile['id'],seed=seed,seat=seat,routed=routed,won=seat in winners(w),winners=list(winners(w)),
        rounds=w.round,achieved=list(achieved(w.empires[seat])),route_counts=dict(route_counts),switches=route.switches,
        decision_p50_ms=float(np.median(timings)),final=world_record(w),trace=trace)


def replay(r):
    state=RouteState(); checked=0
    for row in r['trace']:
        w=world_from_record(row['before']); assert world_record(step(w,row['action']))==row['after']
        if row['context']:
            assert Policy().choose(row['context'])==row['decision']; checked+=1
        if row['proposal']:
            state,actual=choose_route(row['proposal']['context'],state,uncertainty=.35)
            assert actual['chosen']==row['proposal']['chosen']
    return checked


def probe():
    base=World.start(); scenarios={'initial_world':base,'science_capacity':replace(base,empires=(replace(Empire(),tech=2,science=12,buildings=(1,1,1,0,2,0)),)+base.empires[1:]),
        'culture_capacity':replace(base,empires=(replace(Empire(),monuments=1,culture=12,buildings=(1,1,1,0,0,2)),)+base.empires[1:]),
        'military_capacity':replace(base,empires=(replace(Empire(),land=3,army=4),)+base.empires[1:]),
        'military_closed':replace(base,routes=('science','culture'))}
    rows=[]
    for name,w in scenarios.items():
        for p in profiles():
            c,estimates=route_context(w,p,7,0); state,d=choose_route(c)
            rows.append(dict(scenario=name,profile=p['id'],chosen=state.chosen,scores=d['scores'],estimates=estimates))
    return rows


def experiment(output,seeds=4,progress=None):
    output=Path(output); output.mkdir(parents=True,exist_ok=True); before=core_hashes(); runs=[]; checks=0
    with (output/'trajectories.jsonl').open('w',encoding='utf-8') as stream:
        for p in profiles():
            for seed in range(seeds):
                for routed in (False,True):
                    r=run(p,seed,routed); checks+=replay(r)
                    stream.write(json.dumps(r,ensure_ascii=False)+'\n'); runs.append({k:v for k,v in r.items() if k!='trace'})
            if progress: progress(f'victory-routes/{p["id"]}: {len(runs)} games, {checks} choices replayed')
    assert before==core_hashes(); summary=[]
    for p in profiles():
        for routed in (False,True):
            group=[r for r in runs if r['profile']==p['id'] and r['routed']==routed]
            counts=Counter()
            for r in group: counts.update(r['route_counts'])
            summary.append(dict(profile=p['id'],routed=routed,games=len(group),wins=sum(r['won'] for r in group),
                switches=sum(r['switches'] for r in group),chosen_routes=dict(counts),
                median_p50_ms=float(np.median([r['decision_p50_ms'] for r in group]))))
    result=dict(format='victory-route-v1',games=len(runs),seeds=seeds,replayed_decisions=checks,
        core_before=before,core_after=core_hashes(),summary=summary,probe=probe(),runs=runs,
        limitations=['route opportunities/effects/progress are game-authored proxies, not universal automatic win valuation',
            'no opponents/future event simulation in opportunity estimate',
            'development probe; no validated Big Five-to-victory psychological mapping',
            'route preference and route-specific value changed together, so attribution is limited',
            'timings include possible concurrent local experiment load; not isolated throughput measurements',
            'hard rejection/known failure and hysteresis handle route change; calibrated route advantage prediction pending'])
    write_json(output/'evaluation.json',result)
    lines=['# 人格に沿った有限勝利ルート','',f'{len(runs)} headless four-player resource games; {checks} choices replayed; shared core/runtime unchanged.', '',
        '|profile|route layer|games|wins|switches|route decision counts|p50 ms|', '|---|---|---:|---:|---:|---|---:|']
    for s in summary: lines.append(f'|{s["profile"]}|{s["routed"]}|{s["games"]}|{s["wins"]}|{s["switches"]}|{s["chosen_routes"]}|{s["median_p50_ms"]:.2f}|')
    lines+=['','## 得たものと費用','',
        'ゲームが有限の勝利候補、公開資源による進めやすさ、本人への効果を供給し、共通Policyが選ぶ。特性と勝利名をコアに直結しない。科学は自己決定、文化は伝統、勢力は権力という結びつきはこの自作ゲームの仮定。',
        'ルート記憶は単発行動の記憶と分離。わずかなスコア差では変更せず、利用不能・大きな利得差なら変更する。性格自体は更新しない。',
        '研究/建築で消費済みの蓄積を完成済み能力として保持する専用代理値を接続した。ルールだけで評価が自然発生する能力は得ていない。比較はルート層と専用評価の組み合わせであり、どちらの単独効果かは断定できない。',
        '', '## 同じ盤面における人格・機会の選択','', '|scenario|profile|route|','|---|---|---|']
    lines += [f'|{r["scenario"]}|{r["profile"]}|{r["chosen"]}|' for r in result['probe']]
    lines+=['','## 残る制限','']+[f'- {x}' for x in result['limitations']]+['']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8'); return result
