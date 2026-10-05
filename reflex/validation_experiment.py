"""Common cross-genre acceptance harness. No personality-specific assistance.

Every variant uses the same Policy, DecisionLoop and optional NeedPressure.
Rules, goal meanings, public opponent forecasts and progress labels remain
game-owned. Future proxies are experiments; they are not immediate learning.
"""
from pathlib import Path
from dataclasses import asdict,replace
from collections import Counter
import copy
import json
import time
import numpy as np
from .core import Policy,FEATURES,compile_batch,digest
from .decision_loop import DecisionLoop,Request,IntentRequest
from .judgment import Binding
from .pressure import NeedPressure
from .progress import ProgressWatch,ProgressConfig
from .progress_adapters import combat_request,combat_feedback,economy_request,economy_feedback
from .laboratory import profiles
from .cross_games import core_hashes
from .teachers import write_json
from .planning import vector


class Session:
    def __init__(self,enabled,learn=False,progress=False,proof_margin=None):
        self.enabled=enabled;self.loop=None;self.pending=None;self.actions=Counter()
        self.learn=learn;self.situation='public';self.learned_uses=0;self.resets=0
        self.max_debt=0.;self.max_stall=0;self.maintained=0;self.decisions=0;self.elapsed=[]
        self.progress_enabled=progress;self.purpose_builder=None;self.progress_blocks=0;self.progress_unresolved=0
        self.purpose_contract=None;self.purpose_evidence=None
        self.proof_margin=proof_margin

    def request(self,c):
        # Learning requires the same one-turn feature definition on both sides.
        # Other adapters retain censored vector feedback; no future samples.
        estimated=tuple(f for f in FEATURES if f!='cost') if self.learn else ()
        condition=self.situation+':'+c['facts'].get('chosen_route','no-route') if self.learn else self.situation
        purpose=self.purpose_builder(c) if self.progress_enabled else None
        self.purpose_contract=copy.deepcopy(purpose)
        return Request(c,{a['id']:Binding(a['id'],condition,estimated) for a in c['actions']},{a['id']:('cost',) if self.learn else () for a in c['actions']},purpose=purpose)

    def initialize(self,c):
        if self.loop is None:
            self.loop=DecisionLoop(c,pressure=NeedPressure(c['scope']) if self.enabled else None,
                progress=ProgressWatch(c['scope'],ProgressConfig(proof_margin=self.proof_margin)) if self.progress_enabled else None)
            self.monitor=NeedPressure(c['scope'])

    def prepare(self,c,route=None,build=None,route_planner=None):
        self.initialize(c)
        return self.request(c) if route is None else IntentRequest(route,lambda key:self.request(build(key)),uncertainty=.35,route_planner=route_planner)

    def accept(self,result):
        # Independent numeric single-item scoring checks the batched decision.
        if not result.get('deliberation',{}).get('adopted') and not result.get('root_completion',{}).get('adopted') and not result.get('progress',{}).get('applied'):assert Policy().choose(result['context'])==result['decision']
        else:
            assert result['decision']['context_hash']==digest(result['context'])
            if result.get('deliberation',{}).get('adopted'):assert result['deliberation']['target']=='modeled-horizon'
        if 'progress' in result:
            self.progress_blocks+=int(result['progress']['applied']);self.progress_unresolved+=int(result['progress']['unresolved'])
        self.last_plan=copy.deepcopy(result.get('deliberation',{}))
        self.pending=result;key=result['decision']['action_id'];self.actions[key.split(':')[0]]+=1;self.decisions+=1
        if self.progress_enabled:
            self.purpose_evidence=dict(scope=copy.deepcopy(result['context']['scope']),tick=result['context']['tick'],root=key,
                request=asdict(self.purpose_contract),before=self.loop.progress.record(),
                mask={k:result['progress'][k] for k in ('blocked','applied','unresolved')})
            if 'recovery' in result['progress']:self.purpose_evidence['recovery']={k:result['progress']['recovery'][k] for k in ('needed','base_purpose','candidate_purpose','margin')}
        self.learned_uses+=result['learning'][key]['empirical_weight']>0
        return key

    def choose(self,c,route=None,build=None,planner=None,route_planner=None):
        t=time.perf_counter();request=self.prepare(c,route,build,route_planner)
        result=DecisionLoop.decide_batch([(self.loop,request)],planner=planner)[0];self.elapsed.append((time.perf_counter()-t)*1000)
        return self.accept(result)

    def feedback(self,progress,maintained=(),completed=(),observed=None,purpose_feedback=None):
        # Actual goal progression has a separate contract from a predicted
        # consequence vector. Censor that vector rather than fabricate it.
        self.monitor.observe(self.pending['context']['tick'],progress,maintained,completed)
        self.max_stall=max(self.max_stall,max(self.monitor.stalls.values()))
        self.maintained+=len(maintained)
        if self.learn:
            if observed is None:raise ValueError('same-target actual one-turn effect required')
            kwargs=dict(need_progress=progress,maintained=maintained,completed=completed) if self.enabled else {}
            if self.progress_enabled:kwargs['purpose_feedback']=purpose_feedback
            updated=self.loop.observe(self.pending['ticket'],vector(observed),**kwargs)
            self.resets+=updated['memory']['regime_reset']
            if self.enabled:self.max_debt=max(self.max_debt,max(self.loop.pressure.debt.values()))
        elif self.enabled:
            self.loop.abandon(self.pending['ticket'],need_progress=progress,maintained=maintained,completed=completed,purpose_feedback=purpose_feedback)
            self.max_debt=max(self.max_debt,max(self.loop.pressure.debt.values()))
        else:self.loop.abandon(self.pending['ticket'],purpose_feedback=purpose_feedback)
        if self.progress_enabled:
            self.purpose_evidence.update(feedback=asdict(purpose_feedback) if purpose_feedback is not None else None,after=self.loop.progress.record())
        self.pending=None

    def summary(self):
        if self.loop is not None:
            if not self.learn:assert not self.loop.memory.entries, 'future-containing effects must not become immediate samples'
            r=json.loads(json.dumps(self.loop.record()));c=copy.deepcopy(self.initial_context())
            assert DecisionLoop.from_record(c,r).record()==r
        summary=dict(actions=dict(self.actions),decisions=self.decisions,max_need_pressure=self.max_debt,
            longest_observed_stall=self.max_stall,maintained_labels=self.maintained,
            learned_uses=self.learned_uses,regime_resets=self.resets,
            decision_p50_ms=float(np.median(self.elapsed)) if self.elapsed else None)
        if self.progress_enabled:summary.update(progress_blocks=self.progress_blocks,progress_unresolved=self.progress_unresolved)
        return summary

    def initial_context(self):
        # Restore the actor contract with a legal current context; saved state
        # is validated by from_record, not synthesized from a world oracle.
        return self.last_context


def remember(session,c):
    session.last_context=copy.deepcopy(c);return c


def auction(profile,seed,scenario,mode,game='auction',record=True):
    from .contests import PublicBeliefs,make_context,settle,target,capital_price
    from .contests_experiment import opening,terminal_scores,SCENARIOS
    seat=seed%4;s,deck,categories=opening(game,seed,0);beliefs=PublicBeliefs(4,seat)
    ss=Session(mode!='baseline');trace=[];paid=0;negative=0
    controllers=iter(SCENARIOS[scenario]);names={i:next(controllers) for i in range(4) if i!=seat}
    for tick in range(len(deck)):
        before=s;c,_=make_context(s,seat,profile,seed,tick,beliefs=beliefs,samples=32)
        key=ss.choose(remember(ss,c));bid=int(key.split(':')[1])
        # Current sealed bids are constructed only AFTER the focal choice.
        bids=tuple(bid if i==seat else target(before,i,names[i]) for i in range(4))
        s,winner=settle(before,bids);beliefs.reveal(before,bids)
        payment=before.budgets[seat]-s.budgets[seat];paid+=payment
        gain=s.scores[seat]-before.scores[seat];negative+=max(0,-gain)
        progress=float(np.clip((gain-capital_price(before,seat)*payment)/30,-1,1)) if game=='auction' else float(np.clip(gain/70,-1,1))
        # Waiting may preserve a valuable capital option. If all estimated
        # esteem effects are nonpositive, lack of acquisition is not failure.
        useful_pass=game=='auction' and bid==0 and max(sum(o['p']*o['needs'].get('esteem',0) for o in a['outcomes']) for a in c['actions'])<=.005
        ss.feedback({'esteem':progress},('esteem',) if useful_pass else ())
        if tick+1<len(deck):s=replace(s,prize=int(deck[tick+1]),category=categories[tick+1],remaining=tuple(sorted(deck[tick+2:])))
        if record:trace.append(dict(before=asdict(before),after=asdict(s),bids=bids,input_context=ss.last_context,decision=key))
    scores=terminal_scores(s);wins=np.flatnonzero(scores==scores.max())
    return dict(genre=game,scenario=scenario,profile=profile['id'],seed=seed,mode=mode,
        won=seat in wins,win_credit=1/len(wins) if seat in wins else 0,
        score=float(scores[seat]),margin=float(scores[seat]-max(x for i,x in enumerate(scores) if i!=seat)),
        payment=paid,negative_points=negative,unused_budget=s.budgets[seat],**ss.summary(),trace=trace)


def resources(profile,seed,scenario,mode,record=True,planner=None,progress=False):
    from .resource_world import World,step,terminal,winners,economic_move,goal_progress,world_record,deficits
    from .resource_experiment import SCENARIOS,public_event
    from .route_experiment import route_context,routed_context,route_potential
    config=SCENARIOS[scenario];w=World.start(routes=config['routes']);seat=seed%4;ss=Session(mode!='baseline',progress=progress);trace=[];tick=0;last_observed=None
    while not terminal(w):
        before=w;key=None;c=None
        if w.turn==seat:
            if progress:ss.purpose_builder=lambda c,current=before,prior=last_observed:economy_request(current,c,prior)
            rc,_=route_context(w,profile,seed,tick)
            build=lambda route:routed_context(w,profile,seed,tick,None,route)[0]
            initial=remember(ss,build(next(iter(w.routes))))
            rc['scope']['episode']=initial['scope']['episode']
            if planner is not None:
                from .resource_planning import forecast,route_forecast
                joint=lambda cs,ds:forecast(before,cs,ds,planner)
                route_plan=lambda rc:route_forecast(before,rc,planner)
            else:joint=None;route_plan=None
            key=ss.choose(initial,rc,build,planner=joint,route_planner=route_plan);route=ss.loop.strategy.chosen
        else:key=economic_move(w)
        w=step(w,key)
        if w.round!=before.round:w=public_event(w,config['event'])
        if before.turn==seat:
            # Capital and productive prerequisites count as progress through
            # the existing chosen-route potential. Simple stock hoarding alone
            # is not a victory; true winners are scored separately.
            delta=float(np.clip(route_potential(w,seat,route)-route_potential(before,seat,route),-1,1))
            old=deficits(before.empires[seat],before);new=deficits(w.empires[seat],w)
            finished=seat in winners(w)
            # Productive waiting means the selected route's real tokens grew.
            ss.feedback({'growth':delta,'physiology':old['physiology']-new['physiology']},
                ('physiology',) if new['physiology']==0 else (),('growth','physiology') if finished else (),
                purpose_feedback=economy_feedback(before,w,seat,key) if progress else None)
            last_observed=w
        if record:
            row=dict(before=world_record(before),after=world_record(w),action=key)
            if before.turn==seat:row['selected_route']=route
            if progress and before.turn==seat:row['purpose']=[copy.deepcopy(ss.purpose_evidence)]
            if planner is not None and before.turn==seat:
                plan=ss.last_plan;selected=plan.get('selected_plan')
                row['planning']=dict(adopted=plan.get('adopted',False),reason=plan.get('reason',''),
                    selected_path=plan.get('metadata',{}).get('purpose_paths',{}).get(selected))
            trace.append(row)
        tick+=1
    e=w.empires[seat]
    return dict(genre='resources',scenario=scenario,profile=profile['id'],seed=seed,mode=mode,
        won=seat in winners(w),win_credit=1. if seat in winners(w) else 0.,score=goal_progress(e,w.routes),
        shortages=e.shortages,overflow=e.overflow,rounds=w.round,route=ss.loop.strategy.chosen,
        route_switches=ss.loop.strategy.switches,**ss.summary(),trace=trace)


def combat(profile,seed,map_name,goal,mode,record=True,goal_need=None,learn=False,survival_security=False,rival='reference',planner=None,progress_watch=False,initial_world=None,trace_roots=False):
    from .combat import Battle,alive,terminal,victory,make_context,route_context,reference,resolve,progress,battle_record,transition_effect,hit_probability
    if learn and mode=='forecast':raise ValueError('future exposure cannot be learned as immediate feedback')
    from .combat import opponent
    w=Battle.start(map_name,goal) if initial_world is None else initial_world
    if not isinstance(w,Battle) or w.goal!=goal or terminal(w):raise ValueError('nonterminal matching-goal initial combat world required')
    team=seed%2;sessions={};trace=[];conflicts=0;failed_moves=0;tick_cost=[];own_collisions=0;planning_nodes=0;planned_ticks=0;planning_declines=0
    world_seed=int(digest(['combat-world',seed])[:16],16)
    # Offline-only identity: never passed to the planner, choices or RNG.
    trace_execution=planner is not None and planner.trace_execution
    if trace_execution:
        config=asdict(planner);config.pop('trace_execution')
        execution_contract=dict(run=digest([profile,seed,map_name,goal,mode,goal_need,learn,survival_security,rival,progress_watch,config,battle_record(w)]),
            group=f'team-{team}',target='combat-mission-path-'+str(planner.progress_weight),unit='combat-tick')
    while not terminal(w):
        before=w;actors=alive(w,team);items=[];contexts=[];routes={};start=time.perf_counter()
        for actor in actors:
            ss=sessions.setdefault(actor,Session(mode!='baseline',learn,progress=progress_watch,proof_margin=planner.recovery_margin if planner else None))
            if progress_watch:ss.purpose_builder=lambda c,a=actor:combat_request(before,a,c)
            u=w.units[actor];visible=sum(hit_probability(w,actor,j)>0 for j in alive(w,1-team))
            ss.situation=f'{map_name}:health-{u.hp//3}:armed-{u.ammo>0}:visible-{visible}'
            build=lambda route,a=actor:make_context(w,a,profile,seed,route,goal_need=mode=='forecast' if goal_need is None else goal_need,exposure_horizon=3 if mode=='forecast' else 1,survival_security=survival_security)
            c=remember(ss,build('eliminate'));rc=route_context(w,actor,profile,seed)
            items.append((ss,ss.prepare(c,rc,build)));contexts.append(c)
        if planner is not None:
            from .combat_planning import forecast
            joint=lambda cs,ds:forecast(before,actors,cs,ds,planner)
            if planner.recovery_options:
                joint.recover=lambda cs,ds,allowed:forecast(before,actors,cs,ds,planner,root_allowed=allowed)
            if planner.root_completion:
                from .root_completion import combat_completion
                joint.complete_fallback=lambda cs,roots,options:combat_completion(before,actors,cs,roots,options)
        else:joint=None
        results=DecisionLoop.decide_batch([(ss.loop,request) for ss,request in items],planner=joint);elapsed=(time.perf_counter()-start)*1000
        choices={}
        for actor,(ss,_),r in zip(actors,items,results):
            choices[actor]=ss.accept(r);ss.elapsed.append(elapsed);routes[actor]=ss.loop.strategy.chosen
        if results and 'deliberation' in results[0]:
            planning_nodes+=results[0]['deliberation']['metadata']['nodes']
            planned_ticks+=int(results[0]['deliberation']['adopted']);planning_declines+=int(not results[0]['deliberation']['adopted'])
        own_dest=[choices[i] for i in actors if choices[i].startswith('move:')]
        own_collisions+=sum(own_dest.count(k)>1 for k in own_dest)
        for actor in alive(w,1-team):choices[actor]=opponent(w,actor,seed,rival)
        w,audit=resolve(w,choices,world_seed);conflicts+=audit['collisions']
        feedback_start=time.perf_counter()
        for actor in actors:
            failed_moves+=choices[actor].startswith('move:') and before.units[actor].pos==w.units[actor].pos
            route=routes[actor];delta=float(np.clip(progress(w,team,route)-progress(before,team,route),-1,1))
            observed=transition_effect(before,w,actor,choices[actor],route,goal_need=mode=='forecast' if goal_need is None else goal_need,survival_security=survival_security) if learn else None
            sessions[actor].feedback({'esteem':delta},completed=('esteem',) if team in victory(w) else (),observed=observed,
                purpose_feedback=combat_feedback(before,w,actor,choices[actor]) if progress_watch else None)
        tick_cost.append(elapsed+(time.perf_counter()-feedback_start)*1000)
        if record:
            row=dict(before=battle_record(before),after=battle_record(w),choices=choices,audit=audit)
            if trace_roots:row['selected_routes']=dict(routes)
            if progress_watch:row['purpose']=[copy.deepcopy(sessions[i].purpose_evidence) for i in actors]
            if planner is not None and results:
                plan=results[0].get('deliberation',{});selected=plan.get('selected_plan')
                attempt=plan.get('recovery_attempt',plan if 'recovery_options' in plan else None)
                row['planning']=dict(adopted=plan.get('adopted',False),reason=plan.get('reason',''),
                    selected_path=plan.get('metadata',{}).get('plans',{}).get(selected,{}).get('purpose_path_mean'),
                    recovery_attempt=None if attempt is None else {k:attempt.get(k) for k in ('adopted','reason','supported_plans','tier_options_per_actor')})
                if 'root_completion' in results[0]:row['planning']['root_completion']=copy.deepcopy(results[0]['root_completion'])
                if trace_execution and plan.get('adopted'):
                    samples=plan.get('metadata',{}).get('plans',{}).get(selected,{}).get('execution_samples')
                    row['planning']['execution']=dict(**execution_contract,
                        start=before.tick,horizon=planner.horizon,samples=copy.deepcopy(samples))
            trace.append(row)
    stats=[ss.summary() for ss in sessions.values()];counts=Counter()
    for s in stats:counts.update(s['actions'])
    return dict(genre='combat',scenario=map_name+'/'+goal,profile=profile['id'],seed=seed,mode=mode,
        won=team in victory(w),lost=1-team in victory(w),win_credit=1. if team in victory(w) else 0.,
        score=max(progress(w,team,route) for route in ('eliminate','secure')),ticks=w.tick,
        health=sum(w.units[i].hp for i in alive(w,team)),movement_conflicts=conflicts,
        own_proposal_collisions=own_collisions,planning_nodes=planning_nodes,planned_ticks=planned_ticks,planning_declines=planning_declines,
        focal_failed_moves=failed_moves,decision_and_feedback_p50_ms=float(np.median(tick_cost)),
        **(dict(execution_contract=execution_contract) if trace_execution else {}),
        actions=dict(counts),decisions=sum(s['decisions'] for s in stats),
        learned_uses=sum(s['learned_uses'] for s in stats),regime_resets=sum(s['regime_resets'] for s in stats),
        max_need_pressure=max(s['max_need_pressure'] for s in stats),longest_observed_stall=max(s['longest_observed_stall'] for s in stats),
        decision_p50_ms=float(np.median([s['decision_p50_ms'] for s in stats])),trace=trace)


def replay_progress(r):
    """Recheck observed progress and selection masks independently of the loop."""
    from .progress import Activity,PurposeRequest,PurposeFeedback
    count=0;previous={}
    for row in r['trace']:
        for e in row.get('purpose',()):
            watch=ProgressWatch.from_record(e['scope'],e['before']);p=e['request']
            request=PurposeRequest(p['level'],p['readiness'],{k:Activity(**a) for k,a in p['activities'].items()},p.get('maintained',False))
            if r['genre']=='combat':
                from .combat import battle_from_record,legal
                actor=int(e['scope']['npc'].split('-')[-1]);roots=legal(battle_from_record(row['before']),actor)
                actual=combat_feedback(battle_from_record(row['before']),battle_from_record(row['after']),actor,e['root'])
            else:
                from .resource_world import world_from_record,legal
                before=world_from_record(row['before']);after=world_from_record(row['after']);roots=legal(before)
                actual=economy_feedback(before,after,before.turn,e['root'])
                key=digest(e['scope']);prior=previous.get(key)
                assert request.maintained==(prior is not None and economy_feedback(prior,before,before.turn,None).maintained)
                previous[key]=after
            c=dict(scope=e['scope'],actions=[dict(id=k,legal=True,known_failure=False) for k in roots])
            allowed,audit=watch.mask(c,request,e.get('recovery'))
            assert e['root'] in allowed and all(audit[k]==e['mask'][k] for k in e['mask'])
            assert asdict(actual)==e['feedback']
            watch.observe(e['tick'],e['root'],request,PurposeFeedback(**e['feedback']) if e['feedback'] else None)
            assert watch.record()==e['after'];count+=1
    return count


def gambling(profile,seed,scenario,mode,record=True):
    from .gambling import Table,make_context,resolve,terminal,expected,legal,TARGET,record as table_record
    w=Table();ss=Session(mode!='baseline');negative_ev=0;missed=0;trace=[]
    while not terminal(w):
        before=w;c=remember(ss,make_context(w,scenario,profile,seed));key=ss.choose(c)
        best=max(expected(w,scenario,k) for k in legal(w,scenario))
        negative_ev+=expected(w,scenario,key)<-1e-9;missed+=key=='pass' and best>0
        w,delta=resolve(w,scenario,key,seed)
        ss.feedback({'growth':float(np.clip(delta/6,-1,1))},('growth',) if best<=0 else (),('growth',) if w.cash>=TARGET else ())
        if record:trace.append(dict(before=table_record(before),after=table_record(w),action=key,delta=delta))
    return dict(genre='gambling',scenario=scenario,profile=profile['id'],seed=seed,mode=mode,
        won=w.cash>=TARGET,win_credit=1. if w.cash>=TARGET else 0.,score=w.cash,ruined=w.cash==0,
        negative_ev_bets=negative_ev,passed_positive_ev=missed,**ss.summary(),trace=trace)


def replay(r):
    """Re-run real rules from saved pre-states; no forecasts become truth."""
    if r['genre'] in ('auction','hagetaka'):
        from .contests import public_from_record,settle
        for row in r['trace']:
            after,_=settle(public_from_record(row['before']),tuple(row['bids']));expected=public_from_record(row['after'])
            assert replace(after,prize=expected.prize,category=expected.category,remaining=expected.remaining)==expected
    elif r['genre']=='resources':
        from .resource_world import world_from_record,world_record,step
        from .resource_experiment import SCENARIOS,public_event
        for row in r['trace']:
            before=world_from_record(row['before']);after=step(before,row['action'])
            if after.round!=before.round:after=public_event(after,SCENARIOS[r['scenario']]['event'])
            assert after==world_from_record(row['after'])
    elif r['genre']=='combat':
        from .combat import battle_from_record,battle_record,resolve
        for row in r['trace']:
            after,audit=resolve(battle_from_record(row['before']),{int(k):v for k,v in row['choices'].items()},int(digest(['combat-world',r['seed']])[:16],16))
            assert after==battle_from_record(row['after']) and audit==row['audit']
    else:
        from .gambling import Table,resolve,record
        for row in r['trace']:
            after,delta=resolve(Table(**row['before']),r['scenario'],row['action'],r['seed'])
            assert record(after)==row['after'] and delta==row['delta']
    return len(r['trace'])


def experiment(output,seeds=(50,51),progress=None):
    from .combat import MAPS,GOALS
    from .contests_experiment import SCENARIOS as AUCTION
    from .resource_experiment import SCENARIOS as RESOURCE
    from .gambling import SCENARIOS as GAMBLING
    output=Path(output);output.mkdir(parents=True,exist_ok=True);before=core_hashes();runs=[];checks=0
    jobs=[(g,s,auction,dict(game=g)) for g in ('auction','hagetaka') for s in AUCTION]
    jobs += [('resources',s,resources,{}) for s in RESOURCE]
    jobs += [('combat',m+'/'+g,combat,dict(map_name=m,goal=g)) for m in MAPS for g in GOALS]
    jobs += [('gambling',s,gambling,{}) for s in GAMBLING]
    with (output/'trajectories.jsonl').open('w',encoding='utf-8') as stream:
        for genre,scene,runner,extra in jobs:
            modes=('baseline','pressure','forecast') if genre=='combat' else ('baseline','pressure')
            for profile in profiles():
                for seed in seeds:
                    for mode in modes:
                        kw=dict(profile=profile,seed=seed,mode=mode,**extra)
                        if genre!='combat':kw['scenario']=scene
                        r=runner(**kw);checks+=replay(r);stream.write(json.dumps(r,ensure_ascii=False)+'\n');runs.append({k:v for k,v in r.items() if k!='trace'})
            if progress:progress(f'{genre}/{scene}: {len(runs)} games, {checks} rule transitions replayed',flush=True)
    return finalize(output,seeds,runs,checks,before)


def finalize(output,seeds,runs,checks,before):
    """Aggregate recorded games without replaying or changing their decisions."""
    output=Path(output)
    summary=[]
    for genre in ('auction','hagetaka','resources','combat','gambling'):
        for mode in ('baseline','pressure','forecast'):
            for p in profiles():
                rows=[r for r in runs if (r['genre'],r['mode'],r['profile'])==(genre,mode,p['id'])]
                if not rows:continue
                totals={k:sum(r.get(k,0) for r in rows) for k in ('won','lost','ruined','negative_ev_bets','passed_positive_ev','shortages','overflow','movement_conflicts','decisions')}
                summary.append(dict(genre=genre,mode=mode,profile=p['id'],games=len(rows),win_credit=sum(r['win_credit'] for r in rows),
                    mean_score=float(np.mean([r['score'] for r in rows])),max_stall=max(r['longest_observed_stall'] for r in rows),**totals))
    comparisons=[];gaps=[]
    index={(r['genre'],r['scenario'],r['profile'],r['seed'],r['mode']):r for r in runs}
    for genre in ('auction','hagetaka','resources','combat','gambling'):
        for mode in ('pressure','forecast'):
            group=[r for r in runs if (r['genre'],r['mode'])==(genre,mode)]
            if not group:continue
            pairs=[(index[(r['genre'],r['scenario'],r['profile'],r['seed'],'baseline')],r) for r in group]
            comparisons.append(dict(genre=genre,mode=mode,pairs=len(pairs),
                win_better=sum(b['win_credit']>a['win_credit'] for a,b in pairs),
                win_worse=sum(b['win_credit']<a['win_credit'] for a,b in pairs),
                win_same=sum(b['win_credit']==a['win_credit'] for a,b in pairs),
                score_better=sum(b['score']>a['score']+1e-9 for a,b in pairs),
                score_worse=sum(b['score']<a['score']-1e-9 for a,b in pairs),
                score_same=sum(abs(b['score']-a['score'])<=1e-9 for a,b in pairs)))
        for mode in ('baseline','pressure','forecast'):
            group=[s for s in summary if (s['genre'],s['mode'])==(genre,mode)]
            if not group:continue
            rates={s['profile']:s['win_credit']/s['games'] for s in group}
            gaps.append(dict(genre=genre,mode=mode,persona_win_rates=rates,gap=max(rates.values())-min(rates.values()),
                worst_rate=min(rates.values()),total_win_credit=sum(s['win_credit'] for s in group)))
    gambling_runs=[r for r in runs if r['genre']=='gambling']
    adoption=dict(default_policy_changed=False,combat_pressure_accepted=False,combat_forecast_accepted=False,
        reason='experimental combat variants are not default; inspect paired regressions and weakest persona, not just narrower gap',
        gambling_negative_ev_bets=sum(r['negative_ev_bets'] for r in gambling_runs),gambling_ruins=sum(r['ruined'] for r in gambling_runs))
    result=dict(format='cross-genre-validation-v1',seeds=list(seeds),seed_status='50/51 reserved before this shared harness run; 0/1/40 development; not broad generalization evidence',
        games=len(runs),rule_transitions_replayed=checks,core_before=before,core_after=core_hashes(),summary=summary,comparisons=comparisons,persona_gaps=gaps,adoption=adoption,runs=runs,
        contracts=['fixed personality/values; identical per-genre rules/opponents/random namespaces across variants',
            'same DecisionLoop/NeedPressure; game-owned progress labels/effect meanings',
            'numeric batch/single identity checked live on effective contexts; saved input_context is raw input, not effective intent/pressure or adopted teacher data',
            'future-containing effects never stored as immediate empirical outcomes; all vector feedback censored',
            'combat forecast is a stationary HP-weighted threat proxy, not multi-step policy search or calibrated survival'],
        limitations=['small authored games and fixed comparator families; no human-level or arbitrary-rule claim',
            'pressure does not override strongest-principle tier; stagnation can remain',
            'no new empirical learning from this suite; auction public hypothesis updates are existing functionality',
            'JSON/validation route costs included; no large-NPC real-time performance claim'])
    assert before==core_hashes();write_json(output/'evaluation.json',result)
    lines=['# オークション・複数資源/勝利条件・戦闘・ギャンブル共通検証','',f'{len(runs)} games; seeds {list(seeds)}; {checks} real rule transitions replayed.','',
        'baseline は共通接続済み既存評価。pressure は有界な欲求不足の蓄積を追加。戦闘のみ forecast で目標進展の欲求効果と3tick静止脅威代理値も比較する。既定の他実験を置換しない。','',
        '|種別|方式|人格|勝利持分/局数|平均目的点|最大停滞|','|---|---|---|---|---|---|']
    for s in summary:lines.append(f"|{s['genre']}|{s['mode']}|{s['profile']}|{s['win_credit']:g}/{s['games']}|{s['mean_score']:.3f}|{s['max_stall']}|")
    lines += ['', '同一条件対の勝利持分（改善/悪化/同じ）：']
    for c in comparisons:lines.append(f"- {c['genre']}/{c['mode']}: {c['win_better']}/{c['win_worse']}/{c['win_same']}。目的点の改善/悪化/同じ: {c['score_better']}/{c['score_worse']}/{c['score_same']}。")
    lines += ['', '人格間の勝利率の最大差（勝利を均一化する補正なし）：']
    for g in gaps:lines.append(f"- {g['genre']}/{g['mode']}: {g['gap']:.1%}、最小率{g['worst_rate']:.1%}、全人格の勝利持分{g['total_win_credit']:g}。")
    lines += ['', '得たもの: 同じ人格コア・欲求不足・判断/ルート接続を4系統+ハゲタカへ適用し、現実の勝敗/破産/資源不足を比較できる再実行基盤。性格別ハンデ、非公開の当手、実乱数の先読みは追加していない。',
        '', '削ったもの: この試験では未来込みの効果を即時結果として経験学習する接続を使わない。現在の効果ベクトルが未定義な箇所は明示的欠測とし、実際の目的進展だけ別契約で欲求へ返す。',
        '', '増えた費用: 個体ごとの固定5欲求の有界記憶と進展ラベル、JSONコピー/検査。将来評価は戦闘側の代理値であり、一般的な賢さの獲得ではない。',
        '', '合否: 改善・悪化は evaluation.json の対条件と人格別結果で判定する。強い人格を弱めて差を縮める補正は不採用。慎重型の停滞が残る場合は未解決とする。',
        '', 'Colab/GPU/Drive・LLM教師・モデル訓練は使用していない。全てローカルCPU。']
    lines += ['', '実測による採否：']
    for genre in ('auction','hagetaka','resources','combat','gambling'):
        base=next(g for g in gaps if (g['genre'],g['mode'])==(genre,'baseline'))
        for mode in ('pressure','forecast'):
            current=next((g for g in gaps if (g['genre'],g['mode'])==(genre,mode)),None)
            if current is None:continue
            cmp=next(c for c in comparisons if (c['genre'],c['mode'])==(genre,mode))
            verdict='小標本の改善の兆候（一般改善の証明ではない）' if current['worst_rate']>base['worst_rate'] and current['total_win_credit']>=base['total_win_credit'] else '最弱人格の勝利率改善は確認できない'
            lines.append(f"- {genre}/{mode}: 総勝利持分{base['total_win_credit']:g}→{current['total_win_credit']:g}、最小率{base['worst_rate']:.1%}→{current['worst_rate']:.1%}。{verdict}。")
            if genre=='combat' and cmp['win_worse']>cmp['win_better']:lines.append('  改善より悪化条件が多く、品質改善案として棄却。強い人格を弱めて差を縮める修正は採用しない。forecastは目標欲求とH3代理値の複数変更を含み、H3単独の効果ではない。')
    unfavorable=[r for r in gambling_runs if r['scenario']=='unfavorable']
    lines += [f"- ギャンブル{len(gambling_runs)}局: 負の期待利益の賭け{adoption['gambling_negative_ev_bets']}、破産{adoption['gambling_ruins']}。不利条件{len(unfavorable)}局の待機/資金/不足はevaluation.jsonで個別確認。初期資金12と有限16機会の自作ルール内の結果。",
        '', '採用範囲: 再検証基盤と任意のNeedPressure接続。従来のゲーム接続・Policy・数値Populationの既定は維持。戦闘の不足/将来代理値を既定へ置換しない。',
        '', '残る急所: 最強主義の厳密優先と目先の安全代理値が結びつくと、欲求不足だけでは慎重型を動かせない。実際の相手の動き・チームの競合・勝利期限を含む評価を別条件で検証する必要がある。主義を強制解除したり、防御を不可能扱いして押し出す修正はしない。汎用知性/遅延評価・経験学習の統合は未解決。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n');return result
