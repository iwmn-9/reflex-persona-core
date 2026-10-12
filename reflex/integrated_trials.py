"""Existing game rules exercising return units and a detached owned executor.

Adapters below own public support, real feedback and purpose proxies. The
generic actor/return layers contain no game names. The broad bridges do not
invent an empirical target: outcome learning is enabled only for the three
existing rule probes whose immediate feedback contract is explicit.
"""
import ast,copy,json
from .core import Policy,digest,FEATURES,character_key
from .decision_loop import DecisionLoop,Request
from .judgment import Binding
from .flow_rollout import rollout
from .purpose_plan import goal_forecast
from .planning import vector
from .contingent_games import make_bridge


TRAPS=('cyclic_arena','shared_commons','branching_delivery')


def request_for(c,game,*,learn=False,social=None):
    exact={a['id']:() for a in c['actions']};situation='public-rule'
    if game in TRAPS and learn:
        from .judgment_experiment import Arena,Commons,Routes,LEARNED
        exact={'cyclic_arena':Arena,'shared_commons':Commons,'branching_delivery':Routes}[game]().exact()
        if game=='shared_commons':
            history=ast.literal_eval(c['facts']['public_joint_history'])
            situation='claims-'+str(history[-1].count('CLAIM')) if history else 'unobserved'
        elif game=='branching_delivery':situation=c['facts']['room']+'-survey-'+c['facts']['survey']
        else:situation='same-ring'
        bindings={a['id']:Binding(a['id'],situation,tuple(f for f in LEARNED if f not in exact[a['id']]) if a['id']!='CHECK' else ()) for a in c['actions']}
    else:bindings={a['id']:Binding(a['id'],situation,()) for a in c['actions']}
    extra={} if social is None else dict(social_bindings=social)
    return Request(c,bindings,exact,**extra)


def compact(audit):
    """Keep selected modeled actions and state digests, not duplicate memories."""
    out=copy.deepcopy(audit)
    for traces in out.values():
        for t in traces:
            for o in t.get('modeled_observations',[]):
                r=o['checkpoint'];actor=r.get('actor',{})
                o['checkpoint']=dict(sha256=digest(r),executed=r.get('executed'),
                    learned_entries=len(actor.get('memory',{}).get('entries',[])),
                    social_events=len(actor.get('social',{}).get('state',{}).get('events',[])))
    return out


def play(case,profile,seed,arm,*,horizon=2):
    if arm not in ('legacy','returns','owned'):raise ValueError('declared comparison arm required')
    b=make_bridge(profile,case);w=b.start();c=b.observe(b.project(w))
    p=Policy(principle_priority='personal',commitments={character_key(c):.5})
    actor=DecisionLoop(c,policy=p);trace=[];learn=arm=='owned' and case['game'] in TRAPS
    while not b.terminal(w):
        projected=b.project(w);c=b.observe(projected,actor.state);req=request_for(c,case['game'],learn=learn);captured={}
        def planner(cs,ds):
            observer=None;saved=digest(actor.record())
            if arm=='owned':
                from .actor_model import ActorModel
                from .model_observer import ModelFeedback
                # Capture selected transition feedback once, without another RNG
                # draw or peeking at a counterfactual unchosen result.
                transitions={}
                def advance(before,key,model_seed):
                    after,row=b.modeled_advance(before,key,model_seed)
                    transitions[(digest(b.record(before)),key,digest(b.record(after)))]=copy.deepcopy(row)
                    return after,row
                def revealed(before,after,key,cc):
                    row=transitions[(digest(b.record(before)),key,digest(b.record(after)))]
                    return ModelFeedback(None,observed=vector(row) if learn and key!='CHECK' else None)
                observer=ActorModel(actor,root_request=req,
                    request_for=lambda cc:request_for(cc,case['game'],learn=learn),feedback=revealed)
            else:advance=b.modeled_advance
            paths,audit=rollout(cs[0],projected,observe=b.observe,advance=advance,terminal=b.terminal,
                assess=lambda s:b.goal(s).record(),horizon=horizon,seeds=b.seeds,policy=p,observer=observer)
            f=goal_forecast(cs[0],paths,audit,horizon=horizon,unit=b.unit,target=b.target,policy=p,settlement_weight='absolute')
            assert digest(actor.record())==saved,'hypothetical experience reached real actor'
            captured.update(audit=compact(audit),forecast=f)
            return f
        result=DecisionLoop.decide_batch([(actor,req)],False,planner)[0]
        key=result['decision']['action_id'];before=b.record(w);w,row=b.actual(w,key,seed)
        if learn and key!='CHECK':update=actor.observe(result['ticket'],vector(row))
        else:actor.abandon(result['ticket']);update=None
        trace.append(dict(before=before,after=b.record(w),root=key,flow=row,goal=b.goal(w).record(),
            decision=result['decision'],learning=result['learning'],update=update,
            modeled=captured['audit'][key],owner_record_sha256=digest(actor.record())))
        if len(trace)>1000:raise AssertionError('whole episode failed to settle')
    # Replay the actual driver and ledger independently of the decision callback.
    check=b.start()
    for t in trace:
        assert b.record(check)==t['before'];check,flow=b.actual(check,t['root'],seed)
        assert b.record(check)==t['after'] and flow==t['flow'] and b.goal(check).record()==t['goal']
    assert b.terminal(check)
    return dict(case=case,profile=profile['id'],seed=seed,arm=arm,horizon=horizon,goal=b.goal(w).record(),steps=len(trace),
        actions=[t['root'] for t in trace],learned_entries=len(actor.memory.entries),trace=trace,action_replay_verified=True)


def life(seed,domain,layout,arm,*,turns=36,horizon=2,peer_model_factory=None,world_setup=None,objective_gain=False,opportunity_regret=False,forecast_cache=None):
    if type(objective_gain) is not bool or type(opportunity_regret) is not bool:raise ValueError('explicit valuation modes required')
    if forecast_cache is not None and not isinstance(forecast_cache,dict):raise ValueError('explicit paired-experiment cache required')
    if arm not in ('reflex','returns','owned'):raise ValueError('declared life comparison required')
    from .encounters import EncounterWorld
    from .social_withdrawal import WithdrawalSocialMemory
    from .purpose_plan import Goal
    from .public_perception import with_public
    world=EncounterWorld(seed,domain,layout,'public');p=Policy(principle_priority='finite')
    if world_setup is not None:world_setup(world)
    actors={};trace=[];peer_models={}
    def contexts(w,states=None):
        cs,sb=w.contexts()
        for c in cs:
            # Actor counter/ledger identity is independent of the driver's RNG.
            c['seed']=42;c['scope']['episode']=domain+'-'+layout+'-owned'
            if states is not None:c['state']=copy.deepcopy(states[c['scope']['npc']])
        return cs,sb
    cs,_=contexts(world)
    for c in cs:
        own=c['scope']['npc'];actors[own]=DecisionLoop(c,policy=p,social=WithdrawalSocialMemory(c))
        if peer_model_factory is not None:peer_models[own]=peer_model_factory(c)
    def peer_public(cc):
        # Public bodily resources/history are legitimate. The current peer's
        # real traits, values and private DecisionLoop state are not supplied.
        cc=copy.deepcopy(cc);cc['personality']={k:.5 for k in cc['personality']};cc['values']={k:0. for k in cc['values']}
        return cc
    def goal(w,own):
        dead=w.health[own]<=0
        return Goal('resource-and-survival','failure' if dead else 'scored' if w.tick>=turns else 'running',
            -1. if dead else max(-1.,min(1.,w.score[own]/turns)))
    for tick in range(turns):
        cs,bindings=contexts(world);items=[];branch_reports={}
        for c,sb in zip(cs,bindings):items.append((actors[c['scope']['npc']],request_for(c,domain,social=sb)))
        results=[]
        for (actor,req) in items:
            own=req.context['scope']['npc']
            def personal_planner(base,decisions):
                peer_saved=None if peer_model_factory is None else digest(peer_models[own].record())
                projected=copy.deepcopy(world)
                for a in projected.actors:
                    if a!=own:projected.profiles[a]=dict(id='neutral',traits=(.5,)*5,values={})
                if peer_model_factory is not None:projected._peer_model=copy.deepcopy(peer_models[own])
                def observe(q,state):
                    cc=next(c for c in contexts(q)[0] if c['scope']['npc']==own);cc['state']=copy.deepcopy(state)
                    return with_public(cc,dict(tick=q.tick,public_history={a:list(q.history[a]) for a in q.actors},
                        public_health=q.health,public_energy=q.energy,public_stock=q.stock))
                def advance(q,key,model_seed):
                    after=copy.deepcopy(q);model_cs,_=contexts(after)
                    selected={c['scope']['npc']:p.choose(c,False)['action_id'] for c in model_cs};selected[own]=key
                    if peer_model_factory is not None:
                        import numpy as np
                        for j,cc in enumerate(model_cs):
                            peer=cc['scope']['npc']
                            if peer==own:continue
                            probs,_=after._peer_model.forecast(peer,peer_public(cc));keys=sorted(probs)
                            generator=np.random.default_rng(np.random.SeedSequence([model_seed,after.tick,j,731]))
                            selected[peer]=keys[int(generator.choice(len(keys),p=[probs[k] for k in keys]))]
                    after.seed=model_seed;rows,events=after.resolve(selected);after.seed=seed
                    i=after.actors.index(own);r=rows[i]
                    row=after.perceived(r['kind'],r['other_kind'],r['matched'],1.)
                    if r['luck_failure']:row['values']={};row['objective']=-.1
                    social=tuple(events[i])
                    after._selected_feedback=(key,social,copy.deepcopy(row))
                    return after,row
                def request_future(cc):
                    # Binding meanings are reconstructed from the PUBLIC root
                    # action kind/recipient; hidden other-person values absent.
                    from .social import SocialBinding
                    public=json.loads(''.join(cc['facts'][k] for k in sorted(cc['facts']) if k.startswith('perception_')))
                    health=public['public_health']
                    social={a['id']:SocialBinding(a['target'],1. if a['id'].startswith('cooperate:') else -1.,max(0.,1-health[a['target']]/8))
                        for a in cc['actions'] if a['target'] is not None}
                    return request_for(cc,domain,social=social)
                observer=None
                if arm=='owned':
                    from .actor_model import ActorModel
                    from .model_observer import ModelFeedback
                    def feedback(before,after,key,cc):
                        selected,social,row=after._selected_feedback
                        assert selected==key
                        return ModelFeedback(None,social_events=social)
                    observer=ActorModel(actor,root_request=req,request_for=request_future,feedback=feedback)
                cached=None;cache_key=None
                if forecast_cache is not None:
                    # The complete model inputs, including owned memory and
                    # peer posterior, are equal before sharing scenario work.
                    # Valuation mode changes only the scoring AFTER rollout.
                    public=dict(vars(projected));public['history']={a:list(h) for a,h in projected.history.items()}
                    if '_peer_model' in public:public['_peer_model']=public['_peer_model'].record()
                    cache_key=digest(['life-paired-rollout-v1',public,actor.record(),base[0],arm,turns,horizon])
                    cached=forecast_cache.get(cache_key)
                if cached is None:
                    paths,audit=rollout(base[0],projected,observe=observe,advance=advance,
                        terminal=lambda q:q.tick>=turns or q.health[own]<=0,horizon=horizon,seeds=(401,402),
                        assess=lambda q:goal(q,own).record(),observer=observer,policy=p)
                    compact_audit=compact(audit)
                    if forecast_cache is not None:
                        scoring_audit={k:[{key:copy.deepcopy(row[key]) for key in ('assessment','actions','terminal')} for row in rows] for k,rows in audit.items()}
                        if len(forecast_cache)>=512:forecast_cache.clear()
                        forecast_cache[cache_key]=copy.deepcopy((paths,scoring_audit,compact_audit))
                else:paths,audit,compact_audit=copy.deepcopy(cached)
                if peer_model_factory is not None:assert digest(peer_models[own].record())==peer_saved
                branch_reports[own]=compact_audit
                valuation={} if not (objective_gain or opportunity_regret) else dict(
                    objective_origin=goal(projected,own) if objective_gain else None,opportunity_regret=opportunity_regret,
                    competence_support=actor._prepare(req)['progress_allowed'])
                return goal_forecast(base[0],paths,audit,horizon=horizon,unit='ticks',target='resource-and-survival',
                    policy=p,settlement_weight='absolute',**valuation)
            result=DecisionLoop.decide_batch([(actor,req)],False,None if arm=='reflex' or world.health[own]<=0 else personal_planner)[0]
            results.append(result)
        # No peer is advanced until all current intentions have been selected.
        chosen={a:r['decision']['action_id'] for a,r in zip(world.actors,results)}
        peer_updates=[]
        if peer_model_factory is not None:
            public_before={c['scope']['npc']:peer_public(c) for c in cs}
            peer_snapshots={a:digest(peer_models[a].record()) for a in world.actors}
        rows,events=world.resolve(chosen)
        for a,r,ev in zip(world.actors,results,events):actors[a].abandon(r['ticket'],social_events=tuple(ev))
        if peer_model_factory is not None:
            for own in world.actors:
                # Forecast callbacks cannot learn from their imaginary peers.
                assert digest(peer_models[own].record())==peer_snapshots[own]
                for peer in world.actors:
                    if peer!=own:
                        update=peer_models[own].observe(peer,public_before[peer],chosen[peer],f'{tick}-{peer}')
                        peer_updates.append(dict(owner=own,peer=peer,score=update['score'],active=update['active']))
        item=dict(tick=tick,rows=rows,chosen=chosen,
            owner_sha256={a:digest(actors[a].record()) for a in world.actors},
            modeled={a:branch_reports.get(a,{}).get(chosen[a],[]) for a in world.actors})
        if peer_model_factory is not None:item['peer_updates']=peer_updates
        trace.append(item)
    check=EncounterWorld(seed,domain,layout,'public')
    if world_setup is not None:world_setup(check)
    for t in trace:
        rows,events=check.resolve(t['chosen']);assert rows==t['rows']
    result=dict(seed=seed,domain=domain,layout=layout,arm=arm,turns=turns,
        actors=[dict(npc=a,profile=world.profiles[a]['id'],goal=goal(world,a).record(),**world.state(a)) for a in world.actors],
        trace=trace,action_replay_verified=True,
        limitation='neutral declared rival continuation, public histories, no counterpart private memory or true persona')
    if peer_model_factory is not None:result.update(peer_models={k:v.record() for k,v in peer_models.items()},
        limitation='finite declared peer profile hypotheses with public prequential scoring; no counterpart private memory or true persona; modeled posterior frozen between real observations')
    return result
