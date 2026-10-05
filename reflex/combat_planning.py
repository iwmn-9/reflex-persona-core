"""Public joint tactical rollouts for the generic horizon selector.

Finite modeled rival tactics, not the actual controller/intent/random stream.
Continuation tactics are purpose-directed options; root plans are evaluated by
the existing personality Policy. This is authored game evaluation, not a
learned universal value function or proof of optimal play.
"""
from dataclasses import dataclass,asdict
from itertools import product
import copy
import numpy as np
from .core import digest,Policy,compile_batch
from .examples import action
from .deliberation import JointForecast
from .horizon import purpose_return
from .combat import (alive,legal,preview,progress,potential,hit_probability,
    zone_distance,firing_distance,incoming,resolve,terminal,victory,transition_effect,battle_record)


@dataclass(frozen=True)
class TacticalControl:
    horizon: int=6
    samples: int=4
    max_plans: int=24
    max_regret: float=.15
    coordinate: bool=True
    recovery_margin: float=.02
    recovery_options: bool=False
    recovery_plans: int=48
    progress_weight: float=0.
    trace_execution: bool=False
    self_continuation: str='objective'
    root_completion: bool=False
    opponent_model: str='fixed'
    def __post_init__(self):
        for k,lo,hi in (('horizon',1,16),('samples',1,8),('max_plans',2,64)):
            v=getattr(self,k)
            if type(v) is not int or not lo<=v<=hi:raise ValueError('bounded '+k)
        if type(self.coordinate) is not bool or isinstance(self.max_regret,bool) or not np.isfinite(self.max_regret) or not 0<=self.max_regret<=2:
            raise ValueError('explicit coordination and bounded regret')
        if isinstance(self.recovery_margin,bool) or not isinstance(self.recovery_margin,(int,float)) or not np.isfinite(self.recovery_margin) or not 0<=self.recovery_margin<=2:
            raise ValueError('independent bounded recovery gain margin')
        if type(self.recovery_options) is not bool or type(self.recovery_plans) is not int or not 1<=self.recovery_plans<=128:
            raise ValueError('explicit bounded extra recovery options')
        if type(self.trace_execution) is not bool:raise ValueError('explicit execution tracing flag required')
        if self.self_continuation not in ('objective','persona-band'):
            raise ValueError('known experimental self continuation required')
        if type(self.root_completion) is not bool or self.root_completion and (not self.coordinate or self.self_continuation!='objective'):
            raise ValueError('root completion requires explicit coordination and objective continuation')
        if self.opponent_model not in ('fixed','uniform-coverage','goal-uniform'):
            raise ValueError('known bounded opponent model required')
        purpose_return([0.],self.progress_weight)


def destinations(choices):
    return [tuple(map(int,k.split(':')[1:])) for k in choices.values() if k.startswith('move:')]


def no_own_collision(choices):
    d=destinations(choices);return len(d)==len(set(d))


def routes(w,team):
    if w.goal in ('eliminate','secure'):return (w.goal,)
    if w.goal=='both':
        return tuple(r for r in ('eliminate','secure') if progress(w,team,r)<1)
    return ('eliminate','secure')


def tactical_scores(w,actor,route,aggressive=False):
    """Cheap declared rollout tactic, using public geometry and known rules.

    This proposes capability options. It is not used to overwrite personality
    weights, and its stationary own-action proxy is checked by joint rollouts.
    """
    u=w.units[actor];team=u.team
    if w.goal=='both' and w.secured[team]:route='eliminate'
    if not alive(w,1-team):route='secure'
    enemy_pos=tuple(w.units[j].pos for j in alive(w,1-team))
    before=zone_distance(w,u.pos) if route=='secure' else firing_distance(w.walls,u.pos,enemy_pos)
    out={}
    for k in legal(w,actor):
        if k.startswith('shoot:'):
            j=int(k.split(':')[1]);p=hit_probability(w,actor,j)
            score=p*(.20 if route=='eliminate' or aggressive else .10)
            score+=p*.15*(w.units[j].hp<=3)
        elif k.startswith('move:'):
            pos=tuple(map(int,k.split(':')[1:]));after=zone_distance(w,pos) if route=='secure' else firing_distance(w.walls,pos,enemy_pos)
            score=.06*(before-after)+.012*(pos in w.covers)
            if route=='secure' and pos[0]==4 and 1<=pos[1]<=3:score+=.04
        elif k.startswith('heal:'):
            j=int(k.split(':')[1]);score=.035*min(4,9-w.units[j].hp)/4+.12*(w.units[j].hp<=3)
        elif k=='reload':score=.075 if u.ammo==0 else .007
        else:score=.004+.10*(route=='secure' and u.pos[0]==4 and 1<=u.pos[1]<=3)
        out[k]=score
    return out


def tactical_joint(w,actors,route,order=None,aggressive=False,coordinate=True,forced=None,root_allowed=None):
    order=tuple(actors if order is None else order);chosen={};reserved=set()
    if forced:
        chosen.update(forced);reserved.update(destinations(forced))
    for i in order:
        if i in chosen:continue
        scores=tactical_scores(w,i,route.get(i,'eliminate') if isinstance(route,dict) else route,aggressive)
        keys=[k for k in scores if (root_allowed is None or k in root_allowed[i]) and (not coordinate or not k.startswith('move:') or tuple(map(int,k.split(':')[1:])) not in reserved)]
        if not keys:return None
        k=max(keys,key=lambda k:(scores[k],k));chosen[i]=k
        if k.startswith('move:'):reserved.add(tuple(map(int,k.split(':')[1:])))
    return chosen


def opponent_intents(w,team,aggressive=False):
    """Exact legacy public hypothesis; no actual rival controller is consulted."""
    enemies=alive(w,1-team);allowed=routes(w,1-team)
    theirs=tactical_joint(w,enemies,allowed[0] if allowed else 'secure',aggressive=aggressive,coordinate=True)
    if aggressive:
        for j in enemies:
            shots=[k for k in legal(w,j) if k.startswith('shoot:')]
            if shots:theirs[j]=max(shots,key=lambda k:(w.units[int(k.split(':')[1])].hp<=3,hit_probability(w,j,int(k.split(':')[1])),-w.units[int(k.split(':')[1])].hp,k))
    return theirs


def mission(w,team):
    if team in victory(w):return 1.
    if 1-team in victory(w) or not alive(w,team):return -1.
    if terminal(w):return 0. # unfinished at the real deadline is not a victory
    def task(t):
        k,c=progress(w,t,'eliminate'),progress(w,t,'secure')
        return {'eliminate':k,'secure':c,'either':max(k,c),'both':min(k,c)}[w.goal]
    hp=lambda t:sum(w.units[i].hp for i in alive(w,t))/27
    return float(np.clip(.7*task(team)-.35*task(1-team)+.15*(hp(team)-hp(1-team)),-1,1))


def propose(w,actors,contexts,reflex,control,root_allowed=None):
    team=w.units[actors[0]].team;plans=[];seen=set()
    def add(choices,route,label):
        if choices is None:return
        role={i:route[i] if isinstance(route,dict) else route for i in actors}
        key=tuple(choices[i] for i in actors)+tuple(role[i] for i in actors)
        if key in seen or control.coordinate and not no_own_collision(choices):return
        seen.add(key);plans.append((choices,role,label))
    current={i:d['action_id'] for i,d in zip(actors,reflex)}
    chosen=contexts[0]['facts']['chosen_route'];add(current,chosen,'reflex')
    allowed=routes(w,team)
    for r in allowed:
        for offset in range(len(actors)):
            order=actors[offset:]+actors[:offset]
            add(tactical_joint(w,actors,r,order,coordinate=control.coordinate),r,'purpose-option')
    # Support is a means, not an extra victory condition: a secure mission may
    # have a capturing actor and firing-support actors. The common core never
    # maps a personality to one of these game-specific roles.
    if 'secure' in allowed:
        for roles in product(('secure','eliminate'),repeat=len(actors)):
            role=dict(zip(actors,roles))
            add(tactical_joint(w,actors,role,coordinate=control.coordinate),role,'complementary-roles')
    # Include legal root variations with a conflict-aware completion. No root
    # is declared impossible merely because it is defensive or unusual.
    ranked=[]
    for i,c in zip(actors,contexts):
        b=compile_batch([c]);d=Policy().decide(b,False)
        for j,k in enumerate(b.ids[0]):ranked.append((float(d.scores[0,j]),i,k))
    for _,i,k in sorted(ranked,reverse=True):
        for r in allowed:
            add(tactical_joint(w,actors,r,coordinate=control.coordinate,forced={i:k}),r,'persona-root-option')
        if len(plans)>=control.max_plans:break
    plans=plans[:control.max_plans]
    if root_allowed is not None:
        permitted=dict(zip(actors,root_allowed));limit=len(plans)+control.recovery_plans
        # Complete each supported root using supported peers. Persona-first
        # combinations complement purpose tactics when their individual value
        # tiers disagree. They are evaluated, not automatically adopted.
        for r in allowed:
            for offset in range(len(actors)):
                add(tactical_joint(w,actors,r,actors[offset:]+actors[:offset],coordinate=control.coordinate,root_allowed=permitted),r,'supported-purpose-option')
        rankings={}
        for i,c in zip(actors,contexts):
            b=compile_batch([c]);d=Policy().decide(b,False)
            rankings[i]=sorted((k for k in b.ids[0] if k in permitted[i]),key=lambda k:(float(d.scores[0,b.ids[0].index(k)]),k),reverse=True)
        for combo in product(*(rankings[i][:3] for i in actors)):
            for r in allowed:
                add(dict(zip(actors,combo)),r,'supported-persona-combination')
            if len(plans)>=limit:break
        for i in actors:
            for k in rankings[i]:
                for r in allowed:
                    add(tactical_joint(w,actors,r,coordinate=control.coordinate,forced={i:k},root_allowed=permitted),r,'supported-root-completion')
                if len(plans)>=limit:break
            if len(plans)>=limit:break
        plans=plans[:limit]
    return plans


def forecast(w,actors,contexts,reflex,control=None,root_allowed=None):
    control=control or TacticalControl();actors=tuple(actors);team=w.units[actors[0]].team
    plans=propose(w,actors,contexts,reflex,control,root_allowed);roots={};purpose={};future=[copy.deepcopy(c) for c in contexts]
    for c in future:c['actions']=[];c['facts']['planning_target']='modeled joint horizon; not immediate empirical samples'
    nodes=0;labels={};self_model=None;opponent_models=None
    if control.opponent_model!='fixed':
        from .goal_rollout import GoalRollout
        opponent_models=tuple(GoalRollout(w,team,sample,control.opponent_model,control.samples) for sample in range(control.samples))
    if control.self_continuation=='persona-band':
        from .persona_continuation import PersonaContinuation
        self_model=PersonaContinuation(actors,contexts,reflex,control.coordinate)
    for n,(first,route,label) in enumerate(plans):
        key=f'plan-{n:03d}';roots[key]=tuple(first[i] for i in actors);labels[key]=dict(route=route,source=label)
        outcomes=[[] for _ in actors];scores=[];paths=[];execution_samples=[]
        for sample in range(control.samples):
            model=w;path=[];execution=[]
            states=self_model.start(first) if self_model is not None else None
            # Common private PLANNING seeds across plans, disjoint from actual
            # hit RNG. Tactic hypotheses remain coherent within each rollout.
            seed=int(digest(['combat-model',contexts[0]['scope']['episode'],w.tick,sample])[:16],16)
            aggressive=sample%2==1
            for depth in range(control.horizon):
                if terminal(model):break
                if depth==0:ours=first
                elif self_model is None:ours=tactical_joint(model,alive(model,team),route,coordinate=control.coordinate)
                else:ours,states=self_model.choose(model,team,route,states)
                theirs=(opponent_intents(model,team,aggressive) if opponent_models is None
                        else opponent_models[sample].choose(model,depth))
                before=model
                model,_=resolve(model,{**ours,**theirs},seed);nodes+=1
                if control.trace_execution:
                    execution.append(dict(index=before.tick,before=digest(battle_record(before)),after=digest(battle_record(model)),
                        self={str(i):k for i,k in ours.items()},opponent={str(i):k for i,k in theirs.items()},terminal=terminal(model)))
                path.append(mission(model,team))
            # A terminal result persists through the declared equal horizon;
            # otherwise early terminal wins would be averaged on fewer steps.
            path.extend([mission(model,team)]*(control.horizon-len(path)))
            value=purpose_return(path,control.progress_weight);scores.append(value);paths.append(path)
            if control.trace_execution:execution_samples.append(execution)
            for a,i in enumerate(actors):
                e=transition_effect(w,model,i,first[i],contexts[a]['facts']['chosen_route'],survival_security=True,p=1/control.samples)
                e['objective']=value;e['values']['achievement']=value
                outcomes[a].append(e)
        purpose[key]=float(np.mean(scores))
        labels[key]['purpose_path_mean']=np.mean(paths,axis=0).tolist()
        if control.trace_execution:labels[key]['execution_samples']=execution_samples
        for a,c in enumerate(future):c['actions'].append(action(key,*outcomes[a],confidence=.75))
    return JointForecast(tuple(future),roots,purpose,control.horizon,control.max_regret,
        dict(config=asdict(control),nodes=nodes,plans=labels,
             **({} if self_model is None else dict(self_model=dict(scorings=self_model.scorings,batches=self_model.batches,cache_hits=self_model.cache_hits,choices=self_model.choices,coordination_conflicts=self_model.conflicts))),
             **({} if opponent_models is None else dict(opponent_samples=[m.record() for m in opponent_models])),
             model=('public goal-directed/aggressive hypotheses; ' if opponent_models is None else 'experimental '+control.opponent_model+' samples; ')+control.self_continuation+' continuation; not actual controller'),
        target='combat-mission-path-'+str(control.progress_weight))
