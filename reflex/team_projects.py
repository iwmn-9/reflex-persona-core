"""Authored simultaneous team production race, not a commercial game clone.

Public stock, individual energy/skill/prestige and a shared construction goal.
All builds reserve pre-turn materials; excess requests all fail, without actor
priority. An adjacent-game counterpart to tactical movement/healing conflicts.
"""
import copy
from dataclasses import dataclass,replace,asdict
from itertools import product
import numpy as np
from .core import Policy,TRAITS,NEEDS,digest
from .examples import context,action,effect
from .deliberation import JointForecast


@dataclass(frozen=True)
class Worker:
    team:int
    energy:int=6
    skill:int=0
    prestige:float=0.


@dataclass(frozen=True)
class Workshop:
    people:tuple
    stock:tuple=(2,2)
    points:tuple=(0,0)
    tick:int=0
    limit:int=14
    quota:int=24
    scarce:bool=False
    shock:bool=False

    @classmethod
    def start(cls,scenario='balanced',members=3,limit=14):
        if scenario not in ('balanced','exhausted','shock'):raise ValueError('known project scenario required')
        if type(members) is not int or not 2<=members<=4:raise ValueError('two to four members per team')
        ps=tuple(Worker(t,energy=(6,2,4,5)[i] if scenario=='exhausted' else 6)
            for t in (0,1) for i in range(members))
        return cls(ps,limit=limit,quota=8*members,scarce=scenario=='exhausted',shock=scenario=='shock')


def members(w,team):return tuple(i for i,p in enumerate(w.people) if p.team==team)
def winners(w):return tuple(t for t in (0,1) if w.points[t]>=w.quota)
def terminal(w):return bool(winners(w)) or w.tick>=w.limit


def legal(w,actor):
    if terminal(w) or type(actor) is not int or not 0<=actor<len(w.people):return ()
    p=w.people[actor];names=['rest']
    if p.energy>=1:names.append('gather')
    if p.energy>=2 and w.stock[p.team]>=2:names.append('build')
    if p.energy>=2 and p.skill<2 and w.stock[p.team]>=2:names.append('train')
    if p.energy>=1:
        names.extend(f'aid:{i}' for i in members(w,p.team) if i!=actor and w.people[i].energy<=4)
    return tuple(sorted(names))


def resolve(w,choices):
    if set(choices)!=set(range(len(w.people))) or any(k not in legal(w,i) for i,k in choices.items()):
        raise ValueError('one pre-state legal intent for every project participant')
    stock=list(w.stock);points=list(w.points);ps=list(w.people);aids=[0]*len(ps);conflicts=[];overflow=0
    for team in (0,1):
        claimants=[i for i in members(w,team) if choices[i] in ('build','train')]
        failed=2*len(claimants)>w.stock[team]
        if failed:conflicts.extend(claimants)
        for i in members(w,team):
            p=ps[i];key=choices[i]
            if key=='rest':ps[i]=replace(p,energy=min(6,p.energy+3))
            elif key=='gather':
                stock[team]+=1 if w.scarce else 2;ps[i]=replace(p,energy=p.energy-1,prestige=p.prestige+.5)
            elif key in ('build','train'):
                if failed:continue
                stock[team]-=2
                if key=='build':
                    gain=3+p.skill;points[team]+=gain
                    ps[i]=replace(p,energy=p.energy-2,prestige=p.prestige+gain)
                else:ps[i]=replace(p,energy=p.energy-2,skill=p.skill+1)
            else:
                aids[int(key.split(':')[1])]+=2;ps[i]=replace(p,energy=p.energy-1)
        overflow+=max(0,stock[team]-8);stock[team]=min(8,stock[team])
    ps=tuple(replace(p,energy=min(6,p.energy+aids[i])) for i,p in enumerate(ps))
    if w.shock and w.tick+1==5:stock=[x//2 for x in stock]
    after=replace(w,people=ps,stock=tuple(stock),points=tuple(points),tick=w.tick+1)
    return after,dict(material_conflicts=conflicts,aid=aids,overflow=overflow)


def credit(w,team):
    ws=winners(w)
    return 1/len(ws) if team in ws else 0. if ws else .5


def purpose(w,team):
    if terminal(w):return 2*credit(w,team)-1
    other=1-team
    return float(np.clip((w.points[team]-w.points[other])/w.quota+
        .08*(w.stock[team]-w.stock[other])/8,-1,1))


def consequence(before,after,actor,key,p=1.):
    old=before.people[actor];new=after.people[actor];team=old.team
    value=purpose(after,team)-purpose(before,team)
    personal=(new.prestige-old.prestige)/5
    help=sum(after.people[i].energy-before.people[i].energy for i in members(before,team) if i!=actor)/12
    return effect(float(np.clip(value,-1,1)),
        needs={'physiology':(new.energy-old.energy)/6,'esteem':float(np.clip(personal,-1,1)),
               'growth':(new.skill-old.skill)/2},
        values={'achievement':float(np.clip(value,-1,1)),'power':float(np.clip(personal,-1,1)),
                'security':(new.energy-old.energy)/6,'benevolence':float(np.clip(help,-1,1))},
        style={'agreeableness':.3 if key.startswith('aid:') else 0.},cost=.01,p=p)


def greedy(w,team,forced=None):
    """Declared stock-aware capability tactic, no actual rival controller input."""
    chosen=dict(forced or {});reserved=2*sum(k in ('build','train') for k in chosen.values())
    for i in sorted(members(w,team),key=lambda i:(-w.people[i].skill,i)):
        if i in chosen:continue
        p=w.people[i];keys=legal(w,i)
        if 'build' in keys and reserved+2<=w.stock[team]:key='build';reserved+=2
        elif p.energy<=1:key='rest'
        elif w.stock[team]-reserved<2 and 'gather' in keys:key='gather'
        elif p.energy<4:key='rest'
        else:key='gather'
        chosen[i]=key
    return chosen


def make_context(w,actor,profile,seed,state=None):
    choices=[];team=w.people[actor].team
    for key in legal(w,actor):
        ours=greedy(w,team,{actor:key});theirs=greedy(w,1-team)
        after,_=resolve(w,{**ours,**theirs})
        choices.append(action(key,consequence(w,after,actor,key),confidence=.8))
    c=context(f'projects-{seed}',choices,
        needs={'physiology':1-w.people[actor].energy/6,'esteem':.5,'growth':.3},
        values=profile['values'],traits=dict(zip(TRAITS,profile['traits'])),mode=None)
    for n in NEEDS:
        if n not in ('physiology','esteem','growth'):c['needs'][n]=dict(supported=False,enabled=False,deficit=None)
    c.update(scope=dict(game='team-projects-v1',episode=f'projects-{seed}',npc=f'worker-{actor}'),
        seed=seed,tick=w.tick,objective='所属チームの建設目標と、本人の健康・成長・貢献',
        facts={'public_stock':str(w.stock),'public_points':str(w.points),
               'public_rules':str(dict(limit=w.limit,quota=w.quota,scarce=w.scarce,shock=w.shock)),
               'forecast':'public stock-aware declared tactic; no pending real intents'})
    for i,p in enumerate(w.people):c['facts'][f'public_worker_{i}']=str(asdict(p))
    if state is not None:c['state']=copy.deepcopy(state)
    return c


def outside_action(w,actor,kind):
    keys=legal(w,actor)
    if kind==0:return 'rest'
    if 'build' in keys:return 'build'
    return 'gather' if 'gather' in keys else 'rest'


def forecast(w,actors,contexts,decisions,horizon=6,max_plans=64,partner_model='cooperative',partner_weights=None):
    team=w.people[actors[0]].team
    if any(w.people[i].team!=team for i in actors):raise ValueError('one declared team required')
    bank=[];seen=set();current=tuple(d['action_id'] for d in decisions)
    def add(root):
        if root in seen:return
        seen.add(root);bank.append(root)
    add(current);add(tuple(greedy(w,team)[i] for i in actors))
    # Complete immediate reservation alternatives before ranking the remaining
    # bounded bank by public one-step consequence. Keep the incumbent explicitly.
    combinations=list(product(*(legal(w,i) for i in actors)))
    def rank(root):
        ours=greedy(w,team,dict(zip(actors,root)));after,audit=resolve(w,{**ours,**greedy(w,1-team)})
        return (-len(audit['material_conflicts']),purpose(after,team),sum(p.energy for p in after.people if p.team==team),root)
    for root in sorted(combinations,key=rank,reverse=True):
        add(root)
        if len(bank)>=max_plans:break
    if partner_model not in ('cooperative','uncertain'):raise ValueError('declared unsigned partner model')
    unsigned=tuple(i for i in members(w,team) if i not in actors)
    scenarios=2 if unsigned and partner_model=='uncertain' else 1
    weights=np.array((.5,.5) if partner_weights is None else partner_weights,dtype=float)
    if weights.shape!=(2,) or not np.isfinite(weights).all() or np.any(weights<0) or abs(weights.sum()-1)>1e-8:
        raise ValueError('finite normalized declared partner weights')
    future=[copy.deepcopy(c) for c in contexts]
    for c in future:c['actions']=[]
    roots={};scores={};nodes=0
    for index,root in enumerate(bank):
        key=f'plan-{index:03d}';roots[key]=root;outcomes=[[] for _ in actors];values=[]
        for scenario in range(scenarios):
            model=w
            for depth in range(horizon):
                if terminal(model):break
                outsiders={}
                if scenarios>1:
                    for i in unsigned:
                        outsiders[i]=outside_action(model,i,scenario)
                forced={**outsiders,**(dict(zip(actors,root)) if depth==0 else {})}
                ours=greedy(model,team,forced)
                model,_=resolve(model,{**ours,**greedy(model,1-team)});nodes+=1
            value=purpose(model,team);values.append(value)
            for j,actor in enumerate(actors):
                e=consequence(w,model,actor,root[j],p=float(weights[scenario]) if scenarios==2 else 1.);e['objective']=value;e['values']['achievement']=value
                outcomes[j].append(e)
        scores[key]=float(np.dot(values,weights)) if scenarios==2 and partner_weights is not None else float(np.mean(values))
        for j,c in enumerate(future):c['actions'].append(action(key,*outcomes[j],confidence=.8))
    return JointForecast(tuple(future),roots,scores,horizon,.15,
        dict(nodes=nodes,plans=len(roots),partner_model=partner_model,unsigned=list(unsigned),scenarios=scenarios,partner_weights=weights.tolist(),
             model='public stock-aware future tactic; not future personality replanning'),
        target='team-project-purpose')
