"""Authored headless 3v3 grid skirmish. No engine/render/LLM dependency.

Game-owned: movement, LOS, hit probability, damage, kits/ammo, capture and goal
predicates. Policy/routes/runtime remain unchanged. All combat state is public;
this is not fog of war, continuous physics, real-time aiming or a shipped game.
Forecasts assume stationary rivals and uniform target allocation for threats.
"""
from dataclasses import dataclass,replace,asdict
from collections import deque
from functools import lru_cache
import copy
import math
import numpy as np
from .core import NEEDS,TRAITS,Policy,digest,compile_batch
from .examples import context,action,effect
from .routes import choose_route,RouteState

GOALS=('eliminate','secure','either','both')
WIDTH,HEIGHT,HP,MAGAZINE,RANGE=9,5,9,3,4
DAMAGE=3
MAPS={
    'open':dict(walls=(),covers=()),
    'cover':dict(walls=((4,1),(4,3)),covers=((3,1),(3,3),(5,1),(5,3))),
    'choke':dict(walls=((4,0),(4,1),(4,3),(4,4)),covers=((3,2),(5,2))),
}


@dataclass(frozen=True)
class Unit:
    team:int
    x:int
    y:int
    hp:int=HP
    ammo:int=MAGAZINE
    kit:int=1
    guard:bool=False

    @property
    def pos(self): return (self.x,self.y)


@dataclass(frozen=True)
class Battle:
    units:tuple
    walls:tuple=()
    covers:tuple=()
    goal:str='either'
    hold:tuple=(0,0)
    secured:tuple=(False,False)
    tick:int=0
    limit:int=40

    @classmethod
    def start(cls,map_name='open',goal='either',limit=40):
        if map_name not in MAPS or goal not in GOALS or type(limit) is not int or not 1<=limit<=100:
            raise ValueError('known map/goal and bounded ticks required')
        units=tuple(Unit(t,1 if t==0 else 7,y) for t in (0,1) for y in (0,2,4))
        return cls(units,**MAPS[map_name],goal=goal,limit=limit)


def alive(w,team): return tuple(i for i,u in enumerate(w.units) if u.hp>0 and u.team==team)
def in_zone(pos): return pos[0]==4 and 1<=pos[1]<=3


def victory(w):
    winners=[]
    for t in (0,1):
        kill=not alive(w,1-t); capture=w.secured[t]
        achieved={'eliminate':kill,'secure':capture,'either':kill or capture,'both':kill and capture}[w.goal]
        if alive(w,t) and achieved: winners.append(t)
    return tuple(winners)


def terminal(w): return bool(victory(w)) or w.tick>=w.limit or not any(u.hp>0 for u in w.units)


def los(w,a,b):
    """Line between cell centers, touching a solid wall boundary blocks LOS."""
    for x,y in w.walls:
        low,high=0.,1.
        for start,end,center in ((a[0],b[0],x),(a[1],b[1],y)):
            delta=end-start
            if delta==0:
                if not center-.5<=start<=center+.5: low,high=1.,0.; break
            else:
                first=(center-.5-start)/delta; last=(center+.5-start)/delta
                low=max(low,min(first,last)); high=min(high,max(first,last))
        if low<=high: return False
    return True


def hit_probability(w,shooter,target):
    a,b=w.units[shooter],w.units[target]
    distance=abs(a.x-b.x)+abs(a.y-b.y)
    if a.hp<=0 or b.hp<=0 or a.team==b.team or a.ammo<=0 or distance>RANGE or not los(w,a.pos,b.pos): return 0.
    return float(max(.12,.9-.08*distance-.22*(b.pos in w.covers)-.2*b.guard))


def legal(w,actor):
    if terminal(w) or not 0<=actor<len(w.units) or w.units[actor].hp<=0: return ()
    u=w.units[actor]; occupied={v.pos for v in w.units if v.hp>0}
    result=['guard']
    for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)):
        x,y=u.x+dx,u.y+dy
        if 0<=x<WIDTH and 0<=y<HEIGHT and (x,y) not in w.walls and (x,y) not in occupied:
            result.append(f'move:{x}:{y}')
    if u.ammo<MAGAZINE: result.append('reload')
    for j,v in enumerate(w.units):
        if hit_probability(w,actor,j)>0: result.append(f'shoot:{j}')
        if u.kit and v.hp>0 and v.hp<HP and u.team==v.team and abs(u.x-v.x)+abs(u.y-v.y)<=1:
            result.append(f'heal:{j}')
    return tuple(sorted(result))


def capture_tick(w):
    counts=[sum(in_zone(w.units[i].pos) for i in alive(w,t)) for t in (0,1)]
    hold=tuple(min(3,w.hold[t]+1) if counts[t]>counts[1-t] and counts[t]>0 else 0 for t in (0,1))
    return replace(w,hold=hold,secured=tuple(w.secured[t] or hold[t]>=3 for t in (0,1)))


def resolve(w,choices,seed):
    """All intents validated on one pre-state; no actor sees pending choices.

    Move destinations contested by multiple units fail for all. Pre-occupied
    cells cannot be entered even when vacated. Heal/reload, then simultaneous
    volleys: a unit killed in the volley still fires its already committed shot.
    Targets are tracked to their new position; out-of-range shots consume ammo.
    """
    if terminal(w): raise ValueError('battle ended')
    required={i for i,u in enumerate(w.units) if u.hp>0}
    if set(choices)!=required or any(k not in legal(w,i) for i,k in choices.items()):
        raise ValueError('one legal pre-state intent per living unit required')
    units=[replace(u,guard=choices.get(i)=='guard') for i,u in enumerate(w.units)]
    destinations={i:tuple(map(int,k.split(':')[1:])) for i,k in choices.items() if k.startswith('move:')}
    collisions=0
    for i,pos in destinations.items():
        if sum(x==pos for x in destinations.values())==1: units[i]=replace(units[i],x=pos[0],y=pos[1])
        else: collisions+=1
    healing=np.zeros(len(units),dtype=int)
    for i,k in choices.items():
        if k=='reload': units[i]=replace(units[i],ammo=MAGAZINE)
        elif k.startswith('heal:'):
            healing[int(k.split(':')[1])]+=4; units[i]=replace(units[i],kit=units[i].kit-1)
    units=[replace(u,hp=min(HP,u.hp+int(healing[i]))) for i,u in enumerate(units)]
    phase=replace(w,units=tuple(units)); losses=np.zeros(len(units),dtype=int); shots=[]
    for i,k in choices.items():
        if not k.startswith('shoot:'): continue
        target=int(k.split(':')[1]); chance=hit_probability(phase,i,target)
        rng=np.random.default_rng(int(digest([seed,w.tick,i,'actual-hit'])[:16],16))
        hit=bool(rng.random()<chance); losses[target]+=DAMAGE*hit
        units[i]=replace(units[i],ammo=units[i].ammo-1)
        shots.append(dict(shooter=i,target=target,probability=chance,hit=hit))
    units=tuple(replace(u,hp=max(0,u.hp-int(losses[i]))) for i,u in enumerate(units))
    after=capture_tick(replace(w,units=units,tick=w.tick+1))
    return after,dict(shots=shots,collisions=collisions,healing=healing.tolist())


def preview(w,actor,key,hit=False):
    """Conditional single root only; other players stationary, no actual RNG."""
    if key not in legal(w,actor): raise ValueError('legal root required')
    units=list(w.units); units[actor]=replace(units[actor],guard=False); u=units[actor]
    if key=='guard': units[actor]=replace(u,guard=True)
    elif key=='reload': units[actor]=replace(u,ammo=MAGAZINE)
    elif key.startswith('move:'):
        x,y=map(int,key.split(':')[1:]); units[actor]=replace(u,x=x,y=y)
    elif key.startswith('heal:'):
        j=int(key.split(':')[1]); units[actor]=replace(u,kit=u.kit-1)
        units[j]=replace(units[j],hp=min(HP,units[j].hp+4))
    else:
        j=int(key.split(':')[1]); units[actor]=replace(u,ammo=u.ammo-1)
        units[j]=replace(units[j],hp=max(0,units[j].hp-DAMAGE*bool(hit)))
    return capture_tick(replace(w,units=tuple(units),tick=w.tick+1))


def zone_distance(w,pos):
    """Static walkable terrain shortest path; ignores moving units."""
    return terrain_zone_distance(w.walls,pos)


@lru_cache(maxsize=512)
def terrain_zone_distance(walls,pos):
    """Share immutable public terrain only; no personality/HP/occupancy cached."""
    queue=deque([(pos,0)]); seen={pos}
    while queue:
        (x,y),d=queue.popleft()
        if in_zone((x,y)): return d
        for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)):
            p=(x+dx,y+dy)
            if 0<=p[0]<WIDTH and 0<=p[1]<HEIGHT and p not in seen and p not in walls:
                seen.add(p); queue.append((p,d+1))
    return WIDTH*HEIGHT


@lru_cache(maxsize=4096)
def firing_distance(walls,pos,enemy_positions):
    """Static path to a legal firing cell; cache only public immutable geometry.

    No moving occupancy, future enemy motion, current ammo or actual intents.
    """
    view=Battle((),walls=walls); queue=deque([(pos,0)]); seen={pos}
    while queue:
        (x,y),d=queue.popleft()
        if any(abs(x-a)+abs(y-b)<=RANGE and los(view,(x,y),(a,b)) for a,b in enemy_positions): return d
        for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)):
            p=(x+dx,y+dy)
            if 0<=p[0]<WIDTH and 0<=p[1]<HEIGHT and p not in seen and p not in walls:
                seen.add(p); queue.append((p,d+1))
    return WIDTH*HEIGHT


def progress(w,team,route):
    if route=='eliminate':
        enemies=alive(w,1-team)
        if not enemies: return 1.
        damage=1-sum(w.units[i].hp for i in enemies)/(3*HP)
        positions=tuple(w.units[i].pos for i in enemies)
        access=sum(max(0,1-firing_distance(w.walls,w.units[i].pos,positions)/4) for i in alive(w,team))/3
        return .65*damage+.35*access
    if route!='secure': raise ValueError('known combat route required')
    if w.secured[team]: return 1.
    proximity=max((1-min(1,zone_distance(w,w.units[i].pos)/8) for i in alive(w,team)),default=0.)
    return .5*proximity+.5*w.hold[team]/3


def potential(w,team,route):
    if team in victory(w): return 1.
    if 1-team in victory(w) or not alive(w,team): return -1.
    k,c=progress(w,team,'eliminate'),progress(w,team,'secure')
    mission={'eliminate':k,'secure':c,'either':max(k,c),'both':min(k,c)}[w.goal]
    chosen=k if route=='eliminate' else c
    fitness=sum(w.units[i].hp/HP for i in alive(w,team))/3
    ammunition=sum(w.units[i].ammo/MAGAZINE for i in alive(w,team))/3
    return .6*(.6*chosen+.2*mission+.15*fitness+.05*ammunition)


def incoming(w,actor):
    """Public proxy: each armed enemy fires once, targets uniformly among visible allies.

    Does not assert this is the opponent's policy, nor use pending real intents.
    """
    team=w.units[actor].team; result=0.
    for j in alive(w,1-team):
        targets=[i for i in alive(w,team) if hit_probability(w,j,i)>0]
        if actor in targets: result+=DAMAGE*hit_probability(w,j,actor)/len(targets)
    return result


def exposure(w,actor,horizon=3):
    """Finite stationary threat proxy, NOT a calibrated survival probability.

    Guard protects one tick, not every future tick automatically. Remaining
    enemy HP weights later threat capacity, approximating elimination effort.
    No actual future intentions, damage seed or hidden controller is consulted.
    """
    if type(horizon) is not int or not 1<=horizon<=8:raise ValueError('bounded exposure horizon required')
    first=incoming(w,actor)
    units=list(w.units);units[actor]=replace(units[actor],guard=False)
    future=replace(w,units=tuple(units));later=0.;team=w.units[actor].team
    for j in alive(future,1-team):
        targets=[i for i in alive(future,team) if hit_probability(future,j,i)>0]
        if actor in targets:later+=DAMAGE*hit_probability(future,j,actor)/len(targets)*future.units[j].hp/HP
    return min(1.,(first+(horizon-1)*later)/HP)


def route_context(w,actor,profile,seed):
    team=w.units[actor].team; choices=[]
    for route in ('eliminate','secure'):
        allowed=(w.goal in ('either','both') or w.goal==route)
        if w.goal=='both': allowed=allowed and progress(w,team,route)<1
        remaining=1-progress(w,team,route)
        quality=1-.5*remaining
        values={'achievement':quality}
        if route=='eliminate': values.update(power=.6,security=-.2)
        else: values.update(conformity=.5,security=.35)
        choices.append(action(route,effect(quality,needs={'esteem':quality},values=values),legal=allowed))
    c=person_context(w,actor,profile,seed,choices,'combat-route')
    return c


def person_context(w,actor,profile,seed,choices,suffix='combat-action',memory=None):
    u=w.units[actor]; needs={'physiology':1-u.hp/HP,'safety':min(1,incoming(w,actor)/HP+.15),'esteem':.35}
    c=context(f'battle-{seed}',choices,needs,profile['values'],dict(zip(TRAITS,profile['traits'])),mode=None)
    for n in NEEDS:
        if n not in needs: c['needs'][n]=dict(supported=False,enabled=False,deficit=None)
    c.update(scope=dict(game=suffix+'-v1',episode=f'{w.goal}-{seed}',npc=f'unit-{actor}'),seed=seed,tick=w.tick,
        objective='所属チームの実際の勝利条件を達成し、自分と味方が生き残る',
        facts=dict(goal=w.goal,public_tick=str(w.tick),public_hold=str(w.hold),public_secured=str(w.secured),
            walls=str(w.walls),cover=str(w.covers),
            forecast='one own action, static firing-access proxy, last rival posture, uniform-target threat; no actual rival intents/RNG'))
    for i,v in enumerate(w.units): c['facts'][f'public_unit_{i}']=str(asdict(v))
    if memory is not None: c['state']=copy.deepcopy(memory)
    return c


def transition_effect(w,after,actor,key,route,goal_need=False,exposure_horizon=1,p=1.,baseline=None,survival_security=False):
    """Same one-turn feature definition for predictions and revealed transitions.

    Joint observed changes include ally/opponent actions; these are on-policy
    conditional experiences, not a causal estimate of this actor alone. Future
    exposure_horizon>1 is a proxy and must NOT enter immediate empirical memory.
    """
    old=w.units[actor];new=after.units[actor]
    before,threat=baseline if baseline is not None else (potential(w,old.team,route),incoming(w,actor))
    objective=float(np.clip(potential(after,old.team,route)-before,-1,1))
    self_health=(new.hp-old.hp)/HP
    ally_health=sum(after.units[i].hp-w.units[i].hp for i in alive(w,old.team) if i!=actor)/HP
    damage=sum(w.units[i].hp-after.units[i].hp for i in alive(w,1-old.team))/HP
    safe=float(np.clip((threat-incoming(after,actor))/HP,-1,1)) if exposure_horizon==1 else float(np.clip(exposure(w,actor,exposure_horizon)-exposure(after,actor,exposure_horizon),-1,1))
    ammo=(new.ammo-old.ammo)/MAGAZINE
    fulfillment=4*(progress(after,old.team,route)-progress(w,old.team,route)) if goal_need else damage+.08*max(0,ammo)
    if survival_security:safe=self_health
    return effect(objective,needs={'physiology':self_health,'safety':safe,'esteem':float(np.clip(fulfillment,-1,1))},
        values={'achievement':objective,'power':float(np.clip(damage,-1,1)),'benevolence':float(np.clip(ally_health,-1,1)),
            'security':float(new.hp>0)-1 if survival_security else float(np.clip(safe+self_health,-1,1))},
        style={'conscientiousness':.15*max(0,ammo)},
        cost=.08 if key.startswith('heal:') else .015 if key.startswith('shoot:') else .005,p=p)


def incoming_counts(before,posed,actor):
    """Four-point current-volley distribution under the declared threat model.

    Independent uniform visible-target allocation, each pre-turn armed enemy
    fires once. A foe killed by our committed shot still fires this volley.
    No real intents/RNG; movement/target choice/model error remain uncertain.
    """
    units=list(posed.units);team=before.units[actor].team
    for j in alive(before,1-team):units[j]=replace(units[j],hp=before.units[j].hp,ammo=before.units[j].ammo)
    phase=replace(posed,units=tuple(units));mass=[1.]
    for j in alive(before,1-team):
        # Target choice precedes our hidden simultaneous move. Evaluate the
        # shot at the new position, but never let a rival aim at a newly visible
        # actor by peeking at its committed current intent.
        visible=[i for i in alive(before,team) if hit_probability(before,j,i)>0]
        p=hit_probability(phase,j,actor)/len(visible) if actor in visible else 0.
        if not p:continue
        next_mass=[0.]*(len(mass)+1)
        for k,q in enumerate(mass):next_mass[k]+=q*(1-p);next_mass[k+1]+=q*p
        mass=next_mass
    return tuple((k,p) for k,p in enumerate(mass) if p>0)


def make_context(w,actor,profile,seed,route,memory=None,goal_need=False,exposure_horizon=1,survival_security=False):
    if route not in ('eliminate','secure'): raise ValueError('known route required')
    if type(goal_need) is not bool or type(exposure_horizon) is not int or not 1<=exposure_horizon<=8:raise ValueError('bounded explicit combat forecast controls required')
    if type(survival_security) is not bool or survival_security and exposure_horizon!=1:raise ValueError('survival metric requires immediate horizon')
    choices=[];team=w.units[actor].team;baseline=(potential(w,team,route),incoming(w,actor))
    for key in legal(w,actor):
        chance=hit_probability(w,actor,int(key.split(':')[1])) if key.startswith('shoot:') else 1.
        outcomes=[]
        for hit,p in ((True,chance),(False,1-chance)) if key.startswith('shoot:') else ((False,1.),):
            if p<=0:continue
            after=preview(w,actor,key,hit)
            if survival_security:
                for hits,q in incoming_counts(w,after,actor):
                    units=list(after.units);units[actor]=replace(units[actor],hp=max(0,units[actor].hp-DAMAGE*hits))
                    leaf=capture_tick(replace(after,units=tuple(units),hold=w.hold,secured=w.secured))
                    outcomes.append(transition_effect(w,leaf,actor,key,route,goal_need,exposure_horizon,p*q,baseline,True))
            else:outcomes.append(transition_effect(w,after,actor,key,route,goal_need,exposure_horizon,p,baseline))
        choices.append(action(key,*outcomes,confidence=.75))
    c=person_context(w,actor,profile,seed,choices,memory=memory)
    c['facts']['chosen_route']=route
    if survival_security:c['facts']['security_model']='security=one-turn survival loss; injury=safety/physiology; independent uniform-target volley, not actual intents'
    return c


def reference(w,actor,seed):
    """Game-authored objective comparator; no actor's true intentions in forecasts."""
    profile=dict(traits=(.5,)*5,values={})
    route='eliminate' if w.goal=='eliminate' or (w.goal=='both' and w.secured[w.units[actor].team]) else 'secure'
    c=make_context(w,actor,profile,seed,route)
    return max(c['actions'],key=lambda a:(sum(o['p']*(o['objective']+.15*o['needs']['safety']+.06*o['needs']['physiology']) for o in a['outcomes']),a['id']))['id']


def opponent(w,actor,seed,kind='reference'):
    """Actual test controllers. Never passed into focal contexts/predictions.

    raider and a mid-match switch stress the existing uniform-shot assumption;
    they are actual behavioral variation, not additional oracle information.
    """
    if kind not in ('reference','raider','switch'):raise ValueError('known test opponent required')
    if kind=='reference' or kind=='switch' and w.tick<8:return reference(w,actor,seed)
    options=legal(w,actor);shots=[k for k in options if k.startswith('shoot:')]
    if shots:
        return max(shots,key=lambda k:(w.units[int(k.split(':')[1])].hp<=DAMAGE,
            hit_probability(w,actor,int(k.split(':')[1])),-w.units[int(k.split(':')[1])].hp,k))
    if 'reload' in options and w.units[actor].ammo==0:return 'reload'
    team=w.units[actor].team;positions=tuple(w.units[j].pos for j in alive(w,1-team))
    return min(options,key=lambda k:(firing_distance(w.walls,preview(w,actor,k).units[actor].pos,positions),
        -sum(hit_probability(preview(w,actor,k),actor,j) for j in alive(w,1-team)),k))


def battle_record(w): return asdict(w)


def battle_from_record(record):
    return Battle(**{**record,'units':tuple(Unit(**u) for u in record['units']),
        'walls':tuple(tuple(x) for x in record['walls']),'covers':tuple(tuple(x) for x in record['covers']),
        'hold':tuple(record['hold']),'secured':tuple(record['secured'])})
