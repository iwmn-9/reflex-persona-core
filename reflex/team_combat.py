"""Heterogeneous signed allies over the unchanged public combat rule engine."""
import copy
from dataclasses import replace
import numpy as np
from .core import Policy,compile_batch,digest
from .combat import (Battle,alive,legal,make_context,route_context,resolve,terminal,
    transition_effect,battle_record)
from .combat_planning import (TacticalControl,propose,tactical_joint,opponent_intents,
    mission,routes)
from .examples import action
from .deliberation import JointForecast


def opening(scenario,team):
    if scenario in ('open','choke'):return Battle.start(scenario,'either',limit=24)
    if scenario!='rescue':raise ValueError('known team combat scenario')
    w=Battle.start('cover','secure',limit=20);units=list(w.units)
    for t in (0,1):
        for index,i in enumerate(alive(w,t)):
            units[i]=replace(units[i],x=3 if t==0 else 5,y=index+1,
                hp=(9,3,6)[index] if t==team else (9,6,9)[index],ammo=0 if t==team and index==1 else 3)
    return replace(w,units=tuple(units),walls=(),covers=((3,1),(5,3)))


def make_person(w,actor,profile,seed,state=None):
    rc=route_context(w,actor,profile,seed)
    route=Policy(principle_priority='finite').choose(rc,False)['action_id']
    return make_context(w,actor,profile,seed,route,memory=state,survival_security=True)


def forecast(w,actors,contexts,decisions,horizon=6,samples=4,max_plans=24,partner_model='cooperative'):
    team=w.units[actors[0]].team
    if any(w.units[i].team!=team for i in actors):raise ValueError('one signed team required')
    if partner_model not in ('cooperative','uncertain'):raise ValueError('declared unsigned partner model')
    unsigned=tuple(i for i in alive(w,team) if i not in actors)
    control=TacticalControl(horizon=horizon,samples=samples,max_plans=max_plans)
    plans=propose(w,actors,contexts,decisions,control)
    # Preserve the actual uncoordinated proposal as a comparison, even if its
    # movement destinations conflict. Failing actions remain legal rule events.
    current={i:d['action_id'] for i,d in zip(actors,decisions)}
    role={i:c['facts']['chosen_route'] for i,c in zip(actors,contexts)}
    if not any(first==current for first,_,_ in plans):plans.insert(0,(current,role,'independent'))
    plans=plans[:max_plans];roots={};purpose={};future=[copy.deepcopy(c) for c in contexts]
    for c in future:c['actions']=[]
    nodes=0
    for index,(first,roles,label) in enumerate(plans):
        key=f'plan-{index:03d}';roots[key]=tuple(first[i] for i in actors)
        outcomes=[[] for _ in actors];scores=[]
        for sample in range(samples):
            model=w;model_seed=int(digest(['signed-combat-model',contexts[0]['scope']['episode'],w.tick,sample])[:16],16)
            for depth in range(horizon):
                if terminal(model):break
                outsiders={}
                if partner_model=='uncertain':
                    for i in unsigned:
                        if i not in alive(model,team):continue
                        outsiders[i]=('guard' if sample//2%2==0 else
                            tactical_joint(model,(i,),roles,coordinate=False)[i])
                forced={**outsiders,**(first if depth==0 else {})}
                ours=tactical_joint(model,alive(model,team),roles,forced=forced,coordinate=True)
                theirs=opponent_intents(model,team,aggressive=sample%2==1)
                model,_=resolve(model,{**ours,**theirs},model_seed);nodes+=1
            value=mission(model,team);scores.append(value)
            for a,i in enumerate(actors):
                e=transition_effect(w,model,i,first[i],contexts[a]['facts']['chosen_route'],survival_security=True,p=1/samples)
                e['objective']=value;e['values']['achievement']=value;outcomes[a].append(e)
        purpose[key]=float(np.mean(scores))
        for a,c in enumerate(future):c['actions'].append(action(key,*outcomes[a],confidence=.75))
    return JointForecast(tuple(future),roots,purpose,horizon,.15,
        dict(nodes=nodes,plans=len(roots),partner_model=partner_model,unsigned=list(unsigned),
             model='declared public tactical continuation; unsigned allies are hypothesized, not commanded; private future hit stream excluded'),
        target='signed-combat-purpose')
