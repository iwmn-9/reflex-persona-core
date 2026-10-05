"""Same joint-horizon interface in a sequential multi-resource victory race.

Current public supplies persist in the model. Rivals take real modeled turns;
future event schedules and the actual rival controller are not supplied.
"""
from dataclasses import dataclass,asdict,replace
from functools import lru_cache
import copy
import numpy as np
from .core import TRAITS,Policy,digest
from .examples import action
from .planning import vector,from_vector
from .deliberation import JointForecast
from .horizon import purpose_return
from .resource_world import (Empire,step,terminal,winners,economic_move,consequence)
from .route_experiment import route_potential,routed_context


@dataclass(frozen=True)
class EconomyControl:
    horizon: int=4
    max_regret: float=.15
    progress_weight: float=0.
    def __post_init__(self):
        if type(self.horizon) is not int or not 1<=self.horizon<=16:raise ValueError('bounded own-round horizon')
        if isinstance(self.max_regret,bool) or not np.isfinite(self.max_regret) or not 0<=self.max_regret<=2:
            raise ValueError('bounded purpose regret')
        purpose_return([0.],self.progress_weight)


def task(w,seat,route):
    if seat in winners(w):return 1.
    if winners(w):return -1.
    if terminal(w):return 0.
    rival=max((route_potential(w,i,r) for i in range(len(w.empires)) if i!=seat for r in w.routes),default=0)
    return route_potential(w,seat,route)-.5*rival


def forecast(w,contexts,decisions,control=None):
    control=control or EconomyControl()
    if len(contexts)!=1:raise ValueError('resource planner groups one current player')
    c=contexts[0];seat=w.turn;route=c['facts']['chosen_route'];fc=copy.deepcopy(c);fc['actions']=[]
    fc['facts']['planning_target']='modeled own-round horizon with rival turns; current supplies persist; no future event schedule'
    roots={};purpose={};nodes=0;paths={}
    for n,a in enumerate(c['actions']):
        if not a['legal'] or a['known_failure']:continue
        key=f'plan-{n:03d}';roots[key]=(a['id'],);outcomes=[];scores=[]
        for rival_tactic in ('near','far'):
            model=w;own=0;path=[];state=copy.deepcopy(decisions[0]['next_state'] if decisions else c['state'])
            model_seed=int(digest(['economy-model',c['scope'],w.round,rival_tactic])[:16],16)%(2**63)
            while not terminal(model) and own<control.horizon:
                if model.turn==seat:
                    if own==0:
                        k=a['id'];state.update(intent_action=k,age=0)
                    else:k,state=policy_move(model,c,route,model_seed,state,own)
                    own+=1
                else:k=rival_move(model,rival_tactic)
                model=step(model,k);nodes+=1
                if model.turn==seat or terminal(model):path.append(task(model,seat,route))
                # Complete the intervening rivals' turns and production before
                # comparing equal OWN horizons; do not freeze opponents.
                if own==control.horizon:
                    while not terminal(model) and model.turn!=seat:
                        model=step(model,rival_move(model,rival_tactic));nodes+=1
                    if len(path)<own:path.append(task(model,seat,route))
            path.extend([task(model,seat,route)]*(control.horizon-len(path)))
            value=purpose_return(path,control.progress_weight);scores.append(value)
            paths.setdefault(key,[]).append(path)
            e=consequence(w,model,seat);v=np.clip(vector(e),-1,1);v[-1]=a['outcomes'][0]['cost']
            e=from_vector(v,.5);e['objective']=value;e['values']['achievement']=value
            outcomes.append(e)
        purpose[key]=float(np.mean(scores));fc['actions'].append(action(key,*outcomes,confidence=.75))
    return JointForecast((fc,),roots,purpose,control.horizon,control.max_regret,
        dict(config=asdict(control),nodes=nodes,horizon_unit='own rounds',
             model='fixed-personality continuation, public near/far tactical rival hypotheses; no actual controller or future events',
             purpose_paths={k:np.mean(v,axis=0).tolist() for k,v in paths.items()}),
        target='economy-route-path-'+str(control.progress_weight))


def rival_move(w,tactic):
    # The declared solo economic tactic uses other empires only as invasion
    # targets (land/army). Their stocks/buildings cannot affect its choice.
    # Preserve those interactions while sharing identical model computations.
    roster=tuple(e if i==w.turn else Empire(land=e.land,army=e.army) for i,e in enumerate(w.empires))
    return cached_rival_move(replace(w,empires=roster),1 if tactic=='near' else 4)


@lru_cache(maxsize=4096)
def cached_rival_move(w,horizon):return economic_move(w,horizon)


def policy_move(w,c,route,seed,state,tick):
    p=dict(traits=tuple(c['personality'][k] for k in TRAITS),values=c['values'])
    model,_=routed_context(w,p,seed,tick,state,route)
    d=Policy().choose(model)
    return d['action_id'],d['next_state']


def route_forecast(w,c,control):
    """Evaluate victory-route options BEFORE building their immediate targets.

    Changing a route after building action effects would corrupt empirical
    target meanings. This separate macro stage avoids that substitution.
    """
    seat=w.turn;fc=copy.deepcopy(c);fc['actions']=[];roots={};purpose={};nodes=0
    for n,a in enumerate(c['actions']):
        if not a['legal'] or a['known_failure']:continue
        route=a['id'];key=f'route-plan-{n:03d}';roots[key]=(route,);scores=[];rows=[]
        for tactic in ('near','far'):
            model=w;own=0;state=None
            model_seed=int(digest(['economy-route-model',c['scope'],w.round,tactic])[:16],16)%(2**63)
            while not terminal(model) and own<control.horizon:
                if model.turn==seat:k,state=policy_move(model,c,route,model_seed,state,own);own+=1
                else:k=rival_move(model,tactic)
                model=step(model,k);nodes+=1
                if own==control.horizon:
                    while not terminal(model) and model.turn!=seat:
                        model=step(model,rival_move(model,tactic));nodes+=1
            score=task(model,seat,route);scores.append(score)
            e=copy.deepcopy(a['outcomes'][0]);e['p']=.5;e['objective']=score
            e['needs']['growth']=max(0,score);e['values']['achievement']=score
            rows.append(e)
        purpose[key]=float(np.mean(scores));fc['actions'].append(action(key,*rows,confidence=.75))
    fc['facts']['planning_target']='public modeled victory-route horizon before immediate action construction'
    return JointForecast((fc,),roots,purpose,control.horizon,control.max_regret,
        dict(nodes=nodes,config=asdict(control),horizon_unit='own rounds',model='fixed-personality route continuation; near/far rival tactics; no future events'))
