"""Authored finite bankroll experiment, not financial advice or a real casino.

Known odds, no borrowing, no hidden draws in contexts. Actual draws and actor
tie randomness have separate deterministic namespaces. Waiting has no cost.
"""
from dataclasses import dataclass,replace,asdict
import numpy as np
from .core import NEEDS,TRAITS,digest
from .examples import context,action,effect

SCENARIOS=('opportunity','unfavorable','change')
TARGET=18


@dataclass(frozen=True)
class Table:
    cash: int=12
    tick: int=0
    limit: int=16


def offers(w,scenario):
    if scenario not in SCENARIOS:raise ValueError('known gambling scenario required')
    if scenario=='unfavorable' or scenario=='change' and w.tick>=8:
        return {'safe':(1,1,.4),'swing':(2,4,.2),'trap':(2,6,.15)}
    return {'safe':(1,1,.8),'swing':(2,4,.55),'trap':(2,6,.15)}


def legal(w,scenario):
    return ('pass',)+tuple(k for k,(stake,_,_) in offers(w,scenario).items() if stake<=w.cash)


def expected(w,scenario,key):
    if key=='pass':return 0.
    stake,profit,p=offers(w,scenario)[key];return p*profit-(1-p)*stake


def terminal(w):return w.tick>=w.limit or w.cash==0 or w.cash>=TARGET


def resolve(w,scenario,key,seed):
    if terminal(w) or key not in legal(w,scenario):raise ValueError('legal live bet required')
    delta=0
    if key!='pass':
        stake,profit,p=offers(w,scenario)[key]
        # One sealed round draw, independent of chosen action/controller.
        draw=np.random.default_rng(int(digest(['bankroll-actual',seed,w.tick])[:16],16)).random()
        delta=profit if draw<p else -stake
    return replace(w,cash=w.cash+delta,tick=w.tick+1),delta


def make_context(w,scenario,profile,seed,memory=None):
    choices=[];safety=lambda cash:max(0.,(6-cash)/6)
    for key in legal(w,scenario):
        outcomes=[]
        samples=((0,1.),) if key=='pass' else ((offers(w,scenario)[key][1],offers(w,scenario)[key][2]),(-offers(w,scenario)[key][0],1-offers(w,scenario)[key][2]))
        for delta,p in samples:
            advance=float(np.clip(delta/6,-1,1));safe=safety(w.cash)-safety(w.cash+delta)
            outcomes.append(effect(delta/12,needs={'growth':advance,'safety':safe},
                values={'achievement':advance,'power':advance,'security':safe},
                style={'openness':.4 if key=='swing' else -.2 if key=='safe' else 0},p=p))
        choices.append(action(key,*outcomes))
    c=context('bankroll',choices,{'growth':.35,'safety':safety(w.cash)+.1},profile['values'],dict(zip(TRAITS,profile['traits'])),mode=None)
    for n in NEEDS:
        if n not in ('growth','safety'):c['needs'][n]=dict(supported=False,enabled=False,deficit=None)
    c.update(scope=dict(game='bankroll-v1',episode=f'{scenario}-{seed}',npc='bettor'),seed=seed,tick=w.tick,
        objective='有限の機会で資金目標へ進み、破産を避ける',
        facts=dict(cash=str(w.cash),goal=str(TARGET),odds=str(offers(w,scenario)),
            rounds_left=str(w.limit-w.tick),forecast='known current odds; future odds/draws absent; pass costs zero'))
    if memory is not None:c['state']=memory
    return c


def record(w):return asdict(w)
