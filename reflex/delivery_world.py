"""Small public logistics rules: supplies, cargo, energy and two completion routes.

This new test environment provides mechanics and semantic effects, never an
action recommendation. No delivery rule is imported by the persona core.
"""
from dataclasses import dataclass,replace,asdict
import copy
from .core import TRAITS,digest
from .examples import context,action,effect
from .purpose_plan import Goal
from .progress import Activity,PurposeRequest,PurposeFeedback


@dataclass(frozen=True)
class Depot:
    tick: int=0
    limit: int=9
    energy: int=6
    stock: int=0
    cargo: int=0
    delivered: int=0
    community: int=0
    capacity: int=3
    supply: int=2


def terminal(w):return w.tick>=w.limit or w.delivered>=6 or w.community>=4


def legal(w):
    if terminal(w):return ()
    keys=['rest']
    if w.energy>=1:
        keys.append('collect')
        if w.stock>0 and w.cargo<w.capacity:keys.append('load')
    if w.cargo>0 and w.energy>=2:keys.extend(('fulfil','aid'))
    return tuple(keys)


def step(w,key):
    if key not in legal(w):raise ValueError('legal live logistics action required')
    changes=dict(tick=w.tick+1)
    if key=='rest':changes['energy']=min(9,w.energy+3)
    elif key=='collect':changes.update(energy=w.energy-1,stock=w.stock+w.supply)
    elif key=='load':
        n=min(w.stock,w.capacity-w.cargo);changes.update(energy=w.energy-1,stock=w.stock-n,cargo=w.cargo+n)
    else:
        changes.update(energy=w.energy-2,cargo=0)
        changes['delivered' if key=='fulfil' else 'community']=getattr(w,'delivered' if key=='fulfil' else 'community')+w.cargo
    return replace(w,**changes)


def goal(w):
    if w.delivered>=6 or w.community>=4:return Goal('contract-or-community','success',1.)
    if terminal(w):return Goal('contract-or-community','draw',0.)
    return Goal('contract-or-community','running',.5*max(w.delivered/6,w.community/4))


def consequence(before,after):
    g=(goal(after).value-goal(before).value)/2
    health=(after.energy-before.energy)/9
    return effect(g,needs={'physiology':health,'safety':health/2,'growth':g},
        values={'achievement':g,'power':(after.delivered-before.delivered)/6,
            'benevolence':(after.community-before.community)/4},cost=.01)


class DeliveryProbe:
    genre='delivery';seeds=(0,);route=None;target='contract-or-community'
    def __init__(self,spec,profile):
        self.spec=copy.deepcopy(spec);self.profile=copy.deepcopy(profile)
        self.name=digest([asdict(self.start()),profile['id']])[:16]
    def start(self):return Depot(limit=self.spec['limit'],capacity=self.spec.get('capacity',3),supply=self.spec.get('supply',2))
    terminal=staticmethod(terminal);keys=staticmethod(legal);goal=staticmethod(goal)
    def advance(self,w,key,seed):
        after=step(w,key);return after,consequence(w,after)
    actual=advance
    def observe(self,w,memory=None):
        choices=[action(k,self.advance(w,k,0)[1]) for k in legal(w)]
        c=context(self.name,choices,{'physiology':1-w.energy/9,'safety':1-w.energy/9,'growth':.3},
            self.profile['values'],dict(zip(TRAITS,self.profile['traits'])),mode=None)
        c['scope']['game']='logistics';c['tick']=w.tick
        if memory is not None:c['state']=copy.deepcopy(memory)
        c['objective']='本人の契約または共同体への納品条件を期限までに達成する'
        c['facts']['rules']='public energy, material and cargo; contract 6 OR community 4; exact transitions'
        return c
    def purpose(self,w,c,previous=None):
        activities={}
        for k in legal(w):
            after=step(w,k)
            if k=='rest':
                activities[k]=Activity('wait' if w.energy<9 else 'idle','recover-energy','energy-band',3,
                    'energy-storage-full' if w.energy<9 else '')
            else:activities[k]=Activity('attempt','observable-resource-or-delivery-change',digest([asdict(w),k])[:16])
        return PurposeRequest(goal(w).value,min(1.,(w.stock+w.cargo)/12),activities)
    def feedback(self,before,after,key):
        prepared=after.stock+after.cargo>before.stock+before.cargo or after.cargo>before.cargo or after.energy>before.energy
        return PurposeFeedback(goal(after).value,min(1.,(after.stock+after.cargo)/12),
            maintained=prepared,completed=goal(after).status=='success')
