"""Measure a finite directive separately from game purpose and causal control.

Games declare opportunities and observed criteria. This monitor does not rank
actions, invent an RL reward, or infer that fulfilling a criterion was caused
by a command. Identical reissues retain the deadline; unresolved interruptions
and windows without an opportunity never become successful demonstrations.
"""
from dataclasses import dataclass
import copy
from .core import digest,identifier


@dataclass(frozen=True)
class Directive:
    key: str
    target: str
    span: int
    unit: str='owner-decisions'

    def __post_init__(self):
        for x in (self.key,self.target,self.unit):identifier(x)
        if type(self.span) is not int or not 1<=self.span<=128:raise ValueError('bounded execution window required')


class ExecutionWindow:
    def __init__(self,scope,directives):
        if not isinstance(directives,tuple) or not 1<=len(directives)<=16 or any(not isinstance(d,Directive) for d in directives) or len({d.key for d in directives})!=len(directives):raise ValueError('finite declared directives required')
        self.owner=digest(scope);self.directives={d.key:d for d in directives};self.active=None;self.last_tick=-1
        self.counts={d.key:{k:0 for k in ('success','fail','timeout','interrupted','unobserved')} for d in directives}

    def issue(self,scope,key,tick,*,opportunity):
        if digest(scope)!=self.owner or key not in self.directives:raise ValueError('owned declared directive required')
        if type(tick) is not int or tick<self.last_tick or not 0<=tick<2**63 or type(opportunity) is not bool:raise ValueError('observed clock and explicit opportunity required')
        if self.active is not None and tick<self.active['opened']:raise ValueError('directive clock cannot move backward')
        if self.active is not None and tick>self.active['deadline']:raise ValueError('observe the expired directive before issuing another')
        if not opportunity:return dict(issued=False,reissued=False,deadline=None if self.active is None else self.active['deadline'])
        if self.active is not None and self.active['key']==key:return dict(issued=True,reissued=True,deadline=self.active['deadline'])
        if tick+self.directives[key].span-1>=2**63:raise ValueError('execution deadline exceeds clock limit')
        if self.active is not None:self.counts[self.active['key']]['interrupted']+=1
        d=self.directives[key];self.active=dict(key=key,opened=tick,deadline=tick+d.span-1,opportunities=0)
        return dict(issued=True,reissued=False,deadline=self.active['deadline'])

    def observe(self,scope,tick,*,opportunity,met=False,failed=False):
        if digest(scope)!=self.owner or self.active is None:raise ValueError('owned active directive required')
        if type(tick) is not int or not self.last_tick<tick<2**63 or tick<self.active['opened']:raise ValueError('increasing actual observation required')
        if any(type(v) is not bool for v in (opportunity,met,failed)) or met and failed:raise ValueError('explicit noncontradictory criterion observations required')
        if (met or failed) and not opportunity:raise ValueError('criterion requires an observed opportunity')
        row=copy.deepcopy(self.active);within=tick<=row['deadline'];row['opportunities']+=int(opportunity and within);self.last_tick=tick
        status='timeout' if not within else 'success' if met else 'fail' if failed else 'timeout' if tick>=row['deadline'] else 'running'
        if status!='running':
            if not row['opportunities']:status='unobserved'
            self.counts[row['key']][status]+=1;self.active=None
        else:self.active=row
        return dict(key=row['key'],status=status,opportunities=row['opportunities'],deadline=row['deadline'],
            criterion_is_not_causal_control=True)

    def record(self):
        return dict(format='execution-window-v1',owner=self.owner,active=copy.deepcopy(self.active),last_tick=self.last_tick,counts=copy.deepcopy(self.counts),
            directives=[dict(key=d.key,target=d.target,span=d.span,unit=d.unit) for d in self.directives.values()])
