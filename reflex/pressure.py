"""Bounded need frustration from observed lack of progress, not win-rate tuning."""
from dataclasses import dataclass,asdict
import copy
import math
from .core import NEEDS,digest,compile_batch


@dataclass(frozen=True)
class PressureConfig:
    grace: int=3
    rise: float=.1
    relief: float=.25
    limit: float=.7
    progress_floor: float=.005

    def __post_init__(self):
        if type(self.grace) is not int or not 0<=self.grace<=100:raise ValueError('bounded grace required')
        for k in ('rise','relief','limit','progress_floor'):
            v=getattr(self,k)
            if isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or not 0<v<=1:raise ValueError('bounded pressure controls required')


class NeedPressure:
    """Game labels observed progress per supported need; fixed axes stay intact.

    Only explicit channel observations advance that channel. Maintaining a
    useful state suppresses buildup; completion clears it. Missing observations
    are not failures. Pressure cannot invent support/effects for a disabled need.
    """
    def __init__(self,scope,config=None):
        self.scope=copy.deepcopy(scope);self.owner=digest(scope);self.config=config or PressureConfig()
        self.debt={k:0. for k in NEEDS};self.stalls={k:0 for k in NEEDS};self.last_tick=-1

    def observe(self,tick,progress,maintained=(),completed=()):
        if type(tick) is not int or tick<=self.last_tick or not 0<=tick<2**63:raise ValueError('increasing observed tick required')
        if not isinstance(progress,dict) or set(progress)-set(NEEDS):raise ValueError('observed need progress required')
        if set(maintained)-set(progress) or set(completed)-set(progress):raise ValueError('maintenance/completion needs observed evidence')
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not -1<=v<=1 for v in progress.values()):raise ValueError('bounded actual progress required')
        debt=self.debt.copy();stalls=self.stalls.copy();cfg=self.config
        for k,v in progress.items():
            if k in completed:debt[k]=0.;stalls[k]=0
            elif v>cfg.progress_floor:
                debt[k]=max(0.,debt[k]-cfg.relief);stalls[k]=0
            elif k in maintained:
                stalls[k]=0
            else:
                stalls[k]=min(1000000,stalls[k]+1)
                if stalls[k]>cfg.grace:debt[k]=min(cfg.limit,debt[k]+cfg.rise)
        self.debt=debt;self.stalls=stalls;self.last_tick=tick
        return dict(debt=debt.copy(),stalls=stalls.copy())

    def apply(self,context):
        if digest(context['scope'])!=self.owner:raise ValueError('pressure owner mismatch')
        return self._apply(context)

    def apply_route(self,context):
        """Explicit route phase of this actor, checked by DecisionLoop as well."""
        if any(context['scope'][k]!=self.scope[k] for k in ('episode','npc')):raise ValueError('route pressure owner mismatch')
        return self._apply(context)

    def _apply(self,context):
        c=copy.deepcopy(context)
        for k,n in c['needs'].items():
            if n['supported'] and n['enabled']:n['deficit']=min(1.,n['deficit']+self.debt[k])
        c['facts']['need_pressure']='bounded lack-of-progress debt='+str(self.debt)
        compile_batch([c]);return c

    def record(self):
        return dict(version='need-pressure-v1',owner=self.owner,config=asdict(self.config),debt=self.debt.copy(),stalls=self.stalls.copy(),last_tick=self.last_tick)

    @classmethod
    def from_record(cls,scope,r):
        x=cls(scope,PressureConfig(**r['config']))
        if r['version']!='need-pressure-v1' or r['owner']!=x.owner or set(r['debt'])!=set(NEEDS) or set(r['stalls'])!=set(NEEDS):raise ValueError('pressure checkpoint contract mismatch')
        if type(r['last_tick']) is not int or not -1<=r['last_tick']<2**63:raise ValueError('invalid pressure timeline')
        for k in NEEDS:
            v=r['debt'][k];s=r['stalls'][k]
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=x.config.limit or type(s) is not int or not 0<=s<=1000000:raise ValueError('invalid pressure checkpoint')
        x.debt=copy.deepcopy(r['debt']);x.stalls=copy.deepcopy(r['stalls']);x.last_tick=r['last_tick'];return x
