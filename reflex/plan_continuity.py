"""Owned, finite plan intention; never a learned outcome or forced commitment."""
from dataclasses import dataclass
from .core import compile_batch,digest,identifier


@dataclass(frozen=True)
class ArrivalPreference:
    incumbent: str
    arrivals: dict

    def validate(self,forecast):
        identifier(self.incumbent)
        if len(forecast.contexts)!=1 or self.incumbent not in forecast.roots or set(self.arrivals)!=set(forecast.roots):
            raise ValueError('one actor and complete current plan arrival coverage required')
        counts={len(v) for v in self.arrivals.values() if isinstance(v,(list,tuple))}
        if len(counts)!=1 or not 1<=next(iter(counts))<=8:raise ValueError('aligned bounded model branches required')
        for rows in self.arrivals.values():
            if not isinstance(rows,(list,tuple)) or any(v is not None and (type(v) is not int or not 1<=v<=forecast.horizon) for v in rows):
                raise ValueError('success arrival step or explicit unknown required')

    def mask(self,names,eligible,purpose):
        """Prune only modeled-success delays dominated in every aligned branch."""
        keep=eligible.copy();i=names.index(self.incumbent);old=self.arrivals[self.incumbent]
        audit=dict(incumbent=self.incumbent,active=False,excluded=[])
        if not eligible[i]:audit['reason']='incumbent not supported by current persona/progress/purpose';return keep,audit
        if any(v is None for v in old):audit['reason']='incumbent lacks success in every modeled branch';return keep,audit
        audit.update(active=True,reason='do not postpone a currently supported modeled completion without purpose gain')
        for j,key in enumerate(names):
            new=self.arrivals[key]
            if eligible[j] and purpose[j]<=purpose[i]+1e-12 and all(v is not None for v in new) and \
                all(a<=b for a,b in zip(old,new)) and any(a<b for a,b in zip(old,new)):
                keep[j]=False;audit['excluded'].append(key)
        return keep,audit


class PlanIntention:
    def __init__(self,c):
        compile_batch([c]);self.owner=digest(c['scope']);self.preferences=digest([c['personality'],c['values']])
        self.target=None;self.unit=None;self.last_tick=None;self.remaining=()

    def _check(self,c):
        if digest(c['scope'])!=self.owner or digest([c['personality'],c['values']])!=self.preferences:
            raise ValueError('plan intention owner/preferences mismatch')
        if self.last_tick is not None and c['tick']<=self.last_tick:raise ValueError('plan intention requires a later actual decision')

    def offer(self,c,*,target,unit,horizon):
        self._check(c)
        if type(horizon) is not int or not 1<=horizon<=16:raise ValueError('bounded horizon required')
        return self.remaining[:horizon] if (target,unit)==(self.target,self.unit) else ()

    def remember(self,c,forecast,selected,actual_root):
        """Call after actual execution; retain only the all-branch common prefix."""
        self._check(c);remaining=()
        if selected is not None:
            if forecast.roots.get(selected)!=(actual_root,):raise ValueError('actual selected root required')
            key=forecast.audit['plans'][selected]['proposal']
            traces=forecast.audit['continuation_search']['proposals'][key]['branches']
            if not traces or any(not t['actions'] or t['actions'][0]!=actual_root for t in traces):
                raise ValueError('executed root and modeled branches must agree')
            common=[]
            for column in zip(*(t['actions'][1:] for t in traces)):
                if len(set(column))!=1:break
                identifier(column[0]);common.append(column[0])
            remaining=tuple(common)
        self.remaining=remaining;self.last_tick=c['tick'];self.target=forecast.target;self.unit=forecast.audit['unit']

    def record(self):
        return dict(owner=self.owner,preferences=self.preferences,target=self.target,unit=self.unit,
            last_tick=self.last_tick,remaining=list(self.remaining))

    @classmethod
    def from_record(cls,c,row):
        out=cls(c)
        if set(row)!=set(out.record()) or row['owner']!=out.owner or row['preferences']!=out.preferences:
            raise ValueError('owned plan intention checkpoint required')
        if not isinstance(row['remaining'],list) or len(row['remaining'])>15:raise ValueError('bounded remaining intention required')
        for key in row['remaining']:identifier(key)
        if row['last_tick'] is not None and (type(row['last_tick']) is not int or row['last_tick']<0):raise ValueError('valid actual tick required')
        for key in ('target','unit'):
            if row[key] is not None:identifier(row[key])
        if row['last_tick'] is None and any(row[k] for k in ('target','unit','remaining')):raise ValueError('empty initial intention required')
        if row['last_tick'] is not None and (row['target'] is None or row['unit'] is None):raise ValueError('remembered target and time unit required')
        out.target=row['target'];out.unit=row['unit'];out.last_tick=row['last_tick'];out.remaining=tuple(row['remaining'])
        return out
