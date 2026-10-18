"""Explicit cumulative values, separate from discounted intrinsic experiences.

Only a declared WHOLE value channel may represent the change of the declared
goal potential. Its accumulated amount is neither clipped per event nor treated
as repeated momentary satisfaction. This is a semantic adapter contract, not
automatic personality inference or a guarantee that the goal model is correct.
"""
from dataclasses import dataclass,asdict,replace
import copy
from .core import VALUES,FEATURES,digest,identifier,number
from .valuation import ReturnFrame


@dataclass(frozen=True)
class CumulativePreference:
    target: str
    unit: str
    channels: tuple

    def __post_init__(self):
        identifier(self.target);identifier(self.unit)
        if not isinstance(self.channels,tuple) or not 1<=len(self.channels)<=len(VALUES):
            raise ValueError('explicit immutable cumulative value channels required')
        names=[]
        for row in self.channels:
            if not isinstance(row,tuple) or len(row)!=2:raise ValueError('value channel and fixed conversion required')
            key,conversion=row
            if not isinstance(key,str) or key not in {f'values.{name}' for name in VALUES}:raise ValueError('whole declared value channels only')
            number(conversion,-256.,256.)
            if conversion==0:raise ValueError('nonzero cumulative unit conversion required')
            names.append(key)
        if len(set(names))!=len(names):raise ValueError('distinct cumulative value channels required')

    @property
    def identity(self):return digest(['cumulative-goal-preference-v1',asdict(self)])

    def resolve(self,flow,value,origin,*,target):
        if target!=self.target:raise ValueError('matching cumulative goal required')
        number(value,-1.,1.);number(origin,-1.,1.)
        out=copy.deepcopy(flow)
        for key,conversion in self.channels:
            scale=max(1.,2*abs(conversion))
            out['values'][key.split('.')[1]]=(value-origin)*conversion/scale
        return out

    def reframe(self,frame):
        if not isinstance(frame,ReturnFrame):raise ValueError('owned return frame required')
        rows=[]
        for row in frame.effects:
            changed=list(row)
            for key,conversion in self.channels:
                changed[FEATURES.index(key.split('.')[1])]=max(1.,2*abs(conversion))
            rows.append(tuple(changed))
        return replace(frame,effects=tuple(rows))
