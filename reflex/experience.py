"""Optional bounded game-side observed experience; never changes personality."""
from collections import OrderedDict
import numpy as np
from .core import FEATURES, digest


class Experience:
    def __init__(self, scope, capacity=64):
        if not isinstance(capacity,int) or not 1<=capacity<=1024: raise ValueError("bounded capacity required")
        self.scope=digest(scope); self.capacity=capacity; self.entries=OrderedDict()

    def observe(self, scope, situation, action, observed_effects):
        if digest(scope)!=self.scope: raise ValueError("experience belongs to another NPC/episode")
        value=np.asarray(observed_effects,dtype=float)
        if value.shape!=(len(FEATURES),) or not np.isfinite(value).all() or (abs(value)>1).any() or value[-1]<0:
            raise ValueError("bounded observed effect vector required")
        key=(str(situation),str(action))
        count,mean=self.entries.pop(key,(0,np.zeros(len(FEATURES))))
        # Exponential update after eight observations: can adapt when conditions change.
        count=min(count+1,1000000); mean=mean+(value-mean)/min(count,8)
        self.entries[key]=(count,mean)
        while len(self.entries)>self.capacity: self.entries.popitem(last=False)

    def estimate(self, scope, situation, action):
        if digest(scope)!=self.scope: raise ValueError("experience scope mismatch")
        entry=self.entries.get((str(situation),str(action)))
        return None if entry is None else dict(observations=entry[0],mean=entry[1].copy())
