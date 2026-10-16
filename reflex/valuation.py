"""Owned units for bounded forecast transport and cumulative preference scoring.

Immediate observations retain their existing [-1,1] contract. A forecast may
store averaged flows in that same packet, but must restore their declared
return units before the owner's risk, principle tier and scoring are applied.
This frame carries no utility weights, game rule or hidden state.
"""
from dataclasses import dataclass
import numpy as np
from .core import FEATURES,owner_key,identifier,number


@dataclass(frozen=True)
class ReturnFrame:
    owners: tuple
    effects: tuple
    root_terms: tuple
    unit: str

    def __post_init__(self):
        identifier(self.unit)
        if not all(isinstance(x,tuple) for x in (self.owners,self.effects,self.root_terms)) or not self.owners or not len(self.owners)==len(self.effects)==len(self.root_terms):
            raise ValueError('one immutable return frame per owner required')
        for key,row,root in zip(self.owners,self.effects,self.root_terms):
            if not isinstance(key,str) or len(key)!=64 or any(c not in '0123456789abcdef' for c in key):raise ValueError('owned return frame required')
            if not isinstance(row,tuple) or len(row)!=len(FEATURES):raise ValueError('one return scale per feature required')
            for scale in (*row,root):number(scale,1,256)

    @classmethod
    def flows(cls,contexts,normalizer,unit,*,endpoint=False):
        scale=number(normalizer,1,256)
        row=(1. if endpoint else scale,)+(scale,)*(len(FEATURES)-1)
        return cls(tuple(owner_key(c) for c in contexts),(row,)*len(contexts),(scale,)*len(contexts),unit)

    def numeric(self,batch):
        if tuple(batch.owners)!=self.owners:raise ValueError('return frame belongs to another owner or goal')
        return np.asarray(self.effects,float),np.asarray(self.root_terms,float)

    def take(self,indices):
        ids=tuple(int(i) for i in indices)
        return ReturnFrame(tuple(self.owners[i] for i in ids),tuple(self.effects[i] for i in ids),
            tuple(self.root_terms[i] for i in ids),self.unit)

    def record(self):
        return dict(format='return-frame-v1',owners=list(self.owners),effects=[list(r) for r in self.effects],
            root_terms=list(self.root_terms),unit=self.unit)
