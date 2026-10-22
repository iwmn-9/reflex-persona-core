"""Owned control-variate corrections to linear forecast expectations.

Sampled outcomes still price downside risk. Corrections do not create observed
rewards, alter personality, or replace the sampled loss distribution. They only
estimate means in the return frame's restored units; model bias remains.
"""
from dataclasses import dataclass
import numpy as np
from .core import FEATURES, identifier, number


@dataclass(frozen=True)
class ReturnMeans:
    owners: tuple
    plans: tuple
    corrections: tuple

    def __post_init__(self):
        if (not all(isinstance(x, tuple) for x in (self.owners,self.plans,self.corrections))
                or not self.owners or len(self.owners)!=len(self.corrections)
                or not self.plans or len(set(self.plans))!=len(self.plans)):
            raise ValueError('immutable owned plan mean corrections required')
        for key in self.owners:
            if not isinstance(key,str) or len(key)!=64 or any(c not in '0123456789abcdef' for c in key):
                raise ValueError('owned forecast mean required')
        for plan in self.plans:identifier(plan)
        for owner in self.corrections:
            if not isinstance(owner,tuple) or len(owner)!=len(self.plans):raise ValueError('complete plan mean coverage required')
            for row in owner:
                if not isinstance(row,tuple) or len(row)!=len(FEATURES):raise ValueError('complete feature mean coverage required')
                for value in row:number(value,-1024.,1024.)

    def numeric(self,batch):
        if tuple(batch.owners)!=self.owners or any(not set(ids)<=set(self.plans) for ids in batch.ids):
            raise ValueError('mean correction belongs to another owner or candidate set')
        if not np.all(batch.confidence==1.):raise ValueError('mean correction requires supported model probabilities')
        index={key:i for i,key in enumerate(self.plans)}
        return np.array([[self.corrections[i][index[k]] for k in ids] for i,ids in enumerate(batch.ids)],float)

    def retain(self,plans):
        plans=tuple(plans);index={key:i for i,key in enumerate(self.plans)}
        return ReturnMeans(self.owners,plans,tuple(tuple(row[index[k]] for k in plans) for row in self.corrections))

    def take(self,indices):
        ids=tuple(int(i) for i in indices)
        return ReturnMeans(tuple(self.owners[i] for i in ids),self.plans,tuple(self.corrections[i] for i in ids))

    def record(self):
        return dict(format='return-means-v1',owners=list(self.owners),plans=list(self.plans),
            corrections=[[list(row) for row in owner] for owner in self.corrections])
