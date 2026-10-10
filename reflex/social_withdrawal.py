"""Keep three relationship stances; omit automatic rewards for harming enemies.

Dislike can reduce assistance. Whether to take resources or harm someone remains
in the ordinary objective/needs/values judgment, not a universal grudge bonus.
No additional feeling, memory state or personality coefficient is introduced.
"""
from .social_projection import CoarseSocialMemory, CoarseSocialPopulation


class WithdrawalSocialMemory(CoarseSocialMemory):
    def scores(self,ids,targets,bindings,tick):
        scores,audit=super().scores(ids,targets,bindings,tick)
        for j,key in enumerate(ids):
            if key not in audit:continue
            if audit[key]['stance']<0 and bindings[key].support<0:
                scores[j]=0.;audit[key]['adjustment']=0.
        return scores,audit


class WithdrawalSocialPopulation(CoarseSocialPopulation):
    def __init__(self,contexts,policy=None,capacity=64,half_life=24.):
        super().__init__(contexts,policy,capacity,half_life)
        self.social=[WithdrawalSocialMemory(c,capacity,half_life) for c in contexts]
