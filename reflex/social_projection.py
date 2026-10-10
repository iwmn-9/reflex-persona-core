"""Omit fine relationship distinctions at the action boundary.

Evidence remains in SocialMemory for repair, forgetting and recipient isolation.
Only its contribution to action scoring is compressed. These three stances are
engineering choices, not additional personality traits or measured psychology.
"""
import numpy as np
from .social import SocialMemory, SocialPopulation


def stance(signal):
    """Weak evidence is neutral; supported liking/dislike has one magnitude."""
    return 0. if abs(signal)<.15 else float(np.sign(signal)*.5)


class CoarseSocialMemory(SocialMemory):
    def relationship_adjustment(self,status,weight):
        return weight*stance(status['attitude']*status['confidence'])

    def scores(self,ids,targets,bindings,tick):
        scores,audit=super().scores(ids,targets,bindings,tick)
        for row in audit.values():
            signal=row['attitude']*row['confidence']
            row.update(signal=signal,stance=stance(signal))
        return scores,audit


class CoarseSocialPopulation(SocialPopulation):
    """The same numeric reflex and feedback path with a three-stance projection."""
    def __init__(self,contexts,policy=None,capacity=64,half_life=24.):
        super().__init__(contexts,policy,capacity,half_life)
        self.social=[CoarseSocialMemory(c,capacity,half_life) for c in contexts]
