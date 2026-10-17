"""Bounded categorical changepoint prediction from actual observations.

Each component keeps uncertainty over when the present segment started. A
reset starts a symmetric Dirichlet(1/2) segment; it is not a remembered event
or evidence that a real opponent changed. Hazard scales compete in EventLearner.
"""
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=4096)
def categorical_renewal(sequence,alphabet,hazard):
    """Pure predictive sufficient-statistic calculation; cached across clones.

    `sequence` is only the caller's observed, same-stratum categorical history.
    Cache reuse cannot add observations or change any owner's factual memory.
    A possible switch is BEFORE the next observation; new segments have no
    privileged label, observed regime length, or hand-authored response type.
    """
    if (not isinstance(sequence,tuple) or len(sequence)>512 or type(alphabet) is not int or not 1<=alphabet<=256
        or any(type(v) is not int or not 0<=v<alphabet for v in sequence)
        or isinstance(hazard,bool) or not np.isfinite(hazard) or not 0<hazard<1):
        raise ValueError('bounded observed alphabet, sequence and hazard required')
    counts=np.empty((0,alphabet),float);weights=np.empty(0,float)
    for symbol in sequence:
        likelihood=(counts[:,symbol]+.5)/(counts.sum(axis=1)+.5*alphabet)
        masses=np.concatenate(([hazard/alphabet],(1-hazard)*weights*likelihood))
        if not len(weights):masses[:]=1.
        weights=masses/masses.sum()
        counts=np.vstack((np.zeros((1,alphabet)),counts));counts[:,symbol]+=1.
    if not len(weights):return tuple([1/alphabet]*alphabet)
    probs=(counts+.5)/(counts.sum(axis=1)[:,None]+.5*alphabet)
    p=hazard/alphabet+(1-hazard)*(weights@probs)
    return tuple(float(v) for v in p/p.sum())
