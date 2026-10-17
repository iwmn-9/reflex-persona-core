"""Exchangeable stratification of rollout random numbers, never real events."""
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=4096)
def stratified_uniforms(seeds,tick):
    """One uniform marginal per path; each step visits every equal-width stratum.

    Independent step permutations avoid forcing each simulated person into a
    repeated low/high response. This reduces one-step categorical count noise;
    it does not guarantee lower variance for arbitrary nonlinear whole returns.
    Shared banks couple alternative actions without inspecting their rewards.
    """
    if (not isinstance(seeds,tuple) or not 1<=len(seeds)<=256 or len(set(seeds))!=len(seeds)
        or any(type(s) is not int or not 0<=s<2**32 for s in seeds)
        or type(tick) is not int or tick<0):
        raise ValueError('bounded unique seeds and nonnegative tick required')
    rng=np.random.default_rng(np.random.SeedSequence([*seeds,tick,384729]))
    return tuple((rng.permutation(len(seeds))+rng.random(len(seeds)))/len(seeds))
