"""Bounded response-label permutations used by experimental learners."""
from functools import lru_cache
from itertools import permutations


@lru_cache(maxsize=4096)
def complete_templates(models, alphabet):
    if alphabet > 4:
        return models
    # Duplicate shapes retain the same hypothesis identity across contexts.
    return tuple(tuple(model[i] for i in order) for model in models
                 for order in permutations(range(alphabet)))
