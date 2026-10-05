"""Bound scorer working sets without changing per-actor policy or RNG streams."""
from dataclasses import fields
import numpy as np
from .core import Batch, Decisions, Policy


class TiledPolicy(Policy):
    """Drop-in Policy for Population; contiguous, sequential NumPy tiles.

    Input arrays are borrowed as read-only views, outputs retain normal Decisions
    shapes. No threads, candidate pruning, actor throttling or stale snapshots.
    State ownership remains Population's responsibility. Full input/output arrays
    still scale with NPC count; only intermediate scoring arrays are bounded.
    """
    def __init__(self, tile_size=256, residual=None):
        super().__init__(residual)
        if type(tile_size) is not int or tile_size < 1:
            raise ValueError('positive integer tile_size required')
        self.tile_size = tile_size

    def decide(self, batch, stochastic=True):
        count = len(batch.ids)
        if count <= self.tile_size:
            return super().decide(batch, stochastic)
        output = None
        for start in range(0, count, self.tile_size):
            stop = min(start + self.tile_size, count)
            # Slicing, unlike Batch.take's advanced indexing, does not duplicate
            # the large effects tensor. Frozen compiled/runtime inputs stay frozen.
            part = Batch(**{f.name: getattr(batch, f.name)[start:stop]
                            for f in fields(Batch)})
            result = super().decide(part, stochastic)
            if output is None:
                output = {f.name: np.empty((count,) + getattr(result, f.name).shape[1:],
                                          dtype=getattr(result, f.name).dtype)
                          for f in fields(Decisions)}
            for name, value in output.items():
                value[start:stop] = getattr(result, name)
        return Decisions(**output)


def array_bytes(batch):
    """Resident NumPy payload only; excludes Python objects and scorer temporaries."""
    return sum(getattr(batch, f.name).nbytes for f in fields(Batch)
               if isinstance(getattr(batch, f.name), np.ndarray))
