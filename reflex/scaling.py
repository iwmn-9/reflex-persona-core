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
    def __init__(self, tile_size=256, residual=None, *, principle_priority='lexicographic',commitments=None):
        super().__init__(residual,principle_priority=principle_priority,commitments=commitments)
        if type(tile_size) is not int or tile_size < 1:
            raise ValueError('positive integer tile_size required')
        self.tile_size = tile_size

    def decide(self, batch, stochastic=True,*,return_frame=None,appraisal=None,purpose=None,max_social_regret=.15,outcome_audit=None):
        if outcome_audit is not None and not isinstance(outcome_audit,dict):raise ValueError('explicit outcome audit dictionary required')
        count = len(batch.ids)
        if count <= self.tile_size:
            return super().decide(batch, stochastic,return_frame=return_frame,appraisal=appraisal,purpose=purpose,max_social_regret=max_social_regret,outcome_audit=outcome_audit)
        bias=None if appraisal is None else np.asarray(appraisal,dtype=float)
        goal=None if purpose is None else np.asarray(purpose,dtype=float)
        if any(x is not None and x.shape!=batch.legal.shape for x in (bias,goal)):
            raise ValueError('one appraisal/purpose row per actor and root required')
        output = None;audited=None
        for start in range(0, count, self.tile_size):
            stop = min(start + self.tile_size, count)
            # Slicing, unlike Batch.take's advanced indexing, does not duplicate
            # the large effects tensor. Frozen compiled/runtime inputs stay frozen.
            part = Batch(**{f.name: getattr(batch, f.name)[start:stop]
                            for f in fields(Batch)})
            frame=None if return_frame is None else return_frame.take(range(start,stop))
            part_audit={} if outcome_audit is not None else None
            result = super().decide(part, stochastic,return_frame=frame,
                appraisal=None if bias is None else bias[start:stop],purpose=None if goal is None else goal[start:stop],max_social_regret=max_social_regret,outcome_audit=part_audit)
            if part_audit is not None:
                if audited is None:audited={k:np.empty((count,)+part_audit[k].shape[1:]) for k in ('scores','expected')}
                for k in audited:audited[k][start:stop]=part_audit[k]
            if output is None:
                output = {f.name: np.empty((count,) + getattr(result, f.name).shape[1:],
                                          dtype=getattr(result, f.name).dtype)
                          for f in fields(Decisions)}
            for name, value in output.items():
                value[start:stop] = getattr(result, name)
        if outcome_audit is not None:outcome_audit.update(audited,semantics=part_audit['semantics'])
        return Decisions(**output)


def array_bytes(batch):
    """Resident NumPy payload only; excludes Python objects and scorer temporaries."""
    return sum(getattr(batch, f.name).nbytes for f in fields(Batch)
               if isinstance(getattr(batch, f.name), np.ndarray))
