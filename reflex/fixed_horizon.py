"""Supported fixed-H32 composition, accepted in the registered long-play study.

This composes existing conditional continuation estimates, the learning owner,
and its personality selector. One cutoff is used, exactly as in the registered
H32 candidate. There is no cutoff vote or new preparation bonus in execution.
The host supplies JSON public model states and real, separately owned feedback.
"""
import copy
from .core import digest
from .cumulative_preference import CumulativePreference
from .decision_loop import DecisionLoop
from .receding_owner import RecedingOwner
from ._horizon_consensus import consensus_planner


class FixedHorizonOwner(RecedingOwner):
    """Same lifecycle/save interface as RecedingOwner; explicitly selected policy.

    The evaluated composition uses horizon=32, seeds=(701,702,703), and
    recovery_cache=64. Other declared budgets are not independent quality evidence.
    Model/goal callback semantics must be versioned by the supplied model_id.
    """
    def __init__(self, actor, *, observe, events, terminal, assess, request_for,
                 feedback, model_id, target, cumulative_preference, horizon=32,
                 seeds=(701, 702, 703), recovery_cache=64):
        if type(horizon) is not int or not 2 <= horizon <= 128:
            raise ValueError('fixed continuation horizon in 2..128 required')
        if (not isinstance(seeds, tuple) or not 1 <= len(seeds) <= 8
                or any(type(seed) is not int or seed < 0 for seed in seeds)
                or len(set(seeds)) != len(seeds)):
            raise ValueError('one to eight distinct nonnegative model seeds required')
        if type(recovery_cache) is not int or not 0 <= recovery_cache <= 256:
            raise ValueError('within-decision recovery cache in 0..256 required')
        if type(cumulative_preference) is not CumulativePreference or cumulative_preference.target != target:
            raise ValueError('explicit same-target cumulative preference required')
        super().__init__(actor, observe=observe, events=events, terminal=terminal,
            assess=assess, record=lambda w: w, request_for=request_for, feedback=feedback,
            model_id=model_id, target=target, tail_steps=horizon-1, seeds=seeds,
            world_model='posterior', purpose_estimator='conditional', tail_recovery=2,
            cumulative_preference=cumulative_preference)
        self.recovery_cache = recovery_cache

    def _checkpoint_extension(self):
        return dict(kind='fixed-continuation-v1', recovery_cache=self.recovery_cache)

    def _with_actor(self, actor):
        options = self.options
        return FixedHorizonOwner(actor, **{k: options[k] for k in (
            'observe', 'events', 'terminal', 'assess', 'request_for', 'feedback',
            'model_id', 'target', 'cumulative_preference', 'seeds')},
            horizon=options['tail_steps']+1, recovery_cache=self.recovery_cache)

    def decide(self, public):
        if self.actor.pending is not None:
            raise ValueError('resolve actual feedback before the next decision')
        # Fail before any actual intent/experience is committed, even for an
        # invalid JSON model state or a host callback that cannot build a plan.
        digest(public)
        world = copy.deepcopy(public)
        if self.options['terminal'](world):
            raise ValueError('cannot decide in a terminal world')
        self.last = None
        request = self.request_for(self.observe(copy.deepcopy(world), copy.deepcopy(self.actor.state)))
        options = {k: self.options[k] for k in (
            'observe', 'events', 'terminal', 'assess', 'request_for', 'feedback',
            'model_id', 'target', 'seeds', 'world_model', 'purpose_estimator',
            'tail_recovery', 'cumulative_preference')}
        horizon = self.options['tail_steps']+1
        captured = {}
        planner = consensus_planner(self.actor, request, world, captured=captured,
            minimum_horizon=horizon, horizon=horizon, recovery_cache=self.recovery_cache,
            include_retry=True, coherent_cumulative_mean=True, **options)
        def complete(contexts, decisions):
            result = planner(contexts, decisions)
            if result is None:
                raise ValueError('incomplete fixed-horizon forecast; no silent reflex substitution')
            return result
        result = DecisionLoop.decide_batch([(self.actor, request)], False, complete)[0]
        self.last = captured
        return result
