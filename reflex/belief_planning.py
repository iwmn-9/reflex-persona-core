"""Finite public observation trees for learning-aware planning.

The adapter supplies conditional rules, never the realized hidden hypothesis.
Indistinguishable observed outcomes share ONE posterior and future decision.
This is exact Bayesian filtering inside the declared finite model, not a claim
that its hypotheses describe the real world or that personality is optimal.
"""
from dataclasses import dataclass
import copy
import math
from .core import digest
from .closed_loop import Transition


@dataclass(frozen=True)
class HiddenOutcome:
    probability: float
    public: object
    flow: dict
    next_hypothesis: str


@dataclass(frozen=True)
class InformationState:
    public: object
    weights: tuple
    identity: str
    tick: int = 0


class UnsupportedObservation(ValueError):
    """The observed event has zero support; the old belief remains untouched."""


def _probabilities(values):
    return (bool(values) and all(type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1 for p in values)
            and abs(sum(values)-1.) <= 1e-8)


class BeliefKernel:
    """Pure transition/filter shared by imagined and actual public reveals.

    `model(hypothesis, public, action)` enumerates a normalized conditional
    distribution of HiddenOutcome, including optional hidden regime changes.
    `public_record` must capture ALL permitted observed state; flow is observed
    too. Unobservable rewards/state must be marginalized by the adapter first.
    No outcome may smuggle the sampled hypothesis or actual RNG into public.

    `update=False` is an explicit forecast ablation: propagate hidden dynamics
    but don't condition weights on the new observation. It is not a competing
    inference method. Actual execution should use the updating kernel.
    """
    def __init__(self, *, names, model, public_record, model_id, owner, update=True, branches=4096):
        names = tuple(names)
        if not 1 <= len(names) <= 32 or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
            raise ValueError('1..32 named finite hypotheses required')
        if any(not isinstance(n, str) or not n for n in (model_id, owner)) or not callable(model) or not callable(public_record):
            raise ValueError('identified public model and observer required')
        if type(update) is not bool or type(branches) is not int or not 1 <= branches <= 65536:
            raise ValueError('explicit update rule and bounded branch capacity required')
        self.names, self.model, self.public_record = names, model, public_record
        self.update, self.branches = update, branches
        # A factual state belongs to the model/owner, independently of the
        # counterfactual updating ablation used for one forecast.
        self.state_identity = digest(['public-belief-state-v1', model_id, owner, names])
        self.identity = digest([self.state_identity, update, branches])

    def _check(self, state):
        if (not isinstance(state, InformationState) or state.identity != self.state_identity
                or not isinstance(state.weights, tuple) or len(state.weights) != len(self.names)
                or not _probabilities(state.weights) or type(state.tick) is not int or state.tick < 0):
            raise ValueError('normalized belief belonging to this model and observer required')

    def start(self, public, prior):
        state = InformationState(copy.deepcopy(public), tuple(prior), self.state_identity)
        self._check(state)
        return state

    def record(self, state):
        self._check(state)
        return dict(format='public-belief-state-v1', identity=state.identity, tick=state.tick,
                    weights=list(state.weights), public=copy.deepcopy(self.public_record(state.public)))

    def restore(self, record, public_restore):
        if set(record) != {'format','identity','tick','weights','public'} or record['format'] != 'public-belief-state-v1':
            raise ValueError('complete belief checkpoint required')
        state = InformationState(public_restore(copy.deepcopy(record['public'])), tuple(record['weights']), record['identity'], record['tick'])
        self._check(state)
        if self.record(state) != record:
            raise ValueError('lossless public checkpoint required')
        return state

    def _event(self, public, flow):
        return digest([self.public_record(public), flow])

    def __call__(self, state, action):
        self._check(state)
        groups, marginal, count = {}, [0.] * len(self.names), 0
        for h, weight in zip(self.names, state.weights):
            if weight == 0:
                continue
            rows = self.model(h, copy.deepcopy(state.public), action)
            if not isinstance(rows, (tuple, list)) or not rows or len(rows) > self.branches:
                raise ValueError('bounded explicit conditional outcome distribution required')
            if any(not isinstance(r, HiddenOutcome) for r in rows) or not _probabilities([r.probability for r in rows]):
                raise ValueError('normalized conditional likelihoods required')
            count += len(rows)
            if count > self.branches:
                raise ValueError('complete hidden branch capacity exceeded')
            for row in rows:
                if row.next_hypothesis not in self.names or not isinstance(row.flow, dict) or row.flow.get('p') != 1.:
                    raise ValueError('registered successor hypothesis and one observed flow required')
                if row.probability == 0:
                    continue
                mass = weight * row.probability
                if mass == 0:  # underflowed mass cannot define a posterior
                    continue
                event = self._event(row.public, row.flow)
                if event not in groups:
                    groups[event] = [copy.deepcopy(row.public), copy.deepcopy(row.flow), [0.] * len(self.names)]
                index = self.names.index(row.next_hypothesis)
                groups[event][2][index] += mass
                marginal[index] += mass
        transitions = []
        total = math.fsum(math.fsum(row[2]) for row in groups.values())
        prior_total = math.fsum(marginal)
        for event in sorted(groups):
            public, flow, masses = groups[event]
            mass = math.fsum(masses)
            probability = mass/total
            posterior = tuple(p/mass for p in masses) if self.update else tuple(p/prior_total for p in marginal)
            successor = InformationState(public, posterior, state.identity, state.tick+1)
            self._check(successor)
            transitions.append(Transition(probability, successor, flow))
        return transitions

    def reveal(self, state, action, public, flow):
        event = self._event(public, flow)
        for transition in self(state, action):
            if self._event(transition.state.public, transition.flow) == event:
                return transition.state
        raise UnsupportedObservation('observed event absent from declared model; no posterior committed')
