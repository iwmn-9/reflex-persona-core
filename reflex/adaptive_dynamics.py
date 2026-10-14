"""Learn public event laws outside a supplied library of behavior hypotheses.

The rules/observation alphabet still belong to the adapter. New conditional
frequencies are acquired online; no actual latent policy, reward for unchosen
actions, or rollout result enters factual memory. This is a bounded predictive
mixture, not causal discovery or a detector that can prove why behavior changed.
"""
from dataclasses import dataclass
from itertools import combinations
import copy
import math

from .core import digest
from .closed_loop import Transition
from .belief_planning import UnsupportedObservation, _probabilities


def _labels(values, maximum):
    return (isinstance(values, tuple) and len(values) <= maximum
            and len(set(values)) == len(values)
            and all(isinstance(v, str) and 0 < len(v) <= 128 for v in values))


@dataclass(frozen=True)
class Evidence:
    context: tuple
    legal: tuple
    outcome: str


@dataclass(frozen=True)
class PredictiveState:
    identity: str
    weights: tuple
    history: tuple = ()
    observations: int = 0


class EventLearner:
    """Prequential competition between library and learned conditional models.

    Every subset of <=4 observable categorical factors is a possible partition.
    Its cells and distributions are learned only when visited. Short/long
    windows compete by predictions made BEFORE reveal. A fixed share revives
    previously poor predictors after a change, without erasing factual history.
    Legal alphabets are separate strata to avoid learning from censored actions.
    """
    def __init__(self, *, factors=(), names=(), owner, model_id,
                 windows=(8, 32, 128), share=.02, smoothing=.02, learned=True):
        self.factors, self.names = tuple(factors), tuple(names)
        self.windows = tuple(windows)
        if not _labels(self.factors, 4) or not _labels(self.names, 16):
            raise ValueError('bounded named observable factors and library required')
        if (not self.windows or len(self.windows) > 4 or len(set(self.windows)) != len(self.windows)
                or any(type(n) is not int or not 2 <= n <= 512 for n in self.windows)):
            raise ValueError('distinct bounded observation windows required')
        if type(learned) is not bool or (not learned and not self.names):
            raise ValueError('at least one predictive model required')
        if any(type(v) not in (float, int) or not math.isfinite(v) or not 0 < v < 1 for v in (share, smoothing)):
            raise ValueError('positive bounded recovery and smoothing required')
        if not all(isinstance(v, str) and v for v in (owner, model_id)):
            raise ValueError('identified observer and observation semantics required')
        masks = [c for n in range(len(self.factors)+1) for c in combinations(range(len(self.factors)), n)]
        self.experts = tuple((mask, window) for mask in masks for window in self.windows) if learned else ()
        self.share, self.smoothing, self.learned = float(share), float(smoothing), learned
        prior = [2.**-len(mask) for mask, _ in self.experts]
        total = sum(prior)
        mass = .5 if self.names and self.experts else 1.
        self.prior = tuple([mass/len(self.names)]*len(self.names) if self.names else ())
        self.prior += tuple(mass*p/total for p in prior)
        self.identity = digest(['event-learner-v1', self.factors, self.names, owner, model_id,
                                self.windows, self.share, self.smoothing, learned])

    def start(self):
        return PredictiveState(self.identity, self.prior)

    def _inputs(self, context, legal, models):
        if (not isinstance(context, tuple) or len(context) != len(self.factors)
                or any(not isinstance(v, str) or not v or len(v) > 128 for v in context)):
            raise ValueError('one bounded observed categorical value per factor required')
        if not _labels(legal, 256) or not legal or tuple(sorted(legal)) != legal:
            raise ValueError('sorted unique legal observable event symbols required')
        if not isinstance(models, dict) or set(models) != set(self.names):
            raise ValueError('one pre-observation distribution per library hypothesis required')
        rows = []
        for name in self.names:
            row = models[name]
            if not isinstance(row, dict) or set(row) != set(legal) or not _probabilities(list(row.values())):
                raise ValueError('library distributions must cover the legal event alphabet')
            rows.append(tuple((1-self.smoothing)*row[k]+self.smoothing/len(legal) for k in legal))
        return rows

    def _check(self, state):
        if (not isinstance(state, PredictiveState) or state.identity != self.identity
                or not isinstance(state.weights, tuple) or len(state.weights) != len(self.prior)
                or not _probabilities(state.weights) or type(state.observations) is not int
                or state.observations < 0 or not isinstance(state.history, tuple)
                or len(state.history) != min(state.observations, max(self.windows))):
            raise ValueError('complete normalized observer-local predictive memory required')
        for row in state.history:
            if (not isinstance(row, Evidence) or not isinstance(row.context, tuple)
                    or len(row.context) != len(self.factors)
                    or any(not isinstance(v, str) or not v or len(v) > 128 for v in row.context)
                    or not _labels(row.legal, 256) or not row.legal or tuple(sorted(row.legal)) != row.legal
                    or row.outcome not in row.legal):
                raise ValueError('valid observed history required')

    def _forecast(self, state, context, legal, models):
        self._check(state)
        rows = self._inputs(context, legal, models)
        for mask, window in self.experts:
            counts = dict.fromkeys(legal, .5)  # symmetric Dirichlet predictive
            for row in state.history[-window:]:
                if row.legal == legal and all(row.context[i] == context[i] for i in mask):
                    counts[row.outcome] += 1.
            total = math.fsum(counts.values())
            rows.append(tuple(counts[k]/total for k in legal))
        prediction = [math.fsum(w*r[j] for w, r in zip(state.weights, rows)) for j in range(len(legal))]
        total = math.fsum(prediction)
        return rows, dict(zip(legal, (p/total for p in prediction)))

    def predict(self, state, context, legal, models):
        return self._forecast(state, context, legal, models)[1]

    def observe(self, state, context, legal, models, outcome):
        """Return a new memory. Score the old forecast, then learn this reveal."""
        rows, prediction = self._forecast(state, context, legal, models)
        if outcome not in legal:
            raise UnsupportedObservation('event outside rule alphabet; observation contract needs repair')
        j = legal.index(outcome)
        weights = [w*row[j] for w, row in zip(state.weights, rows)]
        total = math.fsum(weights)
        weights = tuple((1-self.share)*w/total+self.share*p for w, p in zip(weights, self.prior))
        normalizer = math.fsum(weights)
        history = (*state.history, Evidence(context, legal, outcome))[-max(self.windows):]
        return PredictiveState(self.identity, tuple(w/normalizer for w in weights), history, state.observations+1)

    def diagnostics(self, state):
        self._check(state)
        n = len(self.names)
        return dict(observations=state.observations, library_weight=sum(state.weights[:n]),
                    learned_weight=sum(state.weights[n:]),
                    experts=[dict(factors=[self.factors[i] for i in mask], window=window, weight=w)
                             for (mask, window), w in zip(self.experts, state.weights[n:])],
                    interpretation='predictive mixture weights; not probability of a true cause')

    def record(self, state):
        self._check(state)
        return dict(format='event-learner-v1', identity=state.identity, weights=list(state.weights),
                    observations=state.observations,
                    history=[dict(context=list(r.context), legal=list(r.legal), outcome=r.outcome) for r in state.history])

    def restore(self, record):
        if set(record) != {'format','identity','weights','observations','history'} or record['format'] != 'event-learner-v1':
            raise ValueError('complete predictive checkpoint required')
        state = PredictiveState(record['identity'], tuple(record['weights']),
            tuple(Evidence(tuple(r['context']), tuple(r['legal']), r['outcome']) for r in record['history']), record['observations'])
        self._check(state)
        if self.record(state) != record:
            raise ValueError('lossless predictive checkpoint required')
        return state


@dataclass(frozen=True)
class PublicEvent:
    symbol: str
    public: object
    flow: dict


@dataclass(frozen=True)
class AdaptiveState:
    public: object
    memory: PredictiveState
    identity: str
    tick: int = 0


class AdaptiveKernel:
    """Same learned event law in actual reveals and ClosedLoop future branches.

    `events(public, action)` enumerates the legal *observable* outcomes of known
    rules. Each symbol has a distinct public state/flow. Never label hidden
    outcomes separately when the owner cannot distinguish them.
    `features` and `models` use only that owner's pre-action public information.
    """
    def __init__(self, *, learner, events, features, models, public_record,
                 model_id, update=True):
        if not isinstance(learner, EventLearner) or not all(callable(f) for f in (events, features, models, public_record)):
            raise ValueError('learner and explicit pure public adapter callbacks required')
        if not isinstance(model_id, str) or not model_id or type(update) is not bool:
            raise ValueError('identified rules and explicit forecast update mode required')
        self.learner, self.events, self.features, self.models = learner, events, features, models
        self.public_record, self.update = public_record, update
        self.state_identity = digest(['adaptive-public-v1', learner.identity, model_id])
        self.identity = digest([self.state_identity, update])

    def _check(self, state):
        if (not isinstance(state, AdaptiveState) or state.identity != self.state_identity
                or type(state.tick) is not int or state.tick < 0):
            raise ValueError('matching observer/rules public state required')
        self.learner._check(state.memory)

    def start(self, public, memory=None):
        state = AdaptiveState(copy.deepcopy(public), memory or self.learner.start(), self.state_identity)
        self._check(state)
        return state

    def record(self, state):
        self._check(state)
        return dict(format='adaptive-public-v1', identity=state.identity, tick=state.tick,
                    public=copy.deepcopy(self.public_record(state.public)), memory=self.learner.record(state.memory))

    def restore(self, record, public_restore):
        if set(record) != {'format','identity','tick','public','memory'} or record['format'] != 'adaptive-public-v1':
            raise ValueError('complete adaptive public checkpoint required')
        state = AdaptiveState(public_restore(copy.deepcopy(record['public'])), self.learner.restore(record['memory']), record['identity'], record['tick'])
        self._check(state)
        if self.record(state) != record:
            raise ValueError('lossless public checkpoint required')
        return state

    def _event(self, public, flow):
        return digest([self.public_record(public), flow])

    def forecast(self, state, action):
        self._check(state)
        rows = self.events(copy.deepcopy(state.public), action)
        if (not isinstance(rows, (tuple, list)) or not 1 <= len(rows) <= 256
                or any(not isinstance(r, PublicEvent) or not isinstance(r.flow, dict) or r.flow.get('p') != 1. for r in rows)):
            raise ValueError('bounded explicit public rule events required')
        symbols = tuple(r.symbol for r in rows)
        if not _labels(symbols, 256) or len({self._event(r.public, r.flow) for r in rows}) != len(rows):
            raise ValueError('unique symbols and distinguishable observed events required')
        legal = tuple(sorted(symbols))
        context = self.features(copy.deepcopy(state.public), action)
        models = self.models(copy.deepcopy(state.public), action)
        probabilities = self.learner.predict(state.memory, context, legal, models)
        return rows, context, legal, models, probabilities

    def _next(self, state, row, context, legal, models):
        memory = self.learner.observe(state.memory, context, legal, models, row.symbol) if self.update else state.memory
        return AdaptiveState(copy.deepcopy(row.public), memory, state.identity, state.tick+1)

    def __call__(self, state, action):
        rows, context, legal, models, probabilities = self.forecast(state, action)
        return [Transition(probabilities[r.symbol], self._next(state, r, context, legal, models), copy.deepcopy(r.flow)) for r in rows]

    def reveal(self, state, action, public, flow):
        rows, context, legal, models, _ = self.forecast(state, action)
        event = self._event(public, flow)
        for row in rows:
            if self._event(row.public, row.flow) == event:
                return self._next(state, row, context, legal, models)
        raise UnsupportedObservation('observed event violates declared public rules; no memory committed')
