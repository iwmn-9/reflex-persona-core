"""Game-independent, observer-owned mixtures of bounded action hypotheses.

Adapters provide legal action distributions and public state identity. Only a
later, complete public action reveal can change weights. These likelihood
weights are predictions within a supplied model family, not true goal labels.
"""
import copy
from dataclasses import dataclass
from .opponent_beliefs import HypothesisTracker, distributions

RETENTION = .9
SMOOTHING = .08
MAX_OPPONENTS, MAX_GOALS, MAX_ACTIONS = 32, 8, 64
MAX_TICK = (1 << 63)-1


def _name(value):
    return isinstance(value, str) and 0 < len(value) <= 128


@dataclass(frozen=True)
class GoalSnapshot:
    scope: str
    owner: str
    tick: int
    state_key: str
    models: tuple
    weights: tuple
    active: tuple

    def probabilities(self, actor, *, adaptive=True):
        tables = dict(self.models)
        if actor not in tables:
            raise ValueError('opponent absent from forecast')
        models = {n: dict(row) for n, row in tables[actor]}
        active = dict(self.active)[actor]
        weights = dict(dict(self.weights)[actor])
        mass = sum(weights[n] for n in active) if adaptive else len(active)
        actions, matrix = distributions(models, tuple(models), SMOOTHING)
        rows = dict(zip(models, matrix))
        return {a: sum((weights[n] if adaptive else 1.)*rows[n][j] for n in active)/mass
                for j, a in enumerate(actions)}


class GoalBeliefs:
    """Fixed actor/goal registration with bounded underlying hypothesis state.

    Memory is observer-local and never shared across games or opponents. The
    caller's state key must describe permitted public observations only. Models
    are detached and validated before pending state or learned weights change.
    """
    def __init__(self, scope, owner, opponents, goals):
        opponents, goals = tuple(opponents), tuple(goals)
        if not _name(scope) or not _name(owner):
            raise ValueError('bounded game and observer names required')
        if not 1 <= len(opponents) <= MAX_OPPONENTS or len(set(opponents)) != len(opponents) or any(not _name(a) or a == owner for a in opponents):
            raise ValueError('distinct bounded opponent names required')
        if not 2 <= len(goals) <= MAX_GOALS or len(set(goals)) != len(goals) or any(not _name(n) for n in goals):
            raise ValueError('2..8 distinct bounded goal names required')
        self.scope, self.owner, self.opponents, self.goals = scope, owner, opponents, goals
        self._trackers = {a: HypothesisTracker(goals, retention=RETENTION, smoothing=SMOOTHING, responsive=False) for a in opponents}
        self._tick, self._pending = -1, None

    def _check(self, tick, state_key, scope, owner):
        if (scope, owner) != (self.scope, self.owner):
            raise ValueError('game namespace or observer mismatch')
        if type(tick) is not int or not 0 <= tick <= MAX_TICK or tick != self._tick+1 or not _name(state_key):
            raise ValueError('consecutive public time and bounded state key required')

    def forecast(self, tick, state_key, models, *, active, scope, owner):
        self._check(tick, state_key, scope, owner)
        if set(models)-set(self.opponents) or set(active) != set(models):
            raise ValueError('registered opponents and matching active goals required')
        tables, weights, selected = [], [], []
        for a in sorted(models):
            actions, _ = distributions(models[a], self.goals, SMOOTHING)
            if not 1 <= len(actions) <= MAX_ACTIONS or any(not _name(k) for k in actions):
                raise ValueError('bounded legal action names required')
            chosen = tuple(active[a])
            if not chosen or len(chosen) != len(set(chosen)) or set(chosen)-set(self.goals):
                raise ValueError('nonempty distinct active goal subset required')
            tables.append((a, tuple((n, tuple((k, float(models[a][n][k])) for k in actions)) for n in self.goals)))
            weights.append((a, tuple(zip(self.goals, self._trackers[a].snapshot().weights))))
            selected.append((a, chosen))
        forecast = GoalSnapshot(scope, owner, tick, state_key, tuple(tables), tuple(weights), tuple(selected))
        if self._pending is not None and self._pending != forecast:
            raise ValueError('pending public forecast differs')
        self._pending = forecast
        return forecast

    def reveal(self, tick, state_key, actions, *, scope, owner):
        self._check(tick, state_key, scope, owner)
        pending = self._pending
        if pending is None or (pending.tick, pending.state_key) != (tick, state_key):
            raise ValueError('exact preceding public forecast required')
        tables = {a: {n: dict(row) for n, row in models} for a, models in pending.models}
        if set(actions) != set(tables) or any(actions[a] not in tables[a][self.goals[0]] for a in tables):
            raise ValueError('complete previously legal public reveals required')
        # Commit all opponents atomically, including unexpected observe failures.
        updated = copy.deepcopy(self._trackers)
        for a in tables:
            updated[a].observe(tables[a], actions[a], f'tick-{tick}')
        self._trackers, self._tick, self._pending = updated, tick, None

    def record(self):
        return dict(scope=self.scope, owner=self.owner, opponents=list(self.opponents), goals=list(self.goals),
            last_tick=self._tick, pending=None if self._pending is None else dict(tick=self._pending.tick, state_key=self._pending.state_key),
            actors={a: dict(weights=dict(zip(self.goals, t.snapshot().weights)), logs=list(t.logs),
                observations=t.observations, retained_ids=len(t.ids), retained_lifts=len(t.lifts)) for a, t in self._trackers.items()})
