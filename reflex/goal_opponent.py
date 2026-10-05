"""Opt-in public goal-conditioned enemy predictions, separate from live Policy.

The existing public tactical score adapter supplies two different purposes.
Likelihood-weighted mixtures may learn which explains an individual's past
revealed actions. They never access the actual controller, its type, seed,
current hidden intention, or future outcomes. No claim of psychological type
identification or calibrated probabilities is made.
"""
from dataclasses import dataclass
import math
from .combat import alive, legal, battle_record
from .combat_planning import tactical_scores, routes
from .congestion_experiment import enemy_hypotheses
from .core import digest
from .goal_beliefs import GoalBeliefs, GoalSnapshot

METHODS = ('uniform', 'fixed', 'fixed_coverage', 'goal_uniform', 'goal_adaptive')
GOALS = ('eliminate', 'secure')
TEMPERATURE, COVERAGE = .04, .2


def goal_models(w, actor):
    """Known rules/public geometry only; no copy of an actual rival policy."""
    models = {}
    for goal in GOALS:
        scores = tactical_scores(w, actor, goal)
        if not scores:
            raise ValueError('living opponent in a nonterminal state required')
        peak = max(scores.values())
        unscaled = {k: math.exp((s-peak)/TEMPERATURE) for k, s in scores.items()}
        total = sum(unscaled.values())
        models[goal] = {k: p/total for k, p in unscaled.items()}
    return models


@dataclass(frozen=True)
class GoalIntentForecast:
    scope: str
    owner: int
    tick: int
    state_hash: str
    fixed: tuple
    goals: GoalSnapshot
    uniform: tuple

    def probabilities(self, method, actor):
        if method not in METHODS:
            raise ValueError('known forecast method required')
        uniforms = dict(self.uniform)
        if actor not in uniforms:
            raise ValueError('opponent absent from forecast')
        uniform = dict(uniforms[actor])
        if method == 'uniform':
            return uniform
        if method in ('fixed', 'fixed_coverage'):
            predicted = {k: sum(dict(h)[actor] == k for h in self.fixed)/len(self.fixed) for k in uniform}
        else:
            predicted = self.goals.probabilities(str(actor), adaptive=method == 'goal_adaptive')
        coverage = 0. if method == 'fixed' else COVERAGE
        return {k: (1-coverage)*predicted[k]+coverage*p for k, p in uniform.items()}

    def occupancy(self, method, move):
        """Chance of any enemy INTENT; F is joint, goal/U assume independence."""
        if method not in METHODS or not move.startswith('move:'):
            raise ValueError('known method and movement destination required')
        uniform = 1-math.prod(1-dict(row).get(move, 0.) for _, row in self.uniform)
        if method == 'uniform':
            return uniform
        if method in ('fixed', 'fixed_coverage'):
            predicted = sum(move in dict(h).values() for h in self.fixed)/len(self.fixed)
        else:
            predicted = 1-math.prod(1-self.goals.probabilities(str(a), adaptive=method == 'goal_adaptive').get(move, 0.) for a, _ in self.uniform)
        coverage = 0. if method == 'fixed' else COVERAGE
        return (1-coverage)*predicted+coverage*uniform

    def record(self):
        return dict(scope=self.scope, owner=self.owner, tick=self.tick, state_hash=self.state_hash,
            weights={a: dict(weights) for a, weights in self.goals.weights},
            active={a: list(names) for a, names in self.goals.active})


class GoalOpponentMemory:
    """Combat-owned observer identity; generic learner has no combat dependency."""
    def __init__(self, scope, owner, opponents):
        opponents = tuple(opponents)
        if type(owner) is not int or not 0 <= owner < 6 or not 1 <= len(opponents) <= 3 or any(type(a) is not int or not 0 <= a < 6 for a in opponents):
            raise ValueError('bounded combat observer and opposing individuals required')
        self._beliefs = GoalBeliefs(scope, str(owner), tuple(map(str, opponents)), GOALS)
        self.scope, self.owner, self.opponents = scope, owner, opponents

    def _check(self, w, scope, owner):
        if (scope, owner) != (self.scope, self.owner) or type(owner) is not int:
            raise ValueError('game namespace or individual owner mismatch')
        if not 0 <= owner < len(w.units) or w.units[owner].hp <= 0:
            raise ValueError('living individual owner required')
        if any(a >= len(w.units) or w.units[a].team == w.units[owner].team for a in self.opponents):
            raise ValueError('opponent identity mismatch')
        if set(alive(w, 1-w.units[owner].team))-set(self.opponents):
            raise ValueError('unregistered opponent')

    def forecast(self, w, *, scope, owner):
        self._check(w, scope, owner)
        state_hash = digest(battle_record(w))
        enemies = alive(w, 1-w.units[owner].team)
        models = {str(a): goal_models(w, a) for a in enemies}
        allowed = routes(w, 1-w.units[owner].team) or ('secure',)
        goals = self._beliefs.forecast(w.tick, state_hash, models, active={str(a): allowed for a in enemies}, scope=scope, owner=str(owner))
        fixed = tuple(tuple(sorted(h.items())) for h in enemy_hypotheses(w, w.units[owner].team))
        uniform = tuple((a, tuple((k, 1/len(legal(w, a))) for k in legal(w, a))) for a in enemies)
        return GoalIntentForecast(scope, owner, w.tick, state_hash, fixed, goals, uniform)

    def reveal(self, w, actions, *, scope, owner):
        self._check(w, scope, owner)
        if set(actions) != set(alive(w, 1-w.units[owner].team)):
            raise ValueError('complete opposing public actions required')
        self._beliefs.reveal(w.tick, digest(battle_record(w)), {str(a): k for a, k in actions.items()}, scope=scope, owner=str(owner))

    def record(self):
        return self._beliefs.record()
