"""Offline, owner-isolated public intent memory; never imported by live decisions.

The adapter supplies known legal actions and public geometry. This is a bounded
behavior frequency model, not a personality diagnosis or calibrated trust score.
No world outcomes, controller labels, random seeds, or current intents enter a
forecast. Observations are legal, later public intent reveals, not rollouts.
"""
from dataclasses import dataclass
import math
from .combat import alive, legal, in_zone, zone_distance, firing_distance, battle_record
from .congestion_experiment import enemy_hypotheses
from .core import digest

METHODS = ('uniform', 'fixed', 'fixed_coverage', 'cold_memory', 'observed_memory')
MIXTURES = {'uniform': (0., 0., 1.), 'fixed': (1., 0., 0.),
            'fixed_coverage': (.8, 0., .2), 'cold_memory': (.4, 0., .6),
            'observed_memory': (.4, .4, .2)}
RETENTION = .9
PRIOR = 2.
CONTEXTS, SIGNATURES, MAX_OPPONENTS = 8, 13, 3


def public_features(w, actor):
    """Fixed-size signatures with no outcome or actual-opponent policy lookup."""
    keys = legal(w, actor)
    if not keys:
        raise ValueError('living actor in a nonterminal public state required')
    u = w.units[actor]
    context = int(u.ammo > 0) * 4 + int(any(k.startswith('shoot:') for k in keys)) * 2 + int(in_zone(u.pos))
    targets = tuple(w.units[j].pos for j in alive(w, 1-u.team))
    old = (zone_distance(w, u.pos), firing_distance(w.walls, u.pos, targets))
    signatures = {}
    sign = lambda x: (x > 0) - (x < 0)
    for key in keys:
        if key.startswith('move:'):
            pos = tuple(map(int, key.split(':')[1:]))
            delta = (zone_distance(w, pos)-old[0], firing_distance(w.walls, pos, targets)-old[1])
            signatures[key] = 4 + (sign(delta[0])+1)*3 + sign(delta[1])+1
        else:
            signatures[key] = {'guard': 0, 'reload': 1, 'shoot': 2, 'heal': 3}[key.split(':')[0]]
    return context, signatures


@dataclass(frozen=True)
class IntentForecast:
    scope: str
    owner: int
    tick: int
    state_hash: str
    # Immutable sorted tuples, detached from the mutable store.
    fixed: tuple
    learned: tuple
    uniform: tuple

    def probabilities(self, method, actor):
        f, l, u = MIXTURES[method]
        learned, uniform = dict(self.learned), dict(self.uniform)
        if actor not in uniform:
            raise ValueError('opponent absent from this forecast')
        lp, up = dict(learned[actor]), dict(uniform[actor])
        return {k: f*sum(dict(h)[actor] == k for h in self.fixed)/len(self.fixed) + l*lp[k] + u*p
                for k, p in up.items()}

    def occupancy(self, method, move):
        """Probability of any opposing INTENT, preserving F's joint dependence."""
        if not move.startswith('move:'):
            raise ValueError('movement destination required')
        f, l, u = MIXTURES[method]
        fixed = sum(move in dict(h).values() for h in self.fixed)/len(self.fixed)
        learned = 1-math.prod(1-dict(p).get(move, 0.) for _, p in self.learned)
        uniform = 1-math.prod(1-dict(p).get(move, 0.) for _, p in self.uniform)
        return f*fixed+l*learned+u*uniform


class PublicOpponentMemory:
    """A game-owned observer ID and explicit opponent IDs own all mutable state.

    Forecast then reveal exactly once per consecutive tick. A reveal must refer
    to the exact public state forecast beforehand. No external mutable handle
    is exposed, and invalid reveals leave both counts and pending state intact.
    """
    def __init__(self, scope, owner, opponents):
        if not isinstance(scope, str) or not scope or len(scope) > 128:
            raise ValueError('bounded game namespace required')
        if type(owner) is not int or not 0 <= owner < 6:
            raise ValueError('individual owner required')
        opponents = tuple(opponents)
        if not 1 <= len(opponents) <= MAX_OPPONENTS or len(set(opponents)) != len(opponents) or any(type(a) is not int or not 0 <= a < 6 or a == owner for a in opponents):
            raise ValueError('1..3 distinct opposing individuals required')
        self.scope, self.owner, self.opponents = scope, owner, opponents
        self._counts = {a: [[0.]*SIGNATURES for _ in range(CONTEXTS)] for a in opponents}
        self._tick = -1
        self._pending = None

    def _check(self, w, scope, owner):
        if (scope, owner) != (self.scope, self.owner):
            raise ValueError('game namespace or individual owner mismatch')
        if not 0 <= owner < len(w.units) or w.units[owner].hp <= 0:
            raise ValueError('living individual owner required')
        if any(a >= len(w.units) or w.units[a].team == w.units[owner].team for a in self.opponents):
            raise ValueError('opponent identity mismatch')
        if set(alive(w, 1-w.units[owner].team))-set(self.opponents):
            raise ValueError('unregistered opponent')
        if w.tick != self._tick+1:
            raise ValueError('consecutive public time required')

    def forecast(self, w, *, scope, owner):
        self._check(w, scope, owner)
        state_hash = digest(battle_record(w))
        if self._pending is not None and self._pending != (w.tick, state_hash):
            raise ValueError('public state differs from pending forecast')
        learned, uniform = [], []
        for a in alive(w, 1-w.units[owner].team):
            context, signatures = public_features(w, a)
            multiplicity = {s: sum(v == s for v in signatures.values()) for s in set(signatures.values())}
            counts = self._counts[a][context]
            denominator = PRIOR+sum(counts[s] for s in multiplicity)
            learned.append((a, tuple((k, (counts[s]/multiplicity[s]+PRIOR/len(signatures))/denominator) for k, s in signatures.items())))
            uniform.append((a, tuple((k, 1/len(signatures)) for k in signatures)))
        fixed = tuple(tuple(sorted(h.items())) for h in enemy_hypotheses(w, w.units[owner].team))
        self._pending = (w.tick, state_hash)
        return IntentForecast(scope, owner, w.tick, state_hash, fixed, tuple(learned), tuple(uniform))

    def reveal(self, w, actions, *, scope, owner):
        self._check(w, scope, owner)
        if self._pending != (w.tick, digest(battle_record(w))):
            raise ValueError('reveal requires exact preceding forecast state')
        required = alive(w, 1-w.units[owner].team)
        if set(actions) != set(required) or any(actions[a] not in legal(w, a) for a in required):
            raise ValueError('complete rule-legal revealed opponent intents required')
        # Validate the whole public event before touching any counts.
        features = {a: public_features(w, a) for a in required}
        for matrix in self._counts.values():
            for row in matrix:
                for s in range(SIGNATURES):
                    row[s] *= RETENTION
        for a, (context, signatures) in features.items():
            if len(signatures) > 1:  # Forced actions carry no behavioral choice evidence.
                self._counts[a][context][signatures[actions[a]]] += 1.
        self._tick = w.tick
        self._pending = None

    def record(self):
        return dict(scope=self.scope, owner=self.owner, opponents=list(self.opponents), last_tick=self._tick,
                    counts={str(a): [list(r) for r in rows] for a, rows in self._counts.items()}, pending=self._pending)
