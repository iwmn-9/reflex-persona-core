"""Experimental public-only, coherent goal/coverage samples for tactical plans.

No learned goal weights. Each sample's broad-coverage mode and each opponent's
purpose stay fixed across the modeled path. Actions respond to modeled public
states using stateless private variates, shared across candidate own plans.
"""
from dataclasses import dataclass
from .combat import alive, legal, battle_record
from .combat_planning import routes, opponent_intents
from .core import digest
from .goal_opponent import goal_models, COVERAGE
from .goal_beliefs import SMOOTHING


def _unit(key):
    # Exactly representable [0,1) variate, used only inside hypothetical models.
    return (int(digest(key)[:16], 16) >> 11) / (1 << 53)


def _sample(probabilities, value):
    total = 0.
    for action, p in sorted(probabilities.items()):
        total += p
        if value < total: return action
    return sorted(probabilities)[-1]  # harmless rounding of normalized mass


@dataclass(frozen=True, init=False)
class GoalRollout:
    root_hash: str
    root_tick: int
    team: int
    sample: int
    method: str
    samples: int
    broad: bool
    goals: tuple

    def __init__(self, w, team, sample, method, samples=4):
        if type(team) is not int or team not in (0, 1) or type(samples) is not int or not 1 <= samples <= 8 or type(sample) is not int or not 0 <= sample < samples:
            raise ValueError('bounded public team and planning sample required')
        if method not in ('uniform-coverage', 'goal-uniform'):
            raise ValueError('known experimental opponent model required')
        root_hash = digest(battle_record(w))
        names = routes(w, 1-team) or ('secure',)
        # Systematic strata retain uniform one-sample marginals while avoiding
        # four identical purpose draws for an individual under a two-goal game.
        phase = sample/samples
        broad = (_unit(['public-goal-rollout', root_hash, 'coverage'])+phase) % 1 < COVERAGE
        goals = tuple((a, names[min(len(names)-1, int(((_unit(['public-goal-rollout', root_hash, 'goal', a])+phase) % 1)*len(names)))]) for a in alive(w, 1-team))
        for key, value in dict(root_hash=root_hash, root_tick=w.tick, team=team, sample=sample,
                               method=method, samples=samples, broad=broad, goals=goals).items():
            object.__setattr__(self, key, value)

    def choose(self, model, depth):
        if type(depth) is not int or not 0 <= depth < 16 or model.tick != self.root_tick+depth:
            raise ValueError('matching bounded model depth required')
        if depth == 0 and digest(battle_record(model)) != self.root_hash:
            raise ValueError('root public state differs from sampled state')
        enemies = alive(model, 1-self.team)
        if set(enemies)-set(dict(self.goals)):
            raise ValueError('unregistered modeled opponent')
        if not self.broad and self.method == 'uniform-coverage':
            return opponent_intents(model, self.team, self.sample % 2 == 1)
        allowed = routes(model, 1-self.team) or ('secure',)
        choices = {}
        for a in enemies:
            keys = legal(model, a)
            if not keys:
                raise ValueError('nonterminal modeled state required')
            if self.broad:
                probabilities = {k: 1/len(keys) for k in keys}
            else:
                goal = dict(self.goals)[a]
                if goal not in allowed: goal = allowed[0]
                expert = goal_models(model, a)[goal]
                probabilities = {k: (1-SMOOTHING)*p+SMOOTHING/len(keys) for k, p in expert.items()}
            value = _unit(['public-goal-rollout', self.root_hash, self.sample, 'action', depth, a])
            choices[a] = _sample(probabilities, value)
        return choices

    def record(self):
        return dict(method=self.method, root_hash=self.root_hash, sample=self.sample, samples=self.samples,
            broad=self.broad, goals={str(a): g for a, g in self.goals},
            note='Private model sample from public state; not actual opponent goals or actions')
