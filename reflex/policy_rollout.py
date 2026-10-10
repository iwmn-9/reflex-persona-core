"""Paired closed-loop rollouts with uncompressed terminal samples.

The adapter owns its hypothetical world and detached actor state. Its choose()
must pass only public observations and the owner's state to the base policy.
This is simulation under a declared base policy, not an optimality theorem.
"""
from dataclasses import dataclass
from .monte_carlo import evaluate_actions


@dataclass(frozen=True)
class TaggedState:
    root: str
    state: object


def evaluate_policy(roots, adapter, budget, scope):
    """Return matched terminal samples; never retain partial sample rounds.

The ordinary Monte Carlo budget still excludes work inside policy callbacks.
Only completed, root-balanced rounds can be used to choose a real action.
"""
    records={name:[] for name in roots}

    class Recording:
        def begin_trial(self): adapter.begin_trial()
        def terminal(self, s): return adapter.terminal(s.state)
        def chance(self, s): return adapter.chance(s.state)
        def legal(self, s): return adapter.legal(s.state)
        def sample(self, s, rng): return TaggedState(s.root,adapter.sample(s.state,rng))
        def step(self, s, a): return TaggedState(s.root,adapter.step(s.state,a))
        def choose(self, s, rng, policy): return adapter.choose(s.state,rng,policy)
        def evaluate(self, s):
            result=adapter.evaluate(s.state)
            records[s.root].append(result)
            return result

    outcomes, stats=evaluate_actions({name:TaggedState(name,s) for name,s in roots.items()},
                                   Recording(),budget,scope)
    completed=stats['completed_samples']
    samples={name:tuple(values[:completed]) for name,values in records.items()}
    assert all(len(values)==completed for values in samples.values())
    return outcomes,samples,stats
