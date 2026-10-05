"""Serial, isolated observer costs; not a real-time throughput guarantee."""
import argparse
from collections import deque
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import math
import statistics
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reflex.combat import legal
from reflex.congestion_experiment import authored_world, enemy_hypotheses
from reflex.goal_beliefs import GoalBeliefs
from reflex.goal_opponent import GoalOpponentMemory, goal_models


def deep_size(obj, seen=None):
    seen = set() if seen is None else seen
    if id(obj) in seen: return 0
    seen.add(id(obj)); total = sys.getsizeof(obj)
    if isinstance(obj, dict):
        total += sum(deep_size(k, seen)+deep_size(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset, deque)):
        total += sum(deep_size(x, seen) for x in obj)
    elif hasattr(obj, '__dict__'):
        total += deep_size(vars(obj), seen)
    return total


def measure(fn, warmups=20, samples=200):
    for _ in range(warmups): fn()
    times = []
    for _ in range(samples):
        start = time.perf_counter(); fn(); times.append((time.perf_counter()-start)*1000)
    return dict(warmups=warmups, samples=samples, p50_ms=statistics.median(times),
        p95_ms=sorted(times)[math.ceil(.95*len(times))-1], max_ms=max(times))


def benchmark():
    w = authored_world('enemy_convergence', 'both', 0)
    memories = [GoalOpponentMemory('cost', i, (3, 4, 5)) for i in (0, 1, 2)]
    for tick in range(16):
        before = replace(w, tick=tick)
        for i, m in enumerate(memories):
            m.forecast(before, scope='cost', owner=i)
            m.reveal(before, {3: 'move:4:1', 4: 'guard', 5: 'guard'}, scope='cost', owner=i)
    w = replace(w, tick=16)
    retained = deep_size(memories[0])
    own_moves = {i: tuple(k for k in legal(w, i) if k.startswith('move:')) for i in range(3)}
    def original():
        for i in range(3):
            fixed = enemy_hypotheses(w, 0)
            for move in own_moves[i]:
                sum(move in h.values() for h in fixed)/len(fixed)
    def candidate():
        for i, m in enumerate(memories):
            f = m.forecast(w, scope='cost', owner=i)
            for move in own_moves[i]: f.occupancy('goal_adaptive', move)
    baseline = measure(original)
    full = measure(candidate)
    pending = deep_size(memories[0])
    # Generic likelihood core only. Each observer belongs to a distinct game;
    # no source geometry calculations or combat forecasts are included here.
    models = {str(a): goal_models(w, a) for a in (3, 4, 5)}
    active = {a: ('eliminate', 'secure') for a in models}
    scaling = []
    for count in (1, 32, 128):
        observers = [GoalBeliefs(f'replica-{i}', 'owner', tuple(models), ('eliminate', 'secure')) for i in range(count)]
        def batch():
            for i, m in enumerate(observers):
                f = m.forecast(0, 'fixed-public-state', models, active=active, scope=f'replica-{i}', owner='owner')
                for a in models: f.probabilities(a)
        measured = measure(batch, warmups=5, samples=50)
        scaling.append(dict(independent_observers=count, opponents_per_observer=3, **measured,
            p50_ms_per_observer=measured['p50_ms']/count))
    return dict(format='goal-opponent-cost-v1', fixture='enemy_convergence/both/seat0, fixed public state after 16 synthetic legal-intent reveals',
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path('reflex/goal_beliefs.py'), Path('reflex/goal_opponent.py')]},
        core_numeric_shape=dict(opponents=3, goals=2, retained_log_weights=6, maximum_retained_observation_ids=192, maximum_retained_lifts=18),
        recursive_python_bytes=dict(one_trained_observer_without_pending=retained, one_trained_observer_with_pending=pending),
        three_observer_original_fixed_forecast_and_all_legal_move_queries=baseline,
        three_observer_goal_forecast_and_all_legal_move_queries=full,
        same_state_p50_ratio=full['p50_ms']/baseline['p50_ms'],
        generic_core_scaling=scaling,
        caveats=['Serial shared CPU timing, not a speed guarantee or live gameplay improvement.',
                 'Synthetic reveal history is a cost fixture, not legal consecutive game transitions or efficacy evidence.',
                 'Original baseline computes only its fixed joint hypotheses. Candidate includes original controls, two soft goal experts, state hash, immutable forecasts and goal occupancy queries.',
                 'Recursive size excludes shared Python code, geometry caches, returned transient forecasts outside retained pending state and caller-owned source models.',
                 'Generic replicas test isolated observer inference only, not 128 actors sharing one combat world, planner costs, or end-to-end game throughput.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True); args = parser.parse_args()
    Path(args.output).write_text(json.dumps(benchmark(), indent=2, allow_nan=False)+'\n')
