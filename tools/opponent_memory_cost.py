"""Measure retained Python bookkeeping separately from packed numeric payload."""
from dataclasses import replace
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reflex.combat import Battle
from reflex.observed_opponent import PublicOpponentMemory


def deep_size(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(deep_size(k, seen)+deep_size(v, seen) for k, v in value.items())
    elif isinstance(value, (list, tuple)):
        size += sum(deep_size(v, seen) for v in value)
    elif hasattr(value, '__dict__'):
        size += deep_size(value.__dict__, seen)
    return size


def measure():
    w = Battle.start('open', 'both', limit=100)
    m = PublicOpponentMemory('x'*128, 0, (3, 4, 5))
    for tick in range(20):
        before = replace(w, tick=tick)
        m.forecast(before, scope='x'*128, owner=0)
        m.reveal(before, {3: 'guard', 4: 'guard', 5: 'guard'}, scope='x'*128, owner=0)
    after_reveal = deep_size(m)
    m.forecast(replace(w, tick=20), scope='x'*128, owner=0)
    return dict(format='observed-opponent-memory-cost-v1', python=sys.version,
        scope='Isolated bookkeeping fixture, no new evaluated gameplay or tuning', counters=312,
        packed_float64_payload_bytes=2496, retained_python_bytes_after_reveal=after_reveal,
        retained_python_bytes_pending_forecast=deep_size(m),
        note='Includes Python object/list/dict/float overhead recursively once per identity and maximal 128-character scope. Not a claim of 2496-byte resident memory. Input world, returned immutable snapshots and shared imported caches excluded; predictor retains no snapshots. Counter/table dimensions are fixed.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='evidence/observed_opponent/memory_cost.json')
    args = p.parse_args()
    Path(args.output).write_text(json.dumps(measure(), indent=2)+'\n')
