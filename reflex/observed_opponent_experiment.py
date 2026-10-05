"""Frozen-trajectory, predict-score-reveal comparison; no live game decisions."""
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import math
import statistics
import time
from .combat import alive, legal, battle_from_record
from .congestion_experiment import jobs
from .continuation_experiment import sources
from .core import digest
from .observed_opponent import PublicOpponentMemory, METHODS

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT/'evidence/observed_opponent/design.json'
EPSILON = 1e-12


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def game_id(r):
    return '|'.join(str(r[k]) for k in ('partition', 'scenario', 'profile', 'seed'))


def binary_metrics(pairs):
    """Proper binary losses plus descriptive, fixed-bin calibration."""
    pairs = list(pairs)
    if not pairs:
        return dict(n=0, positives=0, brier=None, log_loss=None, infinite_log_losses=0,
                    clipped_log_loss=None, ece=None, bins=[])
    bins = [dict(n=0, probability_sum=0., positives=0) for _ in range(10)]
    brier = clipped = finite = 0.
    infinite = zero_misses = certain_false = 0
    for p, outcome in pairs:
        if not math.isfinite(p) or not 0 <= p <= 1 or outcome not in (0, 1):
            raise ValueError('finite probability and binary outcome required')
        likelihood = p if outcome else 1-p
        infinite += likelihood == 0
        if likelihood:
            finite -= math.log(likelihood)
        clipped -= math.log(max(EPSILON, likelihood))
        brier += (p-outcome)**2
        zero_misses += bool(outcome and p == 0)
        certain_false += bool(not outcome and p == 1)
        b = bins[min(9, int(p*10))]
        b['n'] += 1; b['probability_sum'] += p; b['positives'] += outcome
    calibration = [dict(lower=i/10, upper=(i+1)/10, n=b['n'],
        mean_probability=b['probability_sum']/b['n'] if b['n'] else None,
        observed_rate=b['positives']/b['n'] if b['n'] else None) for i, b in enumerate(bins)]
    return dict(n=len(pairs), positives=sum(y for _, y in pairs), brier=brier/len(pairs),
        log_loss=None if infinite else finite/len(pairs), infinite_log_losses=infinite,
        clipped_log_loss=clipped/len(pairs), clipping_epsilon=EPSILON,
        zero_probability_misses=zero_misses, certain_false_positives=certain_false,
        predicted_positive_mass=sum(p for p, _ in pairs),
        negative_probability_mass=sum(p for p, y in pairs if not y),
        ece=sum(abs(b['probability_sum']-b['positives']) for b in bins)/len(pairs), bins=calibration)


def intent_metrics(rows, method):
    rows = list(rows)
    calibration = binary_metrics((p, int(k == r['actual'])) for r in rows for k, p in r['probabilities'][method].items())
    if not rows:
        return dict(n=0, brier=None, log_loss=None, infinite_log_losses=0, calibration=calibration)
    probabilities = [r['probabilities'][method][r['actual']] for r in rows]
    infinite = sum(p == 0 for p in probabilities)
    return dict(n=len(rows), legal_action_negatives=sum(len(r['probabilities'][method])-1 for r in rows),
        brier=sum(sum((p-int(k == r['actual']))**2 for k, p in r['probabilities'][method].items()) for r in rows)/len(rows),
        log_loss=None if infinite else sum(-math.log(p) for p in probabilities)/len(rows),
        infinite_log_losses=infinite, clipped_log_loss=sum(-math.log(max(EPSILON, p)) for p in probabilities)/len(rows),
        clipping_epsilon=EPSILON, calibration=calibration)


def summarize(occupancy, intents):
    return {m: dict(occupancy=binary_metrics((r['probabilities'][m], int(r['actual'])) for r in occupancy),
        reachable_occupancy=binary_metrics((r['probabilities'][m], int(r['actual'])) for r in occupancy if r['enemy_reachable']),
        selected_occupancy=binary_metrics((r['probabilities'][m], int(r['actual'])) for r in occupancy if r['selected']),
        intent=intent_metrics(intents, m)) for m in METHODS}


def evaluate_run(run):
    team = run['seed'] % 2
    first = battle_from_record(run['trace'][0]['before'])
    scope = digest(game_id(run))
    memories = {i: PublicOpponentMemory(scope, i, alive(first, 1-team)) for i in alive(first, team)}
    occupancy, intents = [], []
    forecast_ms, reveal_ms, payload_bytes = [], [], []
    for row in run['trace']:
        w = battle_from_record(row['before'])
        owners = alive(w, team)
        # Construct ALL current forecasts before touching row['choices'].
        start = time.perf_counter()
        forecasts = {i: memories[i].forecast(w, scope=scope, owner=i) for i in owners}
        forecast_ms.append((time.perf_counter()-start)*1000)
        choices = {int(i): k for i, k in row['choices'].items()}
        actual = {a: choices[a] for a in alive(w, 1-team)}
        metadata = dict(game=game_id(run), partition=run['partition'], profile=run['profile'],
                        scenario=run['scenario'], rival=run['rival'], tick=w.tick)
        for i, forecast in forecasts.items():
            for k in legal(w, i):
                if not k.startswith('move:'):
                    continue
                occupancy.append(dict(**metadata, owner=i, action=k, actual=k in actual.values(),
                    enemy_reachable=any(k in legal(w, a) for a in actual), selected=choices[i] == k,
                    probabilities={m: forecast.occupancy(m, k) for m in METHODS}))
        # Same public history per live observer: one canonical observer per tick
        # avoids triplicating the categorical action event. Verify equality.
        canonical = forecasts[min(owners)]
        for a, k in actual.items():
            predicted = {m: canonical.probabilities(m, a) for m in METHODS}
            assert all(all(f.probabilities(m, a) == predicted[m] for m in METHODS) for f in forecasts.values())
            intents.append(dict(**metadata, opponent=a, actual=k, probabilities=predicted))
        start = time.perf_counter()
        for i in owners:
            memories[i].reveal(w, actual, scope=scope, owner=i)
        reveal_ms.append((time.perf_counter()-start)*1000)
        payload_bytes.extend(len(json.dumps(memories[i].record(), separators=(',', ':')).encode()) for i in owners)
    return occupancy, intents, dict(game=game_id(run), ticks=len(run['trace']),
        forecast_ms=forecast_ms, reveal_ms=reveal_ms, max_owner_json_bytes=max(payload_bytes))


def microbench(run):
    w = battle_from_record(run['trace'][0]['before']); team = run['seed'] % 2
    memories = {i: PublicOpponentMemory('timing', i, alive(w, 1-team)) for i in alive(w, team)}
    observations = []
    for repetition in range(220):
        start = time.perf_counter()
        for i, memory in memories.items():
            memory.forecast(w, scope='timing', owner=i)
        if repetition >= 20:
            observations.append((time.perf_counter()-start)*1000)
    ordered = sorted(observations)
    return dict(state=game_id(run), observers=len(memories), warmups=20, samples=200,
        all_observer_forecast_p50_ms=statistics.median(observations),
        all_observer_forecast_p95_ms=ordered[math.ceil(.95*len(ordered))-1],
        max_ms=max(observations), note='Includes repeated fixed hypotheses, feature construction and state hashing; no live decision/planner cost.')


def experiment(input_path, output):
    input_path, output = Path(input_path), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'preregister.json').exists():
        raise ValueError('use a fresh output; frozen preregistration is never overwritten')
    data_hash = sha(input_path)
    verification = json.loads((ROOT/'evidence/controlled_congestion/verification.json').read_text())
    expected_hash = verification['artifact_sha256']['trajectories.jsonl']
    if data_hash != expected_hash:
        raise ValueError('input differs from frozen controlled_congestion raw trajectories')
    rows = [json.loads(line) for line in input_path.read_text().splitlines()]
    runs = sorted((r for r in rows if r['variant'] == 'baseline'), key=game_id)
    expected = {(p, scene+'/'+goal, profile, seed) for p, scene, goal, rival, profile, seed, variant in jobs() if variant == 'baseline'}
    observed = [(r['partition'], r['scenario'], r['profile'], r['seed']) for r in runs]
    if len(observed) != 64 or set(observed) != expected:
        raise ValueError('exact frozen 64-game baseline matrix required')
    manifest = dict(format='observed-opponent-preregister-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        design_sha256=sha(DESIGN), source_hashes=sources(), input_sha256=data_hash,
        games=[game_id(r) for r in runs], methods=METHODS,
        test_sha256=sha(ROOT/'tests/test_observed_opponent.py'))
    write_json(output/'preregister.json', manifest)
    all_occupancy, all_intents, costs, per_game = [], [], [], []
    start = time.perf_counter()
    for run in runs:
        o, i, cost = evaluate_run(run)
        all_occupancy.extend(o); all_intents.extend(i); costs.append(cost)
        per_game.append(dict(game=game_id(run), partition=run['partition'], profile=run['profile'], metrics=summarize(o, i)))
    groups = []
    for partition in ('controlled', 'heldout'):
        o = [r for r in all_occupancy if r['partition'] == partition]
        i = [r for r in all_intents if r['partition'] == partition]
        groups.append(dict(partition=partition, dimension='all', value='all', metrics=summarize(o, i)))
        for dimension in ('profile', 'scenario', 'rival'):
            for value in sorted({r[dimension] for r in i}):
                groups.append(dict(partition=partition, dimension=dimension, value=value,
                    metrics=summarize([r for r in o if r[dimension] == value], [r for r in i if r[dimension] == value])))
    for phase, predicate in [('before_switch', lambda r: r['tick'] < 8), ('after_switch', lambda r: r['tick'] >= 8)]:
        groups.append(dict(partition='heldout', dimension='switch_phase', value=phase,
            metrics=summarize([r for r in all_occupancy if r['partition'] == 'heldout' and r['rival'] == 'switch' and predicate(r)],
                              [r for r in all_intents if r['partition'] == 'heldout' and r['rival'] == 'switch' and predicate(r)])))
    failures = []
    for group in groups:
        if group['dimension'] != 'all':
            continue
        m = group['metrics']
        for baseline in ('cold_memory', 'uniform'):
            for metric in ('brier', 'log_loss'):
                if not m['observed_memory']['occupancy'][metric] < m[baseline]['occupancy'][metric]:
                    failures.append(f"{group['partition']} occupancy {metric} does not improve on {baseline}")
    macro = []
    for partition in ('controlled', 'heldout'):
        rows = [r for r in per_game if r['partition'] == partition]
        for baseline in ('uniform', 'fixed_coverage', 'cold_memory'):
            deltas = [r['metrics']['observed_memory']['occupancy']['brier']-r['metrics'][baseline]['occupancy']['brier'] for r in rows]
            macro.append(dict(partition=partition, baseline=baseline, games=len(rows),
                mean_game_brier_delta=statistics.mean(deltas), improved=sum(d < -1e-12 for d in deltas),
                worsened=sum(d > 1e-12 for d in deltas), unchanged=sum(abs(d) <= 1e-12 for d in deltas)))
    result = dict(format='observed-opponent-evaluation-v1', games=len(runs), ticks=sum(c['ticks'] for c in costs),
        scope='Offline frozen-path prediction; no changes to policy, persona or live experience',
        occupancy_rows=len(all_occupancy), intent_rows=len(all_intents), groups=groups, macro=macro,
        signal_criteria_passed=not failures, signal_failures=failures, integrate=False,
        elapsed_seconds=time.perf_counter()-start, cost=dict(
            forecast_tick_p50_ms=statistics.median(t for c in costs for t in c['forecast_ms']),
            reveal_tick_p50_ms=statistics.median(t for c in costs for t in c['reveal_ms']),
            max_owner_json_bytes=max(c['max_owner_json_bytes'] for c in costs),
            numeric_counters_per_owner=312, numeric_payload_bytes_per_owner=2496),
        microbench=microbench(next(r for r in runs if r['partition'] == 'controlled')))
    assert manifest['source_hashes'] == sources() and manifest['input_sha256'] == sha(input_path)
    assert manifest['design_sha256'] == sha(DESIGN)
    result['frozen_inputs_unchanged'] = True
    write_json(output/'evaluation.json', result)
    write_json(output/'per_game.json', per_game)
    write_json(output/'cost.json', costs)
    raw = output/'predictions.jsonl'
    with raw.open('w') as stream:
        for kind, events in (('occupancy', all_occupancy), ('intent', all_intents)):
            for event in events:
                stream.write(json.dumps(dict(kind=kind, **event), allow_nan=False)+'\n')
    write_json(output/'verification.json', dict(input_sha256=data_hash, predictions_sha256=sha(raw),
        design_sha256=sha(DESIGN), source_hashes=sources(), games=64, no_new_games=True,
        same_input_methods=list(METHODS), source_and_data_unchanged=True))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = experiment(args.input, args.output)
    print(json.dumps({k: result[k] for k in ('games', 'ticks', 'occupancy_rows', 'intent_rows', 'signal_failures', 'cost', 'microbench')}, indent=2))
