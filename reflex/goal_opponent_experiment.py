"""Preregister fresh baselines, then score immutable public forecasts offline.

This harness never supplies prediction results to the baseline planner. The
prediction phase reads current choices only after every living owner's forecast.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import statistics
import subprocess
import time

from .combat import Battle, alive, legal, battle_from_record, terminal
from .combat_planning import TacticalControl
from .congestion_experiment import SCENES, HELDOUT, authored_world, enemy_hypotheses, jobs as old_jobs
from .continuation_experiment import sources
from .core import digest
from .goal_opponent import GoalOpponentMemory, METHODS
from .laboratory import profiles
from .observed_opponent_experiment import binary_metrics, intent_metrics
from .validation_experiment import combat, replay, replay_progress

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / 'evidence/goal_opponent/design.json'
BASE = 'aedb130b0ed7d2fce1f1abaa0b214d5ca313e299'
SEEDS = (1701, 1702)
PARTITIONS = ('controlled', 'heldout')
DOMAINS = ('occupancy', 'reachable_occupancy', 'selected_occupancy', 'intent')
PRIMARY_DOMAINS = ('reachable_occupancy', 'intent')
CONTROLS = ('uniform', 'fixed_coverage')


def write_json(path, value, exclusive=False):
    with Path(path).open('x' if exclusive else 'w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def game_id(run):
    return '|'.join(str(run[k]) for k in ('partition', 'scenario', 'profile', 'seed'))


def jobs():
    """One seat-balanced controlled seed per cell; both seats in heldout."""
    result = []
    for cell, (scene, goal) in enumerate((s, g) for s in SCENES for g in ('secure', 'both')):
        for index, profile in enumerate(profiles()):
            result.append(dict(partition='controlled', scene=scene, goal=goal,
                rival='reference', profile=profile['id'], seed=SEEDS[(cell + index) % 2], limit=24))
    for scene, goal, rival in HELDOUT:
        for profile in profiles():
            for seed in SEEDS:
                result.append(dict(partition='heldout', scene=scene, goal=goal,
                    rival=rival, profile=profile['id'], seed=seed, limit=40))
    for row in result:
        row.update(scenario=row['scene'] + '/' + row['goal'], variant='baseline')
        row['game'] = game_id(row)
    old = {'|'.join(map(str, (p, scene + '/' + goal, profile, seed)))
           for p, scene, goal, _, profile, seed, variant in old_jobs() if variant == 'baseline'}
    ids = [row['game'] for row in result]
    if len(ids) != 56 or len(set(ids)) != 56 or set(ids) & old:
        raise ValueError('fresh, nonoverlapping fixed 56-game matrix required')
    return result


def frozen_files():
    """Include all test/source files and external initial-geometry dependency."""
    files = [DESIGN, ROOT / 'evidence/controlled_congestion/design.json', ROOT / 'pyproject.toml', ROOT / 'tools/goal_opponent_cost.py']
    files += sorted((ROOT / 'tests').glob('*.py'))
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in files}


def configuration():
    return TacticalControl(recovery_options=True, trace_execution=True)


def preregister(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('preregister in a fresh empty output directory')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    manifest = dict(format='goal-opponent-preregister-v1', base=BASE, checkout_head=head,
        created_utc=datetime.now(timezone.utc).isoformat(), jobs=jobs(), methods=list(METHODS),
        planner=asdict(configuration()), source_hashes=sources(), file_hashes=frozen_files(),
        design_sha256=sha(DESIGN), forecasts_enter_live_decisions=False,
        observation='Fresh seeds, familiar authored scenes and driver families; not unseen environments or externally validated generalization')
    write_json(output / 'preregister.json', manifest, exclusive=True)
    return manifest


def check_frozen(output):
    manifest = json.loads((Path(output) / 'preregister.json').read_text())
    if manifest['source_hashes'] != sources() or manifest['file_hashes'] != frozen_files():
        raise ValueError('source, tests or design changed after preregistration; do not tune or silently refreeze')
    if manifest['jobs'] != jobs() or manifest['methods'] != list(METHODS) or manifest['planner'] != asdict(configuration()):
        raise ValueError('preregistered matrix, methods or planner changed')
    return manifest


def run_job(job):
    """Existing planner, fixed original profile, actual-only experience feedback."""
    from . import combat_planning as cp
    profile = next(p for p in profiles() if p['id'] == job['profile'])
    original_profile = copy.deepcopy(profile)
    initial = (authored_world(job['scene'], job['goal'], job['seed'] % 2, job['limit'])
        if job['partition'] == 'controlled' else Battle.start(job['scene'], job['goal'], limit=job['limit']))
    original_forecast = cp.forecast
    cost = dict(planner_seconds=0., planner_calls=0, model_transitions=0)
    def measured(*args, **kwargs):
        start = time.perf_counter()
        result = original_forecast(*args, **kwargs)
        cost['planner_seconds'] += time.perf_counter() - start
        cost['planner_calls'] += 1
        cost['model_transitions'] += result.audit['nodes']
        return result
    cp.forecast = measured
    try:
        start = time.perf_counter()
        result = combat(profile, job['seed'], job['scene'], job['goal'], 'baseline',
            learn=True, survival_security=True, rival=job['rival'], planner=configuration(),
            progress_watch=True, initial_world=initial, trace_roots=True)
        cost['game_seconds'] = time.perf_counter() - start
    finally:
        cp.forecast = original_forecast
    if profile != original_profile:
        raise AssertionError('fixed profile mutated')
    result.update(partition=job['partition'], rival=job['rival'], variant='baseline',
        game=job['game'], baseline_cost=cost)
    checks = verify_run(result, job)
    return result, checks


def verify_run(run, expected=None):
    """Rules, causal continuity, legal intents and original saved hypotheses."""
    if not run['trace'] or run['variant'] != 'baseline':
        raise ValueError('complete nonempty baseline trajectory required')
    if expected:
        for key in ('partition', 'scenario', 'profile', 'seed', 'rival', 'variant'):
            if run[key] != expected[key]:
                raise ValueError('unexpected trajectory condition: ' + key)
        initial = (authored_world(expected['scene'], expected['goal'], expected['seed'] % 2, expected['limit'])
            if expected['partition'] == 'controlled' else Battle.start(expected['scene'], expected['goal'], limit=expected['limit']))
        if battle_from_record(run['trace'][0]['before']) != initial:
            raise ValueError('initial state differs from preregistered world')
    legal_checks = hypothesis_checks = 0
    previous = None
    for row in run['trace']:
        w = battle_from_record(row['before'])
        if previous is not None and previous != w:
            raise ValueError('discontinuous public trajectory')
        choices = {int(i): k for i, k in row['choices'].items()}
        if set(choices) != set(alive(w, 0) + alive(w, 1)):
            raise ValueError('missing or extra actual intents')
        for actor, action in choices.items():
            if action not in legal(w, actor):
                raise ValueError('illegal actual intent')
            legal_checks += 1
        execution = row.get('planning', {}).get('execution')
        if execution:
            fixed = enemy_hypotheses(w, run['seed'] % 2)
            for index, sample in enumerate(execution['samples']):
                if sample:
                    if sample[0]['opponent'] != {str(i): k for i, k in fixed[index % 2].items()}:
                        raise ValueError('existing first-step hypothesis mismatch')
                    hypothesis_checks += 1
        previous = battle_from_record(row['after'])
    if not terminal(previous) or previous.tick != run['ticks'] or run['ticks'] != len(run['trace']):
        raise ValueError('truncated or invalid terminal trajectory')
    return dict(rule_transitions=replay(run), progress_updates=replay_progress(run),
        legal_intents=legal_checks, fixed_hypothesis_checks=hypothesis_checks)


def run_baselines(output, workers=4):
    output = Path(output)
    manifest = check_frozen(output)
    if not 1 <= workers <= 8:
        raise ValueError('workers must be in [1, 8]')
    raw_path = output / 'trajectories.jsonl'
    if raw_path.exists() or (output / 'baseline_verification.json').exists():
        raise FileExistsError('baseline artifacts are never overwritten or selectively rerun')
    start = time.perf_counter()
    checks, results = [], []
    with raw_path.open('x') as stream, ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(run_job, job): job for job in manifest['jobs']}
        for future in as_completed(pending):
            result, verification = future.result()
            stream.write(json.dumps(result, allow_nan=False) + '\n')
            stream.flush()
            checks.append(dict(game=game_id(result), **verification))
            results.append({k: v for k, v in result.items() if k != 'trace'})
            print(f'{len(results)}/{len(pending)} {game_id(result)} {result["baseline_cost"]["game_seconds"]:.1f}s', flush=True)
    check_frozen(output)
    verification = dict(format='goal-opponent-baselines-v1', games=len(results),
        input_sha256=sha(raw_path), checks=sorted(checks, key=lambda r: r['game']),
        elapsed_seconds=time.perf_counter() - start, frozen_unchanged=True,
        complete=set(r['game'] for r in results) == set(j['game'] for j in manifest['jobs']))
    if not verification['complete']:
        raise ValueError('incomplete fresh baseline matrix')
    write_json(output / 'baseline_runs.json', sorted(results, key=lambda r: r['game']), exclusive=True)
    write_json(output / 'baseline_verification.json', verification, exclusive=True)
    return verification


def summarize(occupancy, intents):
    return {method: dict(
        occupancy=binary_metrics((r['probabilities'][method], int(r['actual'])) for r in occupancy),
        reachable_occupancy=binary_metrics((r['probabilities'][method], int(r['actual'])) for r in occupancy if r['enemy_reachable']),
        selected_occupancy=binary_metrics((r['probabilities'][method], int(r['actual'])) for r in occupancy if r['selected']),
        intent=intent_metrics(intents, method)) for method in METHODS}


def evaluate_run(run):
    team = run['seed'] % 2
    first = battle_from_record(run['trace'][0]['before'])
    scope = digest(['fresh-goal-prediction', game_id(run)])
    memories = {i: GoalOpponentMemory(scope, i, alive(first, 1-team)) for i in alive(first, team)}
    occupancy, intents, beliefs = [], [], []
    forecast_ms, reveal_ms, payload_bytes = [], [], []
    for row in run['trace']:
        w = battle_from_record(row['before'])
        owners = alive(w, team)
        # Strict boundary: all forecasts exist before current intents are read.
        start = time.perf_counter()
        forecasts = {i: memories[i].forecast(w, scope=scope, owner=i) for i in owners}
        forecast_ms.append((time.perf_counter() - start)*1000)
        prebelief = memories[min(owners)].record()
        choices = {int(i): k for i, k in row['choices'].items()}
        actual = {i: choices[i] for i in alive(w, 1-team)}
        metadata = dict(game=game_id(run), partition=run['partition'], scenario=run['scenario'],
            profile=run['profile'], rival=run['rival'], tick=w.tick)
        for owner, forecast in forecasts.items():
            for action in legal(w, owner):
                if action.startswith('move:'):
                    occupancy.append(dict(**metadata, owner=owner, action=action,
                        enemy_reachable=any(action in legal(w, actor) for actor in actual),
                        selected=choices[owner] == action, actual=action in actual.values(),
                        probabilities={m: forecast.occupancy(m, action) for m in METHODS}))
        canonical = forecasts[min(owners)]
        for actor, action in actual.items():
            predicted = {m: dict(canonical.probabilities(m, actor)) for m in METHODS}
            for m, distribution in predicted.items():
                if set(distribution) != set(legal(w, actor)) or not math.isclose(sum(distribution.values()), 1., abs_tol=1e-12):
                    raise AssertionError('prediction must cover exactly the full legal action set')
                if any(not math.isfinite(p) or p < 0 or p > 1 for p in distribution.values()):
                    raise AssertionError('invalid probability')
                if any(dict(f.probabilities(m, actor)) != distribution for f in forecasts.values()):
                    raise AssertionError('identical public-history observers disagree')
            intents.append(dict(**metadata, opponent=actor, actual=action, probabilities=predicted))
        start = time.perf_counter()
        for owner in owners:
            memories[owner].reveal(w, actual, scope=scope, owner=owner)
        reveal_ms.append((time.perf_counter() - start)*1000)
        beliefs.append(dict(**metadata, owner=min(owners), forecast=canonical.record(),
            before=prebelief, after=memories[min(owners)].record()))
        payload_bytes.extend(len(json.dumps(memories[i].record(), separators=(',', ':')).encode()) for i in owners)
    return occupancy, intents, beliefs, dict(game=game_id(run), ticks=len(run['trace']),
        forecast_ms=forecast_ms, reveal_ms=reveal_ms, max_owner_json_bytes=max(payload_bytes))


def equal_game_summary(rows):
    """A game with no reachable candidates is absent, never a perfect zero."""
    result = {}
    for method in METHODS:
        result[method] = {}
        for domain in DOMAINS:
            metrics = [row['metrics'][method][domain] for row in rows if row['metrics'][method][domain]['n']]
            infinite_games = sum(m['infinite_log_losses'] > 0 for m in metrics)
            result[method][domain] = dict(games=len(metrics), empty_games=len(rows)-len(metrics),
                brier=statistics.mean(m['brier'] for m in metrics) if metrics else None,
                log_loss=statistics.mean(m['log_loss'] for m in metrics) if metrics and not infinite_games else None,
                infinite_log_loss_games=infinite_games,
                clipped_log_loss=statistics.mean(m['clipped_log_loss'] for m in metrics) if metrics else None)
    return result


def paired_differences(rows, control):
    result = {}
    for domain in DOMAINS:
        result[domain] = {}
        for metric in ('brier', 'log_loss'):
            pairs = [(r['metrics']['goal_adaptive'][domain][metric], r['metrics'][control][domain][metric]) for r in rows]
            differences = [a-b for a, b in pairs if a is not None and b is not None]
            result[domain][metric] = dict(games=len(differences), unavailable_games=len(rows)-len(differences),
                mean_delta=statistics.mean(differences) if differences else None,
                improved=sum(d < -1e-12 for d in differences), worsened=sum(d > 1e-12 for d in differences),
                unchanged=sum(abs(d) <= 1e-12 for d in differences))
    return result


def primary_gate(groups):
    failures = []
    checks = []
    for partition in PARTITIONS:
        group = next(g for g in groups if g['partition'] == partition and g['dimension'] == 'all')
        for aggregation in ('metrics', 'equal_game'):
            for control in CONTROLS:
                for domain in PRIMARY_DOMAINS:
                    for metric in ('brier', 'log_loss'):
                        a = group[aggregation]['goal_adaptive'][domain][metric]
                        b = group[aggregation][control][domain][metric]
                        passed = a is not None and b is not None and a < b
                        label = '/'.join((partition, aggregation, domain, metric, control))
                        checks.append(dict(comparison=label, candidate=a, control=b, passed=passed))
                        if not passed:
                            failures.append(label)
    return dict(primary_gate_passed=not failures, primary_gate_failures=failures, primary_gate_checks=checks,
        exploratory_live_eligible=not failures, live_gameplay_performed=False, integrate=False,
        decision='eligible for separately reviewed exploratory paired gameplay only' if not failures else 'stop at offline prediction; no live gameplay or integration')


def groups_for(occupancy, intents, per_game):
    groups = []
    for partition in PARTITIONS:
        games = [r for r in per_game if r['partition'] == partition]
        for dimension in ('all', 'profile', 'scenario', 'rival'):
            values = ['all'] if dimension == 'all' else sorted({r[dimension] for r in games})
            for value in values:
                selected = [r for r in games if dimension == 'all' or r[dimension] == value]
                ids = {r['game'] for r in selected}
                groups.append(dict(partition=partition, dimension=dimension, value=value,
                    games=len(selected), metrics=summarize([r for r in occupancy if r['game'] in ids], [r for r in intents if r['game'] in ids]),
                    equal_game=equal_game_summary(selected),
                    adaptive_differences={control: paired_differences(selected, control) for control in (*CONTROLS, 'goal_uniform')}))
    for phase, predicate in [('before_switch', lambda r: r['tick'] < 8), ('after_switch', lambda r: r['tick'] >= 8)]:
        groups.append(dict(partition='heldout', dimension='switch_phase', value=phase,
            metrics=summarize([r for r in occupancy if r['partition'] == 'heldout' and r['rival'] == 'switch' and predicate(r)],
                              [r for r in intents if r['partition'] == 'heldout' and r['rival'] == 'switch' and predicate(r)])))
    return groups


def microbench(run):
    w = battle_from_record(run['trace'][0]['before'])
    team = run['seed'] % 2
    memories = {i: GoalOpponentMemory('timing', i, alive(w, 1-team)) for i in alive(w, team)}
    values = []
    for repetition in range(220):
        start = time.perf_counter()
        for i, memory in memories.items():
            memory.forecast(w, scope='timing', owner=i)
        if repetition >= 20:
            values.append((time.perf_counter()-start)*1000)
    return dict(game=game_id(run), living_observers=len(memories), warmups=20, samples=200,
        all_observer_forecast_p50_ms=statistics.median(values),
        all_observer_forecast_p95_ms=sorted(values)[math.ceil(.95*len(values))-1], max_ms=max(values),
        note='Serial fixed cold state; all methods and original hypotheses computed together, no live planner overhead comparison')


def evaluate(output):
    output = Path(output)
    manifest = check_frozen(output)
    raw_path = output / 'trajectories.jsonl'
    baseline = json.loads((output / 'baseline_verification.json').read_text())
    if not baseline['complete'] or baseline['input_sha256'] != sha(raw_path):
        raise ValueError('baseline input missing, incomplete or changed')
    for name in ('evaluation.json', 'per_game.json', 'cost.json', 'predictions.jsonl', 'beliefs.jsonl', 'verification.json'):
        if (output / name).exists():
            raise FileExistsError('offline evidence is never overwritten: ' + name)
    runs = sorted([json.loads(line) for line in raw_path.read_text().splitlines()], key=game_id)
    expected = {job['game']: job for job in manifest['jobs']}
    ids = [game_id(run) for run in runs]
    if len(ids) != len(expected) or set(ids) != set(expected):
        raise ValueError('exact fresh 56-game baseline matrix required')
    occupancy, intents, beliefs, costs, per_game, checks = [], [], [], [], [], []
    start = time.perf_counter()
    for run in runs:
        checks.append(dict(game=game_id(run), **verify_run(run, expected[game_id(run)])))
        o, i, b, cost = evaluate_run(run)
        occupancy.extend(o); intents.extend(i); beliefs.extend(b); costs.append(cost)
        per_game.append(dict(game=game_id(run), partition=run['partition'], profile=run['profile'],
            scenario=run['scenario'], rival=run['rival'], metrics=summarize(o, i)))
    groups = groups_for(occupancy, intents, per_game)
    result = dict(format='goal-opponent-evaluation-v1', games=len(runs), ticks=sum(c['ticks'] for c in costs),
        occupancy_rows=len(occupancy), intent_rows=len(intents), groups=groups, **primary_gate(groups),
        notes=['Occupancy labels are attempted intents, not post-resolution positions',
               'All legal-cell negatives included; reachable subset prevents structural negatives hiding error',
               'Per-game means give equal weight only to games with nonempty domains; correlated rows are not independent samples',
               'Null true log loss with a positive infinite count means infinity, never a clipped finite score',
               'Adaptive versus goal_uniform isolates likelihood weighting from goal support',
               'Heldout means fresh seeds in familiar scenarios and controller families; no retuning'],
        prediction_and_replay_seconds=time.perf_counter()-start,
        cost=dict(forecast_tick_p50_ms=statistics.median(t for c in costs for t in c['forecast_ms']),
            reveal_tick_p50_ms=statistics.median(t for c in costs for t in c['reveal_ms']),
            max_owner_json_bytes=max(c['max_owner_json_bytes'] for c in costs),
            note='Serialized memory payload is not recursive Python heap size; baseline generation and planner work are reported separately'),
        microbench=microbench(next(run for run in runs if run['partition'] == 'controlled')))
    check_frozen(output)
    if sha(raw_path) != baseline['input_sha256']:
        raise ValueError('raw trajectories changed during scoring')
    write_json(output / 'evaluation.json', result, exclusive=True)
    write_json(output / 'per_game.json', per_game, exclusive=True)
    write_json(output / 'cost.json', costs, exclusive=True)
    for name, rows in [('predictions.jsonl', [dict(kind='occupancy', **r) for r in occupancy] + [dict(kind='intent', **r) for r in intents]),
                       ('beliefs.jsonl', beliefs)]:
        with (output / name).open('x') as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
    verification = dict(format='goal-opponent-verification-v1', games=len(runs), fresh_game_ids=ids,
        artifact_sha256={name: sha(output / name) for name in ('trajectories.jsonl', 'predictions.jsonl', 'beliefs.jsonl', 'evaluation.json', 'per_game.json', 'cost.json')},
        source_hashes=sources(), file_hashes=frozen_files(), replay_checks=checks,
        methods=list(METHODS), source_tests_design_and_input_unchanged=True,
        no_previous_game_ids=True, predict_before_reveal=True, no_live_integration=True)
    write_json(output / 'verification.json', verification, exclusive=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('preregister', 'run', 'evaluate'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(argv)
    if args.phase == 'preregister':
        result = preregister(args.output)
        print(json.dumps(dict(games=len(result['jobs']), created_utc=result['created_utc']), indent=2))
    elif args.phase == 'run':
        result = run_baselines(args.output, args.workers)
        print(json.dumps({k: result[k] for k in ('games', 'complete', 'elapsed_seconds')}, indent=2))
    else:
        result = evaluate(args.output)
        print(json.dumps({k: result[k] for k in ('games', 'ticks', 'primary_gate_passed', 'primary_gate_failures', 'cost')}, indent=2))


if __name__ == '__main__':
    main()
