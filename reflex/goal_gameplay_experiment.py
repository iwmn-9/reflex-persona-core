"""Separately preregistered, exploratory equal-goal gameplay ablation.

Offline prediction evidence remains immutable. This fresh experiment compares
three fixed-budget planning models without learning opponent weights or changing
the default. No aggregate result is a default-promotion decision.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import hashlib
import json
import statistics
import subprocess
import time

from .combat import Battle, alive, legal, battle_from_record, terminal
from .combat_planning import TacticalControl
from .congestion_experiment import HELDOUT, authored_world, enemy_hypotheses
from .continuation_experiment import sources
from .core import digest
from .laboratory import profiles
from .root_collision_experiment import movement_and_completion
from .validation_experiment import combat, replay, replay_progress
from .wait_experiment import diagnose

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / 'evidence/goal_gameplay/design.json'
VARIANTS = ('fixed', 'uniform-coverage', 'goal-uniform')
CONTRASTS = (('fixed', 'uniform-coverage'), ('fixed', 'goal-uniform'), ('uniform-coverage', 'goal-uniform'))
SEEDS = (2701, 2702)
PARTITIONS = ('controlled', 'heldout')
METRICS = ('won', 'lost', 'win_credit', 'ticks', 'decisions', 'planned_ticks', 'planning_declines',
    'final_eliminate_progress', 'final_secure_progress', 'expired_stop_selections',
    'unresolved_stop_selections', 'unsupported_root_selections', 'learned_uses', 'regime_resets')
MOVEMENT = ('move_attempts', 'failed_moves', 'friendly_collision_moves', 'opponent_collision_moves',
    'both_collision_moves', 'unclassified_failed_moves', 'unwon_unlost_games')


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def condition_id(row):
    return '|'.join(str(row[k]) for k in ('partition', 'scenario', 'profile', 'seed', 'rival'))


def game_id(row):
    return condition_id(row) + '|' + row['variant']


def jobs():
    conditions = []
    for scene in ('enemy_convergence', 'mixed_junction'):
        for profile in profiles():
            for seed in SEEDS:
                conditions.append(dict(partition='controlled', scene=scene, goal='both', rival='reference',
                    profile=profile['id'], seed=seed, limit=24))
    for cell, (scene, goal, rival) in enumerate(HELDOUT):
        for index, profile in enumerate(profiles()):
            conditions.append(dict(partition='heldout', scene=scene, goal=goal, rival=rival,
                profile=profile['id'], seed=SEEDS[(cell+index) % 2], limit=40))
    work = []
    for condition in conditions:
        for variant in VARIANTS:
            row = dict(condition, variant=variant, scenario=condition['scene']+'/'+condition['goal'])
            row.update(condition=condition_id(row), game=game_id(row))
            work.append(row)
    if len(work) != 96 or len({j['game'] for j in work}) != 96:
        raise ValueError('exactly 32 matched triples required')
    return work


def timing_jobs():
    """Four separate, serial trace-off cost measurements, never efficacy rows."""
    lookup = {row['variant']: row for row in jobs() if row['partition'] == 'controlled'
        and row['scene'] == 'mixed_junction' and row['profile'] == 'care' and row['seed'] == 2702}
    return [dict(lookup[variant], repetition=repetition, trace_execution=False)
        for repetition in (0, 1) for variant in ('fixed', 'goal-uniform')]


def configuration(variant, trace_execution=True):
    if variant not in VARIANTS:
        raise ValueError('unknown preregistered opponent model')
    return TacticalControl(recovery_options=True, trace_execution=trace_execution, opponent_model=variant)


def frozen_files():
    paths = [DESIGN, ROOT/'evidence/controlled_congestion/design.json', ROOT/'pyproject.toml']
    paths += sorted((ROOT/'tests').glob('*.py'))
    # Preserve the completed offline result and its original preregistration.
    paths += [ROOT/'evidence/goal_opponent'/name for name in
        ('design.json', 'preregister.json', 'evaluation.json', 'verification.json', 'projection.json')]
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}


def preregister(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('use a fresh empty output directory')
    offline = json.loads((ROOT/'evidence/goal_opponent/evaluation.json').read_text())
    manifest = dict(format='goal-gameplay-preregister-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        checkout_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_hashes=sources(), file_hashes=frozen_files(), jobs=jobs(), timing_jobs=timing_jobs(),
        configs={v: asdict(configuration(v)) for v in VARIANTS},
        offline_primary_gate_passed=offline['primary_gate_passed'],
        offline_primary_gate_failures=offline['primary_gate_failures'],
        scope='Separately authorized causal exploratory equal-goal gameplay; does not reinterpret offline failure as a pass',
        default_promotion=False, learned_opponent_weights=False)
    write_json(output/'preregister.json', manifest)
    return manifest


def check_frozen(output):
    manifest = json.loads((Path(output)/'preregister.json').read_text())
    if manifest['source_hashes'] != sources() or manifest['file_hashes'] != frozen_files():
        raise ValueError('frozen source, tests, designs or offline evidence changed')
    if manifest['jobs'] != jobs() or manifest['timing_jobs'] != timing_jobs() or manifest['configs'] != {v: asdict(configuration(v)) for v in VARIANTS}:
        raise ValueError('preregistered jobs or configuration changed')
    return manifest


def initial_world(job):
    return (authored_world(job['scene'], job['goal'], job['seed'] % 2, job['limit']) if job['partition'] == 'controlled'
        else Battle.start(job['scene'], job['goal'], limit=job['limit']))


def verify_run(run, expected=None):
    if not run['trace'] or run['variant'] not in VARIANTS:
        raise ValueError('nonempty known-variant trajectory required')
    if expected:
        for key in ('partition', 'scenario', 'profile', 'seed', 'rival', 'variant'):
            if run[key] != expected[key]:
                raise ValueError('wrong declared condition: '+key)
        if battle_from_record(run['trace'][0]['before']) != initial_world(expected):
            raise ValueError('incorrect preregistered initial state')
    legal_count = fixed_count = sampled_count = 0
    previous = None
    for row in run['trace']:
        before = battle_from_record(row['before'])
        if previous is not None and previous != before:
            raise ValueError('discontinuous public states')
        actual = {int(k): v for k, v in row['choices'].items()}
        if set(actual) != set(alive(before, 0) + alive(before, 1)):
            raise ValueError('incomplete actual joint intent')
        for actor, action in actual.items():
            if action not in legal(before, actor):
                raise ValueError('illegal revealed action')
            legal_count += 1
        execution = row.get('planning', {}).get('execution')
        if execution and run['variant'] == 'fixed':
            hypotheses = enemy_hypotheses(before, run['seed'] % 2)
            for index, sample in enumerate(execution['samples']):
                if sample:
                    if sample[0]['opponent'] != {str(i): k for i, k in hypotheses[index % 2].items()}:
                        raise ValueError('fixed baseline first-step hypothesis changed')
                    fixed_count += 1
        elif execution:
            from .goal_rollout import GoalRollout
            for index, sample in enumerate(execution['samples']):
                if sample:
                    predicted = GoalRollout(before, run['seed'] % 2, index, run['variant'], samples=len(execution['samples'])).choose(before, 0)
                    if sample[0]['opponent'] != {str(i): k for i, k in predicted.items()}:
                        raise ValueError('sampled public first-step hypothesis mismatch')
                    sampled_count += 1
        previous = battle_from_record(row['after'])
    if not terminal(previous) or previous.tick != run['ticks'] or len(run['trace']) != run['ticks']:
        raise ValueError('truncated or inconsistent terminal trace')
    return dict(rule_transitions=replay(run), progress_updates=replay_progress(run),
        legal_actual_intents=legal_count, fixed_hypothesis_checks=fixed_count,
        sampled_hypothesis_checks=sampled_count)


def run_job(job, trace_execution=True):
    from . import combat_planning as cp
    profile = next(p for p in profiles() if p['id'] == job['profile'])
    immutable = copy.deepcopy(profile)
    original = cp.forecast
    work = dict(planner_seconds=0., calls=0, model_transitions=0, recovery_calls=0)
    def measured(w, actors, contexts, reflex, control=None, root_allowed=None):
        start = time.perf_counter()
        result = original(w, actors, contexts, reflex, control, root_allowed)
        work['planner_seconds'] += time.perf_counter()-start
        work['calls'] += 1
        work['model_transitions'] += result.audit['nodes']
        work['recovery_calls'] += root_allowed is not None
        bound = (control.max_plans + (control.recovery_plans if root_allowed is not None else 0))*control.samples*control.horizon
        if result.audit['nodes'] > bound:
            raise AssertionError('declared planner transition budget exceeded')
        return result
    cp.forecast = measured
    try:
        start = time.perf_counter()
        run = combat(profile, job['seed'], job['scene'], job['goal'], 'baseline', learn=True,
            survival_security=True, rival=job['rival'], planner=configuration(job['variant'], trace_execution),
            progress_watch=True, initial_world=initial_world(job), trace_roots=True)
        elapsed = time.perf_counter()-start
    finally:
        cp.forecast = original
    if profile != immutable:
        raise AssertionError('fixed persona was mutated')
    run.update(partition=job['partition'], rival=job['rival'], variant=job['variant'],
        game=job['game'], condition=job['condition'], game_seconds=elapsed,
        total_work=work, trace_execution=trace_execution)
    run.update(diagnose(run, profile))
    run.update(movement_and_completion(run))
    return run, verify_run(run, job)


def actual_choices(row):
    return {int(k): v for k, v in row['choices'].items()}


def first_divergence(left, right):
    """Only an own-action divergence on the same state starts a causal split."""
    if condition_id(left) != condition_id(right):
        raise ValueError('matched conditions required')
    team = left['seed'] % 2
    first_route_change = None
    for a, b in zip(left['trace'], right['trace']):
        if a['before'] != b['before']:
            raise AssertionError('public state changed before first differing action')
        w = battle_from_record(a['before'])
        ours, enemies = alive(w, team), alive(w, 1-team)
        ca, cb = actual_choices(a), actual_choices(b)
        if any(ca[i] != cb[i] for i in enemies):
            raise AssertionError('actual enemy differed on the same pre-state')
        if first_route_change is None and a.get('selected_routes') != b.get('selected_routes'):
            first_route_change = w.tick
        changed = [i for i in ours if ca[i] != cb[i]]
        if changed:
            return dict(tick=w.tick, same_public_before=True, public_before_sha256=digest(a['before']),
                same_actual_enemy=True, changed_actors=changed,
                left_choices={str(i): ca[i] for i in ours}, right_choices={str(i): cb[i] for i in ours},
                actual_enemy={str(i): ca[i] for i in enemies}, first_route_change=first_route_change,
                left_planning={k: v for k, v in a.get('planning', {}).items() if k != 'execution'},
                right_planning={k: v for k, v in b.get('planning', {}).items() if k != 'execution'},
                cause_scope='The only configured intervention is the opponent forecast model; later path-dependent learning and observations may also change')
        if a['after'] != b['after']:
            raise AssertionError('identical actions and seed gave a different real transition')
    if len(left['trace']) != len(right['trace']):
        raise AssertionError('trajectory length changed without an earlier changed action')
    return dict(tick=None, same_trajectory=True, same_actual_enemy=True, first_route_change=first_route_change)


def ratio(numerator, denominator):
    return numerator/denominator if denominator else None


def rates(run):
    movement = run['movement']
    attempts = movement.get('move_attempts', 0)
    return dict(failed_move_rate=ratio(movement.get('failed_moves', 0), attempts),
        enemy_only_failure_rate=ratio(movement.get('opponent_collision_moves', 0), attempts),
        friendly_only_failure_rate=ratio(movement.get('friendly_collision_moves', 0), attempts),
        both_failure_rate=ratio(movement.get('both_collision_moves', 0), attempts),
        enemy_contested_failure_rate=ratio(movement.get('opponent_collision_moves', 0)+movement.get('both_collision_moves', 0), attempts),
        unresolved_stop_rate=ratio(run.get('unresolved_stop_selections', 0), run.get('decisions', 0)))


def totals(rows):
    metrics, movement, work = Counter(), Counter(), Counter()
    for row in rows:
        metrics.update({k: row.get(k, 0) for k in METRICS})
        movement.update(row['movement'])
        work.update(row['total_work'])
    merged = dict(movement=movement, **metrics)
    all_rates = [rates(row) for row in rows]
    return dict(games=len(rows), metrics=dict(metrics), movement=dict(movement), work=dict(work),
        game_seconds=sum(r['game_seconds'] for r in rows), pooled_rates=rates(merged),
        equal_game_rates={name: dict(mean=statistics.mean(values) if values else None,
                nonempty_games=len(values), empty_games=len(rows)-len(values))
            for name in rates(merged) for values in [[r[name] for r in all_rates if r[name] is not None]]},
        mean_eliminate_progress=ratio(metrics['final_eliminate_progress'], len(rows)),
        mean_secure_progress=ratio(metrics['final_secure_progress'], len(rows)),
        longest_stall=max((r['longest_observed_stall'] for r in rows), default=0),
        longest_unresolved_stop_run=max((r.get('longest_unresolved_stop_run', 0) for r in rows), default=0))


def paired_delta(left, right):
    ra, rb = rates(left), rates(right)
    return dict(condition=condition_id(left), partition=left['partition'], scenario=left['scenario'],
        profile=left['profile'], rival=left['rival'], seed=left['seed'], left=left['variant'], right=right['variant'],
        metrics={k: right.get(k, 0)-left.get(k, 0) for k in (*METRICS, 'longest_observed_stall', 'longest_unresolved_stop_run')},
        movement={k: right['movement'].get(k, 0)-left['movement'].get(k, 0) for k in MOVEMENT},
        rates={k: rb[k]-ra[k] if ra[k] is not None and rb[k] is not None else None for k in ra},
        work={k: right['total_work'].get(k, 0)-left['total_work'].get(k, 0) for k in left['total_work']},
        game_seconds_delta=right['game_seconds']-left['game_seconds'])


def pair_summary(pairs):
    result = {}
    for category in ('metrics', 'movement', 'rates'):
        result[category] = {}
        for key in pairs[0][category] if pairs else ():
            values = [row[category][key] for row in pairs if row[category][key] is not None]
            result[category][key] = dict(mean_delta=statistics.mean(values) if values else None,
                paired_games=len(values), unavailable_games=len(pairs)-len(values),
                lower=sum(v < -1e-12 for v in values), higher=sum(v > 1e-12 for v in values),
                unchanged=sum(abs(v) <= 1e-12 for v in values))
    return result


def aggregate(runs):
    lookup = {(condition_id(row), row['variant']): row for row in runs}
    conditions = sorted({condition_id(row) for row in runs})
    pairs = [paired_delta(lookup[(c, left)], lookup[(c, right)]) for c in conditions for left, right in CONTRASTS]
    groups = []
    for partition in PARTITIONS:
        subset = [row for row in runs if row['partition'] == partition]
        for dimension in ('all', 'profile', 'scenario', 'rival'):
            for value in ['all'] if dimension == 'all' else sorted({r[dimension] for r in subset}):
                selected = [row for row in subset if dimension == 'all' or row[dimension] == value]
                selected_pairs = [row for row in pairs if row['partition'] == partition and (dimension == 'all' or row[dimension] == value)]
                groups.append(dict(partition=partition, dimension=dimension, value=value,
                    variants={v: totals([r for r in selected if r['variant'] == v]) for v in VARIANTS},
                    contrasts=[dict(left=left, right=right, **pair_summary([p for p in selected_pairs if (p['left'], p['right']) == (left, right)])) for left, right in CONTRASTS]))
    regressions = []
    for group in groups:
        if group['dimension'] not in ('all', 'profile'):
            continue
        for left, right in CONTRASTS:
            a, b = group['variants'][left], group['variants'][right]
            reasons = []
            for key in ('won', 'win_credit', 'final_eliminate_progress', 'final_secure_progress'):
                if b['metrics'].get(key, 0) < a['metrics'].get(key, 0)-1e-12:
                    reasons.append(key+' decreased')
            for key in ('lost', 'expired_stop_selections', 'unresolved_stop_selections', 'unsupported_root_selections'):
                if b['metrics'].get(key, 0) > a['metrics'].get(key, 0):
                    reasons.append(key+' increased')
            if b['movement'].get('unwon_unlost_games', 0) > a['movement'].get('unwon_unlost_games', 0):
                reasons.append('unresolved games increased')
            if b['longest_stall'] > a['longest_stall']:
                reasons.append('longest stall increased')
            if b['longest_unresolved_stop_run'] > a['longest_unresolved_stop_run']:
                reasons.append('longest unresolved stop run increased')
            for rate in ('failed_move_rate', 'enemy_contested_failure_rate', 'unresolved_stop_rate'):
                x, y = a['pooled_rates'][rate], b['pooled_rates'][rate]
                if x is not None and y is not None and y > x+1e-12:
                    reasons.append(rate+' increased')
            if reasons:
                regressions.append(dict(partition=group['partition'], dimension=group['dimension'], value=group['value'], left=left, right=right, reasons=reasons))
    return groups, pairs, regressions


def experiment(output, workers=4):
    output = Path(output)
    manifest = check_frozen(output)
    if not 1 <= workers <= 8:
        raise ValueError('workers must be in [1, 8]')
    if (output/'trajectories.jsonl').exists() or (output/'evaluation.json').exists():
        raise FileExistsError('do not overwrite or selectively rerun evidence')
    start = time.perf_counter()
    results, checks = {}, []
    with (output/'trajectories.jsonl').open('x') as stream, ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_job, job): job['game'] for job in manifest['jobs']}
        for future in as_completed(futures):
            run, verification = future.result()
            results[game_id(run)] = run
            checks.append(dict(game=game_id(run), **verification))
            stream.write(json.dumps(run, allow_nan=False)+'\n'); stream.flush()
            print(f'{len(results)}/{len(futures)} {game_id(run)} {run["game_seconds"]:.1f}s', flush=True)
    if set(results) != {job['game'] for job in manifest['jobs']}:
        raise ValueError('incomplete matched triples')
    check_frozen(output)
    runs = [results[job['game']] for job in manifest['jobs']]
    lookup = {(condition_id(row), row['variant']): row for row in runs}
    divergences = [dict(condition=c, left=left, right=right, **first_divergence(lookup[(c, left)], lookup[(c, right)]))
        for c in sorted({condition_id(row) for row in runs}) for left, right in CONTRASTS]
    slim = [{k: v for k, v in row.items() if k != 'trace'} for row in runs]
    groups, pairs, regressions = aggregate(slim)
    result = dict(format='goal-gameplay-evaluation-v1', games=len(runs), matched_triples=32,
        groups=groups, pairs=pairs, regressions=regressions, divergences=divergences, runs=slim,
        replay_checks=checks, elapsed_seconds=time.perf_counter()-start,
        source_hashes=sources(), file_hashes=frozen_files(), source_tests_design_offline_unchanged=True,
        offline_primary_gate_passed=manifest['offline_primary_gate_passed'],
        exploratory=True, learned_opponent_weights=False, defaults_changed=False, default_promotion=False,
        limitations=['Fresh seeds in familiar authored conditions; no broad generalization claim',
            'Differences are causal within these matched seeded runs, but cells share scenarios and actors/ticks are correlated',
            'Sampling and goal-support are distinct contrasts; no adaptive learner is used',
            'Profile win equality is not a goal, and adverse outcomes do not prove a persona is defective',
            'Parallel full-game timing includes path differences and shared-host contention; inspect separate serial fixture'])
    write_json(output/'evaluation.json', result)
    write_json(output/'verification.json', dict(games=len(runs), matched_triples=32,
        artifact_sha256={name: sha(output/name) for name in ('trajectories.jsonl', 'evaluation.json')},
        source_hashes=sources(), file_hashes=frozen_files(), complete=True,
        same_state_first_divergences_checked=True, exact_rules_and_progress_replayed=True))
    return result


def semantic_signature(run):
    rows = []
    for source in run['trace']:
        row = copy.deepcopy(source)
        if 'planning' in row:
            row['planning'].pop('execution', None)
        rows.append(row)
    return digest(dict(trace=rows, metrics={k: run.get(k, 0) for k in METRICS},
        movement=run['movement'], learned_uses=run['learned_uses'], total_nodes=run['total_work']['model_transitions']))


def serial_timing(output):
    output = Path(output)
    manifest = check_frozen(output)
    verification = json.loads((output/'verification.json').read_text())
    if not verification['complete'] or verification['artifact_sha256']['trajectories.jsonl'] != sha(output/'trajectories.jsonl'):
        raise ValueError('complete unchanged primary trajectories required')
    if (output/'serial_timing.json').exists() or (output/'timing_trajectories.jsonl').exists():
        raise FileExistsError('serial timing fixture is not selectively repeated')
    primary = {game_id(r): r for r in [json.loads(line) for line in (output/'trajectories.jsonl').read_text().splitlines()]}
    observations = []
    with (output/'timing_trajectories.jsonl').open('x') as stream:
        for job in manifest['timing_jobs']:
            run, checks = run_job(job, trace_execution=False)
            same = semantic_signature(run) == semantic_signature(primary[job['game']])
            if not same:
                raise AssertionError('trace-off timing altered decisions, transitions, progress or learned-use counts')
            stream.write(json.dumps(dict(repetition=job['repetition'], **run), allow_nan=False)+'\n')
            stream.flush()
            observations.append(dict(game=job['game'], variant=job['variant'], repetition=job['repetition'],
                game_seconds=run['game_seconds'], decision_and_feedback_p50_ms=run['decision_and_feedback_p50_ms'],
                work=run['total_work'], checks=checks, semantic_trace_match=True))
    check_frozen(output)
    summary = {variant: dict(samples=2, median_game_seconds=statistics.median(r['game_seconds'] for r in observations if r['variant'] == variant),
        all_game_seconds=[r['game_seconds'] for r in observations if r['variant'] == variant]) for variant in ('fixed', 'goal-uniform')}
    result = dict(format='goal-gameplay-serial-cost-v1', primary_efficacy_games=96, separate_timing_games=4,
        observations=observations, summary=summary,
        median_game_ratio=summary['goal-uniform']['median_game_seconds']/summary['fixed']['median_game_seconds'],
        note='One fixed authored condition, two repetitions each, alternating variant order. Path differences remain; not pure per-call overhead, a performance guarantee, or an efficacy sample.',
        trajectories_sha256=sha(output/'timing_trajectories.jsonl'), source_hashes=sources(), file_hashes=frozen_files())
    write_json(output/'serial_timing.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('preregister', 'run', 'timing'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(argv)
    if args.phase == 'preregister':
        result = preregister(args.output)
        print(json.dumps(dict(primary_games=len(result['jobs']), separate_timing_games=len(result['timing_jobs'])), indent=2))
    elif args.phase == 'run':
        result = experiment(args.output, args.workers)
        print(json.dumps(dict(games=result['games'], regressions=len(result['regressions']), defaults_changed=False), indent=2))
    else:
        result = serial_timing(args.output)
        print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
