"""Exact historical replay or one profiled H32 decision; never adoption evidence.

No new world is played. Hidden laws score stored predictions only. A mismatch
fails the diagnostic; neither owner hashes nor the comparison boundary relax.
"""
import argparse
from collections import defaultdict
import cProfile
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import pstats
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EVIDENCE = ROOT / 'evidence/long_horizon/continuation_selected_confirmation'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source_fingerprint():
    files = sorted([*ROOT.glob('reflex/**/*.py'), *ROOT.glob('tools/**/*.py'),
                    ROOT / 'pyproject.toml'])
    entries = {p.relative_to(ROOT).as_posix(): sha(p.read_bytes()) for p in files}
    return dict(files=len(entries), sha256=sha(json.dumps(
        entries, sort_keys=True, separators=(',', ':')).encode()))


def read_inputs():
    plan_path, runs_path = EVIDENCE / 'plan.json', EVIDENCE / 'runs.jsonl.gz'
    plan_bytes, runs_bytes = plan_path.read_bytes(), runs_path.read_bytes()
    plan = json.loads(plan_bytes)
    runs = [json.loads(line) for line in gzip.decompress(runs_bytes).splitlines()]
    specs = {s['id']: s for s in plan['specs']}
    assert len(specs) == len(plan['specs']) == 40, 'expected 40 distinct registered conditions'
    indexed = {}
    for row in runs:
        spec, retry, result = row['spec'], row['retry'], row['result']
        key = (spec['id'], retry)
        assert type(retry) is bool and key not in indexed, 'duplicate or invalid arm'
        assert specs.get(spec['id']) == spec, 'unregistered condition'
        assert result['complete'] and len(result['trace']) == spec['turns'] == 128
        indexed[key] = row
    assert set(indexed) == {(s, arm) for s in specs for arm in (False, True)}, 'missing arm'
    return plan, indexed, {'plan.json': sha(plan_bytes), 'runs.jsonl.gz': sha(runs_bytes)}


def metrics(rows):
    assert rows
    n = len(rows)
    return dict(observations=n,
                expected_brier_excess=math.fsum(r['expected_brier_excess'] for r in rows) / n,
                observed_log_loss=math.fsum(r['observed_log_loss'] for r in rows) / n,
                dominant_correct_fraction=sum(r['dominant_correct'] for r in rows) / n)


def history(plan, indexed):
    from tools.review_adverse_recovery import historical_diagnosis
    result = historical_diagnosis(plan)
    assert len(result['cases']) == 80 and len(result['pairs']) == 40
    specs = {s['id']: s for s in plan['specs']}
    family, persona, world = defaultdict(list), defaultdict(list), defaultdict(list)
    audited = set()
    for case in result['cases']:
        spec, arm = specs[case['id']], case['arm']
        key = (case['id'], arm)
        assert key not in audited and arm in ('original-H16', 'stable-H32')
        audited.add(key)
        rows = case['predictions']
        assert case['replayed'] == len(rows) == 128
        assert [r['tick'] for r in rows] == list(range(128))
        family[spec['game'], arm].extend(rows)
        persona[spec['game'], spec['profile'], arm].extend(rows)
        world[spec['game'], spec['seed'], spec['regime'], arm].extend(rows)
    adverse = []
    for identifier, spec in specs.items():
        left, right = (indexed[identifier, arm]['result'] for arm in (False, True))
        assert all(a['truth'] == b['truth'] and a['response'] == b['response']
                   for a, b in zip(left['trace'], right['trace'])), 'unpaired actual outcomes'
        score = right['score'] - left['score']
        if score < -1e-12:
            conditional = (sum(r['conditional_increment'] for r in right['trace'])
                           - sum(r['conditional_increment'] for r in left['trace']))
            adverse.append(dict(id=identifier, game=spec['game'], profile=spec['profile'],
                                score_delta=score, conditional_delta=conditional,
                                realized_residual_delta=score-conditional))
    assert len(adverse) == 5, 'registered stable comparison no longer has five score losses'
    worlds = []
    for (game, seed, regime, arm), rows in sorted(world.items()):
        windows = [dict(start=start, end=start+8,
                        **metrics([r for r in rows if start <= r['tick'] < start+8]))
                   for start in range(0, 128, 8)]
        worlds.append(dict(game=game, seed=seed, regime=regime, arm=arm,
                           **metrics(rows), windows=windows))
    return dict(pairs=40, histories=80, actual_decisions=10240, exact_owner_replay=True,
                identical_prediction_pairs=sum(r['same_predictions'] for r in result['pairs']),
                by_family=[dict(game=g, arm=a, **metrics(rows))
                           for (g, a), rows in sorted(family.items())],
                by_persona=[dict(game=g, profile=p, arm=a, **metrics(rows))
                            for (g, p, a), rows in sorted(persona.items())],
                by_world=worlds, adverse_cases=sorted(adverse, key=lambda r: r['id']),
                limit='Stored actual histories, not new games or causal attribution. Shared-world personas are not independent worlds.')


def profile_summary(profiler):
    if not profiler.getstats():
        return []
    rows = []
    for (filename, line, name), (primitive, calls, own, cumulative, _) in pstats.Stats(profiler).stats.items():
        path = Path(filename)
        location = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name
        rows.append(dict(function=f'{location}:{line}:{name}', primitive_calls=primitive,
                         calls=calls, self_seconds=own, cumulative_seconds=cumulative))
    return sorted(rows, key=lambda r: (-r['cumulative_seconds'], r['function']))[:15]


def root(indexed, identifier, tick, profiler):
    from tools.probe_adverse_forecasts import probe
    if (identifier, True) not in indexed or not 0 <= tick < 128:
        raise ValueError('registered condition and tick in 0..127 required')
    row = indexed[identifier, True]
    result = profiler.runcall(probe, row, tick, 'stable', (701, 702, 703))
    previous = row['result']['trace'][tick]
    assert result['factual_owner'] == previous['before_owner'], 'root owner differs'
    return dict(id=identifier, tick=tick, action=result['action'],
                factual_owner=result['factual_owner'], selected_purpose=result['selected_purpose'],
                recorded_action=previous['action'],
                action_matches_recorded=result['action'] == previous['action'],
                purpose_delta_from_recorded=result['selected_purpose']-previous['selected_purpose'],
                simulated_steps=result['simulated_steps'], profiled_seconds=result['seconds'],
                limit='One known-state decision with profiling overhead, not game quality or production latency.')


def snapshot(indexed, identifier, tick, profiler):
    """Require the entire current owner exactly; report every earlier hash mismatch."""
    from reflex.core import character_key, digest
    from reflex.decision_loop import DecisionLoop
    from reflex.model_observer import resolve_feedback
    from tools.run_adverse_recovery import decide, game_for
    if (identifier, True) not in indexed or not 0 <= tick < 128:
        raise ValueError('registered condition and tick in 0..127 required')
    row = indexed[identifier, True]; game = game_for(row['spec']); world = game.start()
    actor = game.actor(game.context(world), learning='segmented')
    axes = character_key(game.context(world)); mismatches = []
    for past in row['result']['trace'][:tick]:
        assert world == past['before'], ('physical before', past['tick'])
        context = game.context(world, actor.state)
        decision = DecisionLoop.decide_batch([(actor, game.request(context))], False,
                                             _counterfactual_roots=(past['action'],))[0]
        after, flow = game.transition(world, past['action'], past['response'])
        resolve_feedback(actor, decision['ticket'], game.feedback(world, after, past['action'], context, flow))
        assert after == past['after'], ('physical after', past['tick'])
        assert character_key(game.context(after)) == axes
        if digest(actor.record()) != past['owner']:
            mismatches.append(past['tick'])
        world = after
    prior = row['result']['trace'][tick]
    assert world == prior['before'], 'diagnostic world mismatch'
    assert digest(actor.record()) == prior['before_owner'], 'diagnostic owner mismatch'
    result, cap = profiler.runcall(decide, actor, game, world, 'stable')
    assert character_key(result['context']) == axes
    forecast = cap['forecast']; best = {}
    for name, roots in forecast.roots.items():
        best[roots[0]] = max(best.get(roots[0], -1.), forecast.purpose[name])
    return dict(id=identifier, tick=tick, factual_owner=prior['before_owner'],
        current_world_owner_exact=True, original_history_exact=not mismatches,
        earlier_owner_hash_mismatches=mismatches,
        action=result['decision']['action_id'], recorded_action=prior['action'],
        action_matches_recorded=result['decision']['action_id'] == prior['action'],
        selected_purpose=result['deliberation']['selected_purpose'], estimated_purpose_by_root=best,
        simulated_steps=cap['simulated_steps'],
        limit='Exact current snapshot only. Earlier hash mismatches remain failures of historical replay. Not new-game quality evidence.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('history', 'root', 'snapshot'))
    parser.add_argument('--id', dest='identifier')
    parser.add_argument('--tick', type=int)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode in ('root', 'snapshot') and (args.identifier is None or args.tick is None):
        parser.error('root requires --id and --tick')
    if args.mode == 'history' and (args.identifier is not None or args.tick is not None):
        parser.error('history always audits all registered conditions')
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    summary = dict(mode=args.mode, complete=False, adoption=False, state='running',
                   started_at_utc=datetime.now(timezone.utc).isoformat(),
                   runtime=dict(python=platform.python_version(), system=platform.system()))
    profiler = cProfile.Profile() if args.mode in ('root', 'snapshot') else None
    with output.open('x', encoding='utf-8') as stream:
        def write():
            stream.seek(0)
            json.dump(summary, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
            stream.write('\n'); stream.truncate(); stream.flush()
        write()
        try:
            import numpy as np
            summary['runtime']['numpy'] = np.__version__
            summary['source'] = source_fingerprint()
            plan, indexed, summary['inputs_sha256'] = read_inputs()
            if args.mode == 'history':
                summary['result'] = history(plan, indexed)
            else:
                probe = snapshot if args.mode == 'snapshot' else root
                summary['result'] = probe(indexed, args.identifier, args.tick, profiler)
            assert source_fingerprint() == summary['source'], 'source changed during diagnostic'
            assert read_inputs()[2] == summary['inputs_sha256'], 'inputs changed during diagnostic'
            summary.update(complete=True, state='complete', source_unchanged=True, inputs_unchanged=True)
        except BaseException as exc:
            failure_context = {}
            cursor = exc.__traceback__
            while cursor is not None:
                local = cursor.tb_frame.f_locals
                spec, step = local.get('spec'), local.get('t')
                if isinstance(spec, dict) and isinstance(spec.get('id'), str):
                    failure_context['id'] = spec['id']
                if isinstance(step, dict) and type(step.get('tick')) is int:
                    failure_context['tick'] = step['tick']
                cursor = cursor.tb_next
            summary.update(state='failed', complete=False, error=dict(
                type=type(exc).__name__, message=str(exc)[:600], context=failure_context,
                at=[dict(file=Path(f.filename).name, line=f.lineno, function=f.name)
                    for f in traceback.extract_tb(exc.__traceback__)[-4:]]))
        finally:
            if profiler is not None:
                profiler.disable()
                summary['profile_top'] = profile_summary(profiler)
            summary['elapsed_seconds'] = time.monotonic()-started
            write()
    print(json.dumps(dict(state=summary['state'], complete=summary['complete'], adoption=False,
                          output=str(output), elapsed_seconds=summary['elapsed_seconds'])))
    return 0 if summary['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
