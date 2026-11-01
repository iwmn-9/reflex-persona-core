"""Compare two stable H32 runtimes in separate processes on one identical root.

Input JSON: format='colab-stable-root-v1', spec, world, owner (the complete
resolved DecisionLoop record), response, and provenance. The caller constructs
this root by factual replay on the current platform. This tool does not relax
or replace a historical Windows replay audit. The response is used only after
selection, for one actual transition; no hidden law enters planning.

The exact-runtime benchmark's complete-output comparison is extended with the
same actual feedback and configuration-bound save/restore. No cProfile, new
planning rule, tolerance, reduced horizon, or omitted semantic field is used.
"""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback


FORMAT = 'colab-optimized-root-v1'
ARITHMETIC = {
    'MKL_NUM_THREADS': '1',
    'OMP_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'OPENBLAS_CORETYPE': 'Haswell',
    'NPY_DISABLE_CPU_FEATURES': 'AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL',
}
ADAPTERS = ('tools/continuing_games.py', 'tools/preparation_world.py',
            'tools/run_continuation_portfolio.py', 'tools/run_owned_value.py')


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def sha(value):
    return hashlib.sha256(value).hexdigest()


def write(path, value):
    path.write_bytes(encoded(value))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sources(root):
    files = sorted({*root.glob('reflex/**/*.py'), *root.glob('tools/**/*.py'),
                    root / 'pyproject.toml'})
    manifest = {p.relative_to(root).as_posix(): sha(p.read_bytes()) for p in files}
    return dict(sha256=sha(encoded(manifest)), files=len(manifest),
                adapters={name: manifest[name] for name in ADAPTERS})


def plain(value, np):
    def convert(item):
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            return {field.name: getattr(item, field.name) for field in dataclasses.fields(item)}
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError('unsupported comparison value: ' + type(item).__qualname__)
    return json.loads(json.dumps(value, default=convert, ensure_ascii=False, allow_nan=False))


def check_imports(source):
    for name, module in list(sys.modules.items()):
        if name == 'reflex' or name.startswith('reflex.') or name == 'tools' or name.startswith('tools.'):
            file = getattr(module, '__file__', None)
            paths = [file] if file is not None else list(getattr(module, '__path__', ()))
            require(paths and all(Path(p).resolve().is_relative_to(source) for p in paths),
                    'foreign imported module: ' + name)


def worker(source, snapshot, output):
    source, snapshot, output = source.resolve(), snapshot.resolve(), output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(encoded(dict(format=FORMAT, state='running', complete=False, adoption=False)))
    report = dict(format=FORMAT, state='running', complete=False, adoption=False,
                  source=str(source), tool_sha256=sha(Path(__file__).read_bytes()))
    started, cpu_started = time.perf_counter(), time.process_time()
    try:
        require(all(os.environ.get(k) == v for k, v in ARITHMETIC.items()),
                'set the registered arithmetic environment before starting Python')
        require(platform.python_version() == '3.12.14', 'Python 3.12.14 required')
        require(not any(n == 'reflex' or n.startswith('reflex.') or n == 'tools'
                        or n.startswith('tools.') for n in sys.modules), 'fresh worker process required')
        sys.path.insert(0, str(source))
        import numpy as np
        import reflex.fixed_horizon as fixed
        from reflex.core import character_key, digest
        from reflex.decision_loop import DecisionLoop
        from reflex.model_observer import resolve_feedback
        from tools.run_continuation_portfolio import game_for
        from tools.run_owned_value import cumulative_basis

        check_imports(source)
        require(np.__version__ == '2.3.5', 'NumPy 2.3.5 required')
        report['runtime'] = dict(python=sys.version, numpy=np.__version__,
                                 platform=platform.platform(), machine=platform.machine(),
                                 arithmetic={k: os.environ.get(k) for k in ARITHMETIC})
        report['source_before'] = sources(source)
        raw = snapshot.read_bytes()
        report['input_sha256'] = sha(raw)
        data = json.loads(raw)
        require(set(data) == {'format', 'spec', 'world', 'owner', 'response', 'provenance'}
                and data['format'] == 'colab-stable-root-v1', 'known complete root snapshot required')
        require(isinstance(data['provenance'], dict), 'explicit root provenance required')
        spec, world, record = data['spec'], data['world'], data['owner']
        report['root'] = dict(id=spec['id'], tick=world['tick'], world_sha256=sha(encoded(world)),
                              owner_sha256=sha(encoded(record)), provenance=data['provenance'])
        game = game_for(spec)
        require(not game.terminal(world), 'live public root required')
        require(record['last_tick'] + 1 == world['tick'], 'resolved owner immediately before root required')
        context = game.context(world, record['state'])
        template_actor = game.actor(game.context(world), learning='segmented')
        actor = DecisionLoop.from_record(context, record, predictor=template_actor.predictor)
        require(encoded(actor.record()) == encoded(record), 'root owner restoration changed input')
        axes = character_key(context)

        def controller(owner):
            return fixed.FixedHorizonOwner(owner, observe=game.context, events=game.events,
                terminal=game.terminal, assess=lambda w: game.goal(w).record(),
                request_for=game.request, feedback=game.feedback, model_id=game.model_id,
                target=game.target, cumulative_preference=cumulative_basis(game, True),
                horizon=32, seeds=(701, 702, 703), recovery_cache=64)

        loop = controller(actor)
        root_save = plain(loop.checkpoint(world), np)
        loop = loop.restored(world, root_save)
        require(encoded(loop.actor.record()) == encoded(record), 'controller root restore changed owner')
        report['stage'] = 'deciding'
        write(output, report)
        print(json.dumps(dict(state='deciding', source=str(source), id=spec['id'],
                              tick=world['tick'])), flush=True)
        wall, cpu = time.perf_counter(), time.process_time()
        result = loop.decide(world)
        report['decision_seconds'] = dict(cpu=time.process_time()-cpu, wall=time.perf_counter()-wall)
        require(result.get('deliberation', {}).get('adopted') is True, 'complete H32 decision required')
        require(character_key(result['context']) == axes, 'personality changed during planning')
        require(sha(encoded(world)) == report['root']['world_sha256'], 'planner changed public input')
        semantic = dict(root_owner=record, root_checkpoint=root_save,
                        decision=plain(result, np), captured=plain(loop.last, np),
                        pending=plain(loop.actor.pending, np))

        # A single common actual observation, deliberately withheld from decide().
        action = result['decision']['action_id']
        after, flow = game.transition(world, action, data['response'])
        feedback = game.feedback(world, after, action, result['context'], flow)
        observation = loop.resolve(result['ticket'], feedback)
        factual_owner = plain(loop.actor.record(), np)
        require(character_key(game.context(after)) == axes, 'personality changed after real feedback')
        detached = DecisionLoop.from_record(context, record, predictor=template_actor.predictor)
        forced = DecisionLoop.decide_batch([(detached, game.request(context))], False,
                                           _counterfactual_roots=(action,))[0]
        resolve_feedback(detached, forced['ticket'], feedback)
        require(encoded(detached.record()) == encoded(factual_owner),
                'forecast contaminated actual learning or forced-action replay differs')
        saved = plain(loop.checkpoint(after), np)
        restored = controller(template_actor).restored(after, saved)
        require(encoded(restored.actor.record()) == encoded(factual_owner), 'actual owner restore differs')
        restored_save = plain(restored.checkpoint(after), np)
        require(encoded(saved) == encoded(restored_save), 'controller checkpoint round trip differs')
        semantic.update(response=data['response'], after=after, flow=plain(flow, np),
            feedback=plain(feedback, np), observation=plain(observation, np),
            factual_owner=factual_owner, actual_checkpoint=saved,
            restored_owner=plain(restored.actor.record(), np), restored_checkpoint=restored_save)
        require(sha(encoded(record)) == report['root']['owner_sha256'], 'input owner record was mutated')
        require(sha(encoded(world)) == report['root']['world_sha256'], 'input world was mutated')
        require(sha(snapshot.read_bytes()) == report['input_sha256'], 'root input changed during measurement')
        report['source_after'] = sources(source)
        require(report['source_before'] == report['source_after'], 'source changed during measurement')
        check_imports(source)
        require(sha(Path(__file__).read_bytes()) == report['tool_sha256'], 'measurement tool changed')
        report.update(state='complete', stage='complete', complete=True, semantic=semantic,
                      semantic_sha256=sha(encoded(semantic)),
                      fields_sha256={k: sha(encoded(v)) for k, v in semantic.items()},
                      action=action, final_owner=digest(loop.actor.record()),
                      checks=dict(source_unchanged=True, input_unchanged=True,
                                  root_restore=True, persona=True, forced_feedback_replay=True,
                                  actual_restore=True, source_import_isolation=True))
    except BaseException as error:
        report.update(state='failed', error=dict(type=type(error).__name__, message=str(error),
                      traceback=traceback.format_exc(limit=8)))
    report['total_seconds'] = dict(cpu=time.process_time()-cpu_started, wall=time.perf_counter()-started)
    write(output, report)
    print(json.dumps({k: report[k] for k in ('state', 'complete', 'source', 'root',
                     'decision_seconds', 'action', 'error') if k in report}), flush=True)
    return 0 if report['complete'] else 1


def compare(reference, candidate, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(encoded(dict(format=FORMAT, state='running', complete=False, adoption=False)))
    result = dict(format=FORMAT, complete=False, adoption=False)
    try:
        old_raw, new_raw = reference.read_bytes(), candidate.read_bytes()
        old, new = json.loads(old_raw), json.loads(new_raw)
        result['reports_sha256'] = dict(reference=sha(old_raw), candidate=sha(new_raw))
        require(all(r.get('format') == FORMAT and r.get('complete') is True
                    and r.get('state') == 'complete' for r in (old, new)), 'both workers must complete')
        for row in (old, new):
            require(row['semantic_sha256'] == sha(encoded(row['semantic'])), 'semantic report digest mismatch')
            require(row['fields_sha256'] == {k: sha(encoded(v)) for k, v in row['semantic'].items()},
                    'field report digest mismatch')
            require(row['source_before'] == row['source_after'] and all(row['checks'].values()),
                    'worker integrity checks incomplete')
        checks = {name: encoded(old[name]) == encoded(new[name])
                  for name in ('input_sha256', 'tool_sha256', 'runtime', 'root')}
        checks['adapters'] = old['source_before']['adapters'] == new['source_before']['adapters']
        keys = sorted(old['semantic'].keys() | new['semantic'].keys())
        checks['all_semantic_fields'] = old['fields_sha256'] == new['fields_sha256']
        differing = [k for k in keys if old['fields_sha256'].get(k) != new['fields_sha256'].get(k)]
        old_cost, new_cost = old['decision_seconds'], new['decision_seconds']
        result.update(complete=True, state='complete', passed=all(checks.values()), checks=checks,
            root=old['root'], differing_semantic_fields=differing,
            decision_seconds=dict(reference=old_cost, candidate=new_cost),
            cpu_speedup=old_cost['cpu']/new_cost['cpu'] if new_cost['cpu'] > 0 else None,
            semantic_sha256=dict(reference=old['semantic_sha256'], candidate=new['semantic_sha256']),
            scope='One common platform root, complete H32 forecast and one actual feedback/save boundary; '
                  'not a historical-platform replay, long-play quality, or general speed guarantee.')
    except BaseException as error:
        result.update(state='failed', passed=False, error=dict(type=type(error).__name__,
                      message=str(error), traceback=traceback.format_exc(limit=5)))
    write(output, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result.get('passed') else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help='two isolated sequential workers, then exact comparison')
    run.add_argument('--reference-source', type=Path, required=True)
    run.add_argument('--candidate-source', type=Path, required=True)
    run.add_argument('--input', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True, help='fresh directory')
    work = sub.add_parser('worker', help='one isolated source; never run both sources in this process')
    work.add_argument('--source', type=Path, required=True)
    work.add_argument('--input', type=Path, required=True)
    work.add_argument('--output', type=Path, required=True, help='fresh JSON file')
    comp = sub.add_parser('compare', help='compare already completed worker files without recalculation')
    comp.add_argument('--reference', type=Path, required=True)
    comp.add_argument('--candidate', type=Path, required=True)
    comp.add_argument('--output', type=Path, required=True, help='fresh JSON file')
    args = parser.parse_args()
    if args.command == 'worker':
        return worker(args.source, args.input, args.output)
    if args.command == 'compare':
        return compare(args.reference, args.candidate, args.output)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = dict(format=FORMAT, complete=False, adoption=False, workers={})
    for label, source in (('reference', args.reference_source), ('candidate', args.candidate_source)):
        status['active'] = label
        write(output/'status.json', status)
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), 'worker',
            '--source', str(source.resolve()), '--input', str(args.input.resolve()),
            '--output', str(output/(label+'.json'))], check=False, cwd=source.resolve())
        status['workers'][label] = completed.returncode
        write(output/'status.json', status)
        if completed.returncode:
            status.update(state='failed', active=None)
            write(output/'status.json', status)
            return 1
    code = compare(output/'reference.json', output/'candidate.json', output/'comparison.json')
    status.update(state='complete' if code == 0 else 'failed', complete=code == 0, active=None)
    write(output/'status.json', status)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
