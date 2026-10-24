"""Change only hypothetical random draws at all preregistered cutoff disagreements.

Known-state sensitivity, not new real games or independent policy validation.
No sample bank is selected to replace the registered playing configuration.
"""
import argparse
import copy
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def initialize(source):
    sys.path.insert(0, source)
    # Diagnostics use spare CPU without reprioritizing the playing processes.
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint)
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000):
            raise OSError(ctypes.get_last_error(), 'cannot set diagnostic priority')


def probe(job):
    root, case, bank, seeds = job
    from reflex.core import digest
    from reflex.decision_loop import DecisionLoop
    from reflex.model_observer import resolve_feedback
    from tools.horizon_consensus import consensus_planner
    from tools.run_continuation_portfolio import game_for
    from tools.run_owned_value import cumulative_basis

    loaded = Path(sys.modules[consensus_planner.__module__].__file__).resolve()
    assert loaded.is_relative_to(root / '_source')

    start = time.monotonic()
    path = root / (case['id'] + '-' + bank + '.json')
    if path.exists():
        result = json.loads(path.read_bytes())
        assert result['complete'] and result['case'] == case and result['seeds'] == list(seeds)
        assert result['source_owner'] == case['actual_fixed32_owner']
        return result
    data = (root / case['input']).read_bytes()
    assert sha(data) == case['sha256']
    envelope = json.loads(data); saved = envelope['payload']
    assert sha(encoded(saved)) == envelope['sha256']
    spec = saved['contract']['configuration']['spec']; game = game_for(spec)
    actor = game.actor(game.context(game.start()), learning='segmented')
    for entry in saved['trace'][:case['tick']]:
        assert digest(actor.record()) == entry['before_owner']
        context = game.context(entry['before'], actor.state)
        result = DecisionLoop.decide_batch([(actor, game.request(context))], False,
            _counterfactual_roots=(entry['action'],))[0]
        after, flow = game.transition(entry['before'], entry['action'], entry['response'])
        resolve_feedback(actor, result['ticket'], game.feedback(entry['before'], after, entry['action'], context, flow))
        assert after == entry['after'] and digest(actor.record()) == entry['owner']
    entry = saved['trace'][case['tick']]; world = copy.deepcopy(entry['before'])
    before = actor.record(); owner = digest(before)
    assert owner == entry['before_owner'] == case['actual_fixed32_owner']
    request = game.request(game.context(world, actor.state)); captured = {}
    options = dict(spec['controller'])
    options.pop('horizon_consensus'); options.pop('minimum_horizon', None)
    planner = consensus_planner(actor, request, world, observe=game.context,
        events=game.events, terminal=game.terminal, assess=lambda w:game.goal(w).record(),
        request_for=game.request, feedback=game.feedback, model_id=game.model_id,
        target=game.target, horizon=32, seeds=tuple(seeds),
        cumulative_preference=cumulative_basis(game, True), captured=captured, **options)
    decision = DecisionLoop.decide_batch([(copy.deepcopy(actor), request)], False, planner)[0]
    audit = captured['horizon_consensus']
    assert len(audit['ballots']) == 17 and audit['ballots'][-1]['horizon'] == 32
    assert decision['decision']['action_id'] == audit['selected']['action']
    assert actor.record() == before and world == entry['before']
    result = dict(complete=True, case=case, bank=bank, seeds=list(seeds),
        source_owner=owner, factual_owner_unchanged=True, factual_world_unchanged=True,
        ballots=audit['ballots'], consensus_selected=audit['selected'],
        fixed32_selected=audit['ballots'][-1], votes=audit['votes'],
        simulated_steps=captured['simulated_steps'], seconds=time.monotonic()-start,
        source_file=loaded.relative_to(root / '_source').as_posix(),
        limit=__doc__)
    tmp = path.with_suffix('.new'); tmp.write_bytes(encoded(result)); tmp.replace(path)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    args = parser.parse_args(); root = Path(args.root).resolve()
    plan = json.loads((root / 'plan.json').read_bytes())
    assert sha(Path(__file__).read_bytes()) == plan['diagnostic_sha256']
    manifest = json.loads((root / 'source_snapshot.json').read_bytes())
    assert sha(encoded(manifest)) == plan['source']
    for name, expected in manifest.items():
        assert sha((root / '_source' / name).read_bytes()) == expected
    jobs = [(root, case, bank['id'], bank['seeds']) for case in plan['cases'] for bank in plan['banks']]
    state = dict(complete=False, adoption=False, pid=os.getpid(), registered=len(jobs), completed=0)
    def status(**values):
        state.update(values, updated=datetime.now(timezone.utc).isoformat())
        tmp=root/'status.new'; tmp.write_bytes(encoded(state)); tmp.replace(root/'status.json')
    status(phase='known-state-sampling-sensitivity')
    rows=[]
    try:
        with ProcessPoolExecutor(max_workers=plan['workers'], initializer=initialize,
                initargs=(str(root/'_source'),)) as pool:
            for future in as_completed([pool.submit(probe, job) for job in jobs]):
                row=future.result(); rows.append(row)
                status(completed=len(rows), last_case=row['case']['id'], last_bank=row['bank'])
                print(json.dumps(dict(case=row['case']['id'], bank=row['bank'],
                    fixed32=row['fixed32_selected']['action'], consensus=row['consensus_selected']['action'])), flush=True)
        result=dict(complete=True, adoption=False, source=plan['source'], rows=sorted(rows,
            key=lambda r:(r['case']['id'], r['bank'])), plan_sha256=sha((root/'plan.json').read_bytes()), limit=__doc__)
        (root/'summary.json').write_bytes(encoded(result))
        status(complete=True, phase='diagnostic-complete-needs-review')
    except Exception as error:
        status(error=repr(error)); raise


if __name__ == '__main__':
    main()
