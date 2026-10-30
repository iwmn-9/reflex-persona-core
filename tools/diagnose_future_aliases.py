"""Observe coarse future choices without changing the registered forecast.

Only a detached copy is offered the existing goal-tie refinement. Its choice is
recorded, never committed to the original hypothetical or factual trajectory.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    worker, probe, source, output = map(Path, (args.worker, args.probe, args.input, args.output))
    assert not output.exists()
    source_row = json.loads(source.read_bytes())
    assert sha(source) == args.input_sha256
    os.environ.update(json.loads((worker / args.plan).read_bytes())['arithmetic'])
    os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    sys.path.insert(0, str(worker))
    from reflex.core import digest
    from reflex.decision_loop import DecisionLoop
    from reflex.goal_ties import has_aliases, reflex_goal_planner
    from tools import run_adverse_recovery as runner
    game = runner.game_for(source_row['configuration']['spec'])
    old_decide, old_batch = runner.decide, DecisionLoop.decide_batch
    observations, seen, enabled = [], set(), False
    captured_result = {}

    def observed_batch(items, stochastic=True, planner=None, *, _counterfactual_roots=None):
        snapshots = []
        if enabled and planner is None and _counterfactual_roots is None and not stochastic:
            for own, request in items:
                if has_aliases(request.context):
                    stamp = digest([own.record(), request.context])
                    if stamp not in seen:
                        seen.add(stamp)
                        snapshots.append((stamp, copy.deepcopy(own), copy.deepcopy(request)))
        original = old_batch(items, stochastic, planner, _counterfactual_roots=_counterfactual_roots)
        # Registered portfolio invokes one hypothetical owner at a time.
        if snapshots:
            assert len(items) == len(original) == len(snapshots) == 1
        for stamp, own, request in snapshots:
            public = game.world(request.context)
            refine = reflex_goal_planner(own, request, public, events=game.events,
                                        assess=lambda w: game.goal(w).record(), target=game.target)
            trial = old_batch([(own, request)], False, refine)[0] if refine is not None else None
            observations.append(dict(owner_and_context=stamp, tick=request.context['tick'], public=public,
                original_action=original[0]['decision']['action_id'],
                detached_action=None if trial is None else trial['decision']['action_id'],
                detached_goal_tie=None if trial is None else trial.get('deliberation', {}).get('goal_ties'),
                original_context=original[0]['context'],
                note='Detached refinement only; original decision and feedback remain unchanged.'))
        return original

    def observed_decide(*pargs, **kwargs):
        nonlocal enabled
        enabled = True
        try:
            result, cap = old_decide(*pargs, **kwargs)
        finally:
            enabled = False
        captured_result.update(modeled=cap['modeled'], original_decision=result['decision'])
        return result, cap

    spec = importlib.util.spec_from_file_location('registered_probe', probe)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    probe_output = output.with_suffix('.registered.json')
    assert not probe_output.exists()
    probe_args = argparse.Namespace(worker=str(worker), plan=args.plan, input=str(source),
        input_sha256=args.input_sha256, tick=args.tick, start=701, samples=3, output=str(probe_output))
    runner.decide = observed_decide
    DecisionLoop.decide_batch = staticmethod(observed_batch)
    try:
        module.main(probe_args)
    finally:
        runner.decide = old_decide
        DecisionLoop.decide_batch = staticmethod(old_batch)
    reproduced = json.loads(probe_output.read_bytes())
    assert reproduced['exact_registered_forecast_reproduced']
    baseline = json.loads(Path(args.original_probe).read_bytes())
    for key in ('decision', 'deliberation', 'purpose_by_plan', 'personal_ranking', 'selection_inputs',
                'registration', 'runtime', 'factual_owner', 'input_sha256'):
        assert reproduced[key] == baseline[key], ('observational instrumentation changed registered output', key)
    output.write_text(json.dumps(dict(adoption=False, no_forecast_change=True,
        source_sha256=sha(__file__), probe_sha256=sha(probe), source_input_sha256=sha(source),
        original_probe_sha256=sha(args.original_probe), observed_probe_sha256=sha(probe_output),
        factual_owner=reproduced['factual_owner'], registration=reproduced['registration'],
        observations=observations, modeled=captured_result['modeled'],
        changed_detached_choices=sum(o['detached_action'] is not None and
            o['detached_action'] != o['original_action'] for o in observations),
        scope='A mechanism diagnosis on an already exposed root, not a new actual world or an adopted controller.'),
        sort_keys=True, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(observed_alias_states=len(observations),
        changed_detached_choices=sum(o['detached_action'] is not None and
            o['detached_action'] != o['original_action'] for o in observations))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('worker', 'probe', 'plan', 'input', 'input-sha256', 'original-probe', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--tick', type=int, required=True)
    main(parser.parse_args())
