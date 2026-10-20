"""Recompute an interrupted arm without replacing its already lived history.

The frozen actor still chooses every action. Only its factual prediction may
be normalized to its original value, within the existing eight-ULP replay
bound, for the exact original predictor and current actual tick. Hypothetical
future ticks never read the retained history. Any different action, observation,
or resulting owner stops the continuation before the checkpoint is written.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


# Forecast diagnostics are recalculated on the continuation host. Experience
# and learner states are exact, not approximately equivalent.
EXPERIENCE = ('tick', 'before', 'after', 'action', 'response', 'goal', 'law',
              'truth', 'mode', 'owner', 'before_owner', 'needs', 'checkpoint')


def check_step(actual, original):
    for key in EXPERIENCE:
        assert actual[key] == original[key], ('retained experience changed', original['tick'], key)


def verify(root, row, boundary, retained_sha):
    root = Path(root)
    raw = (root / 'retained_execution.json').read_bytes()
    assert sha(raw) == retained_sha
    envelope = json.loads(raw)
    saved = envelope['payload']
    assert sha(encoded(saved)) == envelope['sha256']
    trace = saved['trace']
    assert boundary['original_turns'] <= len(trace) <= row['executed']
    for actual, original in zip(row['trace'], trace):
        # The saved published diagnostic fields also remain unchanged.
        assert all(actual[k] == v for k, v in original.items()), 'retained trace replaced'
    receipt = json.loads((root / 'retained_history_replay.json').read_bytes())
    assert receipt['version'] == 'retained-history-replay-v1'
    assert receipt['complete'] and receipt['checked_until'] == len(trace)
    assert receipt['start'] == boundary['resume_at']
    assert receipt['retained_checkpoint_sha256'] == sha(raw)
    # Validate every numeric normalization rather than trusting a flag.
    spec = importlib.util.spec_from_file_location(
        'retained_probability_check', Path(__file__).with_name('audit_recorded_forecasts.py'))
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    for item in receipt['normalizations']:
        tick = item['tick']
        assert receipt['start'] <= tick < len(trace)
        assert checker.probability_differences(trace[tick]['law'], item['fresh']) == item['differences']
    return receipt


def main():
    script, *args = sys.argv[1:]
    root = Path(args[args.index('--root') + 1]).resolve()
    source = (root / '_source').resolve()
    assert Path(script).resolve() == source / 'tools/run_horizon_study.py'
    sys.path.insert(0, str(source))
    from reflex.core import digest
    from reflex.event_reader import EventReader
    from tools.owned_checkpoint import OwnedCheckpoint
    assert Path(sys.modules['tools.owned_checkpoint'].__file__).resolve().is_relative_to(source)
    spec = importlib.util.spec_from_file_location(
        'retained_probability_check', Path(__file__).with_name('audit_recorded_forecasts.py'))
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    original_raw = (root / 'retained_execution.json').read_bytes()
    original = json.loads(original_raw)
    assert sha(encoded(original['payload'])) == original['sha256']
    retained = original['payload']['trace']
    checkpoint = json.loads((root / 'checkpoint.json').read_bytes())
    start = len(checkpoint['payload']['trace'])
    assert checkpoint['payload']['trace'] == retained[:start]
    assert checkpoint['payload']['contract'] == original['payload']['contract']
    assert start <= len(retained)
    receipt = dict(version='retained-history-replay-v1', start=start,
                   retained_checkpoint_sha256=sha(original_raw), checked_until=start,
                   checker_sha256=sha(Path(__file__).read_bytes()),
                   complete=start == len(retained), normalizations=[])
    target = root / 'retained_history_replay.json'

    def write():
        temporary = target.with_suffix('.new')
        temporary.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf8')
        temporary.replace(target)

    actual_tick = start
    original_predict, original_save = EventReader.predict, OwnedCheckpoint.save

    def predict(self, context):
        fresh = original_predict(self, context)
        tick = context['tick']
        if tick != actual_tick or not start <= tick < len(retained):
            return fresh
        step = retained[tick]
        if digest(self.record()) != digest(step['checkpoint']['predictor']):
            return fresh
        changes = checker.probability_differences(step['law'], fresh)
        if changes:
            item = dict(tick=tick, fresh=fresh, differences=changes)
            if item not in receipt['normalizations']:
                receipt['normalizations'].append(item)
        # Preserve recomputed key order, just as the existing replay auditor.
        return {key: step['law'][key] for key in fresh}

    def save(self, world, owner, rng, trace, checks, elapsed, *, terminal):
        nonlocal actual_tick
        tick = len(trace) - 1
        assert tick == actual_tick, 'unexpected factual replay timeline'
        if tick < len(retained):
            check_step(trace[-1], retained[tick])
            # Decisions/learning above have been freshly recomputed. Retain the
            # original diagnostic observations too, not a new host's rounding.
            trace[-1] = copy.deepcopy(retained[tick])
            receipt['checked_until'] = tick + 1
            receipt['complete'] = tick + 1 == len(retained)
            write()
        actual_tick = tick + 1
        return original_save(self, world, owner, rng, trace, checks, elapsed, terminal=terminal)

    EventReader.predict, OwnedCheckpoint.save = predict, save
    write()
    sys.argv = [script, *args]
    try:
        runpy.run_path(script, run_name='__main__')
    finally:
        EventReader.predict, OwnedCheckpoint.save = original_predict, original_save
        write()


if __name__ == '__main__':
    main()
