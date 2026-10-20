"""Exact future choices versus rolling deadlines in continuing public worlds.

Both arms use the existing owned conditional tree and the SAME personality.
Only the actual retained evaluation deadline differs. Future choices use the
remaining tree horizon, so the fixed-window arm reproduces that time contract
after real feedback. Public payout changes are never passed to forecasts.

This is a mechanism diagnosis, not an efficient replacement for sampled
planning in large branching games or independent game-family confirmation.
"""
import argparse
import copy
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reflex.closed_loop import DecisionWindow, owned_rollout
from reflex.core import character_key, digest
from reflex.decision_loop import DecisionLoop
from reflex.model_observer import resolve_feedback
from tools.owned_checkpoint import OwnedCheckpoint, encoded, sha
from tools.preparation_world import DelayedYield
from tools.run_owned_value import cumulative_basis


def execution(root, delay, shock, mode, *, turns=128, horizon=16, max_decisions=None):
    if mode not in ('rolling', 'fixed-window'):
        raise ValueError('registered deadline mode required')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config = dict(delay=delay, shock=shock, mode=mode, turns=turns, horizon=horizon,
                  driver_sha256=sha(Path(__file__).read_bytes()),
                  adapter_sha256=sha(Path(__file__).with_name('preparation_world.py').read_bytes()))
    g = DelayedYield(delay, turns=turns)
    w = g.start()
    actor = g.actor(g.context(w), learning='segmented')
    fixed = character_key(g.context(w))
    clock = DecisionWindow(g.context(w), horizon)
    store = OwnedCheckpoint(root / 'checkpoint.json',
        ('delayed-exact-window', 0, 0, shock, turns, horizon, 0, 0),
        mode, configuration=config, every=1, max_decisions=max_decisions)
    trace = []
    rng = np.random.default_rng(0)
    checks = dict(fixed_axes=0, owner_parity=0, clock=0)
    started = time.monotonic()
    if store.saved is not None:
        saved = store.saved
        w, trace, checks = saved['world'], saved['trace'], saved['checks']
        actor = DecisionLoop.from_record(g.context(trace[-1]['before']),
            saved['owner'], predictor=actor.predictor)
        clock = DecisionWindow.from_record(g.context(w), trace[-1]['clock'])
        rng.bit_generator.state = saved['rng']
        started -= saved['elapsed']
    while not g.terminal(w):
        c = g.context(w)
        request = g.request(c)
        before = digest(actor.record())
        detached = copy.deepcopy(actor)
        remaining = clock.remaining(c) if mode == 'fixed-window' else horizon
        ctrl = owned_rollout(actor, observe=g.context, events=g.events,
            terminal=g.terminal, assess=lambda w: g.goal(w).record(), record=lambda w: w,
            request_for=g.request, feedback=g.feedback, model_id=g.model_id, target=g.target,
            tree_steps=remaining, tail_steps=0, world_model='posterior', future_choices='planned',
            cumulative_preference=cumulative_basis(g, True))
        capture = {}
        planner = ctrl.planner(actor, request, w, captured=capture)

        def complete(contexts, decisions):
            forecast = planner(contexts, decisions)
            if forecast is None:
                raise ValueError('exact tree exceeded budget; no reflex fallback or partial success')
            return forecast

        result = DecisionLoop.decide_batch([(actor, request)], False, complete)[0]
        key = result['decision']['action_id']
        after, flow = g.transition(w, key, '0')
        if shock and after['tick'] == turns // 2:
            after['payout'] = 2.
        feedback = g.feedback(w, after, key, c, flow)
        resolve_feedback(actor, result['ticket'], feedback)
        replay = DecisionLoop.decide_batch([(detached, request)], False,
                                           _counterfactual_roots=(key,))[0]
        resolve_feedback(detached, replay['ticket'], feedback)
        assert digest(actor.record()) == digest(detached.record())
        assert character_key(result['context']) == fixed
        clock.advance(c)  # only the resolved real transition advances the clock
        assert clock.used == len(trace) + 1
        trace.append(dict(tick=w['tick'], before=w, action=key, after=after,
            before_owner=before, owner=digest(actor.record()), horizon=remaining,
            clock=clock.record(), selected_purpose=result['deliberation']['selected_purpose'],
            modeled_actions=list(capture['node']['returns'][0][2]),
            tree_nodes=ctrl.last['nodes']))
        w = after
        for name in checks:
            checks[name] += 1
        store.save(w, actor, rng, trace, checks, time.monotonic()-started, terminal=g.terminal(w))
        if len(trace) % 16 == 0:
            print(json.dumps(dict(delay=delay, shock=shock, mode=mode,
                turn=len(trace), cash=w['cash'])), flush=True)
        if store.stopping(len(trace)) and not g.terminal(w):
            return None
    state = g.start()
    for i, step in enumerate(trace):
        assert step['before'] == state and step['clock']['used'] == i + 1
        assert step['horizon'] == (horizon - i % horizon if mode == 'fixed-window' else horizon)
        state, _ = g.transition(state, step['action'], '0')
        if shock and state['tick'] == turns // 2:
            state['payout'] = 2.
        assert state == step['after']
    assert state == w and len(trace) == turns
    assert checks == dict(fixed_axes=turns, owner_parity=turns, clock=turns)
    result = dict(configuration=config, complete=True, turns=turns, final=w,
        score=g.goal(w).value, trace=trace, checks=checks, physical_replay=turns,
        seconds=time.monotonic()-started,
        preparation_ticks=[t['tick'] for t in trace if t['action'] == 'y'])
    (root / 'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    return result


def pair(root, spec):
    rows = []
    for mode in ('rolling', 'fixed-window'):
        path = Path(root) / f"delay-{spec['delay']}-shock-{int(spec['shock'])}" / mode
        dest = path / 'result.json'
        row = json.loads(dest.read_bytes()) if dest.exists() else execution(path, **spec, mode=mode)
        saved = json.loads((path / 'checkpoint.json').read_bytes())
        assert sha(encoded(saved['payload'])) == saved['sha256']
        assert saved['payload']['trace'] == row['trace']
        assert saved['payload']['world'] == row['final']
        assert saved['payload']['contract']['configuration'] == row['configuration']
        assert all(row['configuration'][k] == v for k, v in dict(spec, mode=mode).items())
        assert row['complete'] and row['turns'] == row['physical_replay'] == 128
        rows.append(row)
    comparison = dict(**spec, scores=[r['score'] for r in rows],
        cash=[r['final']['cash'] for r in rows], delta=rows[1]['score']-rows[0]['score'],
        preparation_ticks=[r['preparation_ticks'] for r in rows], seconds=[r['seconds'] for r in rows])
    print(json.dumps(comparison), flush=True)
    return comparison


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--frozen', action='store_true')
    a = p.parse_args()
    root = Path(a.root).resolve()
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        dispatch(root, Path(__file__).name, [])
        return
    for path, expected in json.loads((root/'source_snapshot.json').read_bytes()).items():
        assert hashlib.sha256((root/'_source'/path).read_bytes()).hexdigest() == expected
    plan = dict(cases=[dict(delay=d, shock=s) for d in (3, 7, 11) for s in (False, True)],
        turns=128, horizon=16, modes=['rolling', 'fixed-window'],
        semantics=__doc__, acceptance='All six pairs complete and audited, including adverse cases. Mechanism diagnosis only; no global adoption.')
    reg = root / 'registration.json'
    if reg.exists():
        assert json.loads(reg.read_bytes()) == plan
    else:
        reg.write_text(json.dumps(plan, indent=2)+'\n', encoding='utf8')
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows = [f.result() for f in as_completed([pool.submit(pair, root, s) for s in plan['cases']])]
    prefixes = 0
    for delay in (3, 7, 11):
        for mode in plan['modes']:
            runs = [json.loads((root/f'delay-{delay}-shock-{s}'/mode/'result.json').read_bytes()) for s in (0, 1)]
            for x, y in zip(runs[0]['trace'][:64], runs[1]['trace'][:64]):
                assert all(x[k] == y[k] for k in ('before', 'action', 'before_owner', 'clock', 'horizon'))
            prefixes += 1
    (root/'summary.json').write_text(json.dumps(dict(complete=True, comparisons=rows,
        identical_pre_shock_prefixes=prefixes), indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
