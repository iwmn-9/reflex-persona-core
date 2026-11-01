"""Actual-experience intervention after a registered, replayed owner prefix.

Truth belongs to this evaluator, never to the controller. The intervention
stops actual experience updates, not needs, purpose progress or imagined learning.
"""
import copy
import json
import os
import time
from pathlib import Path

import numpy as np

from reflex.core import character_key, digest
from reflex.decision_loop import DecisionLoop
from reflex.model_observer import resolve_feedback
from tools.completion_metrics import summarize_trace
from tools.frozen_experience import FrozenExperience
from tools.owned_checkpoint import OwnedCheckpoint, encoded, sha
from tools.run_adverse_recovery import decide, replay
from tools.run_continuation_portfolio import game_for
from tools.run_owned_value import law, sample

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.new')
    temporary.write_bytes(encoded(value)); temporary.replace(path)


def restore(game, payload):
    world = copy.deepcopy(payload['world'])
    template = game.actor(game.context(game.start()), learning='segmented')
    actor = DecisionLoop.from_record(game.context(payload['trace'][-1]['before']),
                                    payload['owner'], predictor=template.predictor)
    rng = np.random.default_rng()
    rng.bit_generator.state = payload['rng']
    assert actor.record() == payload['owner']
    return world, actor, rng


def load_origin(path, spec, boundary):
    envelope = json.loads(Path(path).read_bytes()); origin = envelope['payload']
    assert sha(encoded(origin)) == envelope['sha256']
    config = origin['contract']['configuration']
    assert config['spec'] == spec and config['arm'] == 'expected'
    assert config['continuation_scope'] == 'shared'
    assert origin['world']['tick'] == len(origin['trace']) == boundary
    assert origin['owner']['last_tick'] == boundary-1
    assert origin['checks'] == dict(fixed_axes=boundary, owner_parity=boundary, restored=boundary)
    replay(game_for(spec), spec, origin['trace'], origin['owner'], origin['rng'])
    return origin


def truth_after(spec, tick, boundary, world_mode):
    if world_mode not in ('changed', 'steady'):
        raise ValueError('registered world intervention required')
    if tick < boundary:
        return law(tick, spec['turns'], spec['regime'])
    previous = law(boundary-1, spec['turns'], spec['regime'])
    dominant = max(previous, key=previous.get)
    # Rotate an event category, not a game/action-specific policy.
    top = str((int(dominant)+2) % 3) if world_mode == 'changed' else dominant
    return {key: .8 if key == top else .1 for key in ('0', '1', '2')}


def forced_feedback(game, actor, world, action, response, frozen):
    context = game.context(world, actor.state)
    result = DecisionLoop.decide_batch([(actor, game.request(context))], False,
                                      _counterfactual_roots=(action,))[0]
    after, flow = game.transition(world, action, response)
    resolve_feedback(actor, result['ticket'], game.feedback(world, after, action, context, flow))
    if frozen is not None:
        actor = frozen.after_feedback(actor, context)
    return after, actor


def audit_branch(game, spec, origin, trace, owner, rng_state, mode, experience, boundary):
    assert trace[:boundary] == origin['trace']
    world, actor, rng = restore(game, origin)
    frozen = FrozenExperience(actor) if experience == 'frozen' else None
    axes = character_key(game.context(world))
    for row in trace[boundary:]:
        assert world == row['before'] and digest(actor.record()) == row['before_owner']
        prediction = copy.deepcopy(actor).predictor.predict(game.context(world, actor.state))
        assert prediction == row['response_prediction']
        truth = truth_after(spec, world['tick'], boundary, mode)
        assert truth == row['truth']
        uniform = rng.random()
        assert uniform == row['uniform'] and sample(truth, uniform) == row['response']
        increment = sum(p*(game.goal(game.transition(world, row['action'], z)[0]).value-game.goal(world).value)
                        for z, p in truth.items())
        assert increment == row['conditional_increment']
        world, actor = forced_feedback(game, actor, world, row['action'], row['response'], frozen)
        assert world == row['after'] and digest(actor.record()) == row['owner']
        assert character_key(game.context(world)) == axes
    assert actor.record() == owner and rng.bit_generator.state == rng_state
    return len(trace)-boundary


def run_branch(directory, spec, origin_path, world_mode, experience, registration,
               *, max_decisions=None, boundary=32, window=32):
    if experience not in ('learning', 'frozen'):
        raise ValueError('registered actual-experience control required')
    if boundary < 1 or window < 1 or boundary+window > spec['turns']:
        raise ValueError('bounded actual continuation required')
    truth_after(spec, boundary, boundary, world_mode)
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    origin = load_origin(origin_path, spec, boundary)
    origin_config = origin['contract']['configuration']
    assert origin_config['registration'] == registration
    assert origin_config['horizon'] == 32 and origin_config['seeds'] == [701,702,703]
    assert origin_config['recovery_cache'] == 64
    assert origin_config['runner_sha256'] == sha((ROOT/'tools/run_adverse_recovery.py').read_bytes())
    g = game_for(spec); world, actor, rng = restore(g, origin)
    if g.terminal(world):
        raise ValueError('terminal origin must be retained as an unavailable registered case')
    frozen = FrozenExperience(actor) if experience == 'frozen' else None
    config = dict(spec=spec, world_mode=world_mode, experience=experience,
        boundary=boundary, window=window, origin_sha256=sha(Path(origin_path).read_bytes()),
        horizon=32, seeds=[701,702,703], continuation_scope='shared', registration=registration,
        sources={name:sha((ROOT/name).read_bytes()) for name in (
            'tools/recovery_benefit.py', 'tools/frozen_experience.py', 'tools/run_adverse_recovery.py',
            'tools/run_continuation_portfolio.py', 'tools/preparation_world.py')})
    store = OwnedCheckpoint(directory/'checkpoint.json', ('recovery-benefit', spec['id'], world_mode),
        experience, configuration=config, every=1, max_decisions=max_decisions)
    assert store.contract['runtime'] == origin['contract']['runtime']
    assert store.contract['sources'] == origin['contract']['sources']
    trace = copy.deepcopy(origin['trace']); checks = copy.deepcopy(origin['checks']); elapsed = 0.
    if store.saved is not None:
        world, actor, rng = restore(g, store.saved)
        trace, checks, elapsed = store.saved['trace'], store.saved['checks'], store.saved['elapsed']
        audit_branch(g, spec, origin, trace, actor.record(), rng.bit_generator.state,
                     world_mode, experience, boundary)
    else:
        # A fresh branch starts at the boundary; max_decisions counts new work only.
        store.initial_tick = boundary
    axes = character_key(g.context(world)); start = time.monotonic()
    end = boundary+window
    def progress(phase, complete=False):
        save(directory/'progress.json', dict(case=spec['id'], world_mode=world_mode, experience=experience,
            actual_steps=len(trace)-boundary, scheduled_steps=window, complete=complete,
            phase=phase, pid=os.getpid(), updated=time.time()))
    while world['tick'] < end and not g.terminal(world):
        before = actor.record(); detached = copy.deepcopy(actor)
        context = g.context(world, actor.state)
        prediction = detached.predictor.predict(context)
        progress('deciding')
        result, captured = decide(actor, g, world, 'expected', continuation_scope='shared')
        action = result['decision']['action_id']
        truth = truth_after(spec, world['tick'], boundary, world_mode)
        uniform = rng.random(); response = sample(truth, uniform)
        after, flow = g.transition(world, action, response)
        resolve_feedback(actor, result['ticket'], g.feedback(world, after, action, result['context'], flow))
        if frozen is not None:
            actor = frozen.after_feedback(actor, context)
        control_world, detached = forced_feedback(g, detached, world, action, response, frozen)
        assert detached.record() == actor.record() and control_world == after
        assert character_key(result['context']) == axes
        saved_actor = json.loads(encoded(actor.record()))
        actor = DecisionLoop.from_record(context, saved_actor, predictor=actor.predictor)
        assert actor.record() == saved_actor
        for key in checks: checks[key] += 1
        trace.append(dict(tick=world['tick'], before=world, after=after, action=action,
            before_owner=digest(before), owner=digest(saved_actor), truth=truth, response=response,
            uniform=uniform, response_prediction=prediction,
            conditional_increment=sum(p*(g.goal(g.transition(world, action, z)[0]).value-g.goal(world).value)
                                      for z,p in truth.items()),
            selected_plan=result['deliberation']['selected_plan'],
            simulated_steps=captured['simulated_steps']))
        world = after
        finished = world['tick'] == end or g.terminal(world)
        store.save(world, actor, rng, trace, checks, elapsed+time.monotonic()-start, terminal=finished)
        progress('resolved')
        if store.stopping(len(trace)) and not finished:
            return None
    count = audit_branch(g, spec, origin, trace, actor.record(), rng.bit_generator.state,
                         world_mode, experience, boundary)
    assert checks == dict(fixed_axes=len(trace), owner_parity=len(trace), restored=len(trace))
    metrics = summarize_trace(trace, scheduled_turns=end, terminal=g.terminal(world),
        truth_schedule=[truth_after(spec,t,boundary,world_mode) for t in range(end)])
    # summarize_trace.complete describes a whole-world terminal. This is a
    # finite observation window, so expose both concepts without conflating them.
    metrics['world_terminal'] = metrics.pop('complete')
    metrics['window_complete'] = count == window
    result = dict(complete=True, window_complete=count==window, world_terminal=g.terminal(world),
        configuration=config, runtime=store.contract['runtime'], final=world,
        origin_owner=digest(origin['owner']), origin_world=origin['world'], origin_rng=origin['rng'],
        score_increment=g.goal(world).value-g.goal(origin['world']).value,
        conditional_increment=sum(t['conditional_increment'] for t in trace[boundary:]),
        actual_steps=count, trace=trace, checks=checks, final_owner=digest(actor.record()),
        final_rng=rng.bit_generator.state, metrics=metrics, seconds=elapsed+time.monotonic()-start)
    save(directory/'result.json', result); progress('complete', True)
    return result
