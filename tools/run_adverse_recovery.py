"""Registered long-play comparisons; stable H32 versus declared research arms.

All personality and physical rules remain unchanged. Hidden law is evaluator
only. Each arm retains actual learning, complete outcome traces and world RNG.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reflex.core import character_key, digest
from reflex.decision_loop import DecisionLoop
from reflex.fixed_horizon import FixedHorizonOwner
from reflex.model_observer import resolve_feedback
from reflex._horizon_consensus import consensus_planner
from tools.owned_checkpoint import OwnedCheckpoint, encoded, sha
from tools.run_continuation_portfolio import game_for
from tools.run_owned_value import cumulative_basis, law, sample


def seeds_for(arm):
    return tuple(range(701, 709)) if arm == 'sample8' else (701, 702, 703)


def decide(actor, game, world, arm, seeds=None):
    seeds = seeds_for(arm) if seeds is None else seeds
    if arm == 'sample8' and seeds != seeds_for(arm):
        raise ValueError('fixed nested eight-sample bank required')
    options = dict(observe=game.context, events=game.events, terminal=game.terminal,
        assess=lambda w: game.goal(w).record(), request_for=game.request,
        feedback=game.feedback, model_id=game.model_id, target=game.target,
        cumulative_preference=cumulative_basis(game, True), seeds=seeds, recovery_cache=64)
    if arm == 'stable':
        controller = FixedHorizonOwner(actor, **options)
        result = controller.decide(world)
        return result, controller.last
    if arm not in ('stratified', 'persistent', 'sample8'):
        raise ValueError('registered research arm required')
    request = game.request(game.context(world, actor.state))
    captured = {}
    planner = consensus_planner(actor, request, world, captured=captured,
        minimum_horizon=32, horizon=32, world_model='persistent' if arm == 'persistent' else 'posterior',
        purpose_estimator='conditional', tail_recovery=2, include_retry=True,
        coherent_cumulative_mean=True, sampling='stratified' if arm == 'stratified' else 'independent', **options)
    result = DecisionLoop.decide_batch([(actor, request)], False, planner)[0]
    if not result.get('deliberation', {}).get('adopted'):
        raise ValueError('incomplete research forecast; no reflex fallback')
    return result, captured


def replay(game, spec, trace, expected_owner, expected_rng):
    world = game.start()
    actor = game.actor(game.context(world), learning='segmented')
    axes = character_key(game.context(world))
    rng = np.random.default_rng(spec['seed'])
    for row in trace:
        assert world == row['before'] and digest(actor.record()) == row['before_owner']
        context = game.context(world, actor.state)
        result = DecisionLoop.decide_batch([(actor, game.request(context))], False,
                                           _counterfactual_roots=(row['action'],))[0]
        truth = law(world['tick'], spec['turns'], spec['regime'])
        assert truth == row['truth']
        response = sample(truth, rng.random())
        assert response == row['response']
        after, flow = game.transition(world, row['action'], response)
        resolve_feedback(actor, result['ticket'], game.feedback(world, after, row['action'], context, flow))
        assert after == row['after'] and digest(actor.record()) == row['owner']
        assert character_key(game.context(after)) == axes
        world = after
    assert actor.record() == expected_owner and rng.bit_generator.state == expected_rng
    return len(trace)


def run(directory, spec, arm, *, max_decisions=None, registration=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(spec=spec, arm=arm, horizon=32, seeds=list(seeds_for(arm)), recovery_cache=64,
        runner_sha256=sha(Path(__file__).read_bytes()),
        adapters={name: sha((ROOT/'tools'/name).read_bytes()) for name in (
            'run_continuation_portfolio.py', 'preparation_world.py')})
    if registration is not None:
        config['registration'] = registration
    store = OwnedCheckpoint(directory/'checkpoint.json', ('adverse-recovery', spec['id']), arm,
        configuration=config, every=1, max_decisions=max_decisions)
    g = game_for(spec)
    world = g.start()
    actor = g.actor(g.context(world), learning='segmented')
    axes = character_key(g.context(world))
    rng = np.random.default_rng(spec['seed'])
    trace, checks, elapsed = [], dict(fixed_axes=0, owner_parity=0, restored=0), 0.
    if store.saved is not None:
        saved = store.saved
        world, trace, checks, elapsed = saved['world'], saved['trace'], saved['checks'], saved['elapsed']
        actor = DecisionLoop.from_record(g.context(trace[-1]['before']), saved['owner'], predictor=actor.predictor)
        rng.bit_generator.state = saved['rng']
        assert actor.record() == saved['owner']
    start = time.monotonic()
    while not g.terminal(world):
        before = actor.record()
        detached = copy.deepcopy(actor)
        context = g.context(world, actor.state)
        prediction = detached.predictor.predict(context)
        result, captured = decide(actor, g, world, arm)
        action = result['decision']['action_id']
        truth = law(world['tick'], spec['turns'], spec['regime'])
        response = sample(truth, rng.random())
        after, flow = g.transition(world, action, response)
        feedback = g.feedback(world, after, action, result['context'], flow)
        resolve_feedback(actor, result['ticket'], feedback)
        control = DecisionLoop.decide_batch([(detached, g.request(context))], False,
                                            _counterfactual_roots=(action,))[0]
        resolve_feedback(detached, control['ticket'], feedback)
        assert detached.record() == actor.record()
        assert character_key(result['context']) == axes
        checks['fixed_axes'] += 1
        checks['owner_parity'] += 1
        saved_actor = json.loads(encoded(actor.record()))
        actor = DecisionLoop.from_record(context, saved_actor, predictor=actor.predictor)
        assert actor.record() == saved_actor
        checks['restored'] += 1
        selected_plan = result['deliberation']['selected_plan']
        proposal = captured['forecast'].audit['plans'][selected_plan]['proposal']
        trace.append(dict(tick=world['tick'], before=world, after=after, action=action,
            before_owner=digest(before), owner=digest(saved_actor), truth=truth, response=response,
            conditional_increment=sum(p*(g.goal(g.transition(world, action, z)[0]).value-g.goal(world).value)
                                      for z, p in truth.items()),
            selected_purpose=result['deliberation']['selected_purpose'],
            simulated_steps=captured['simulated_steps'],
            response_prediction=prediction,
            purpose_by_plan=captured['forecast'].purpose,
            selected_plan=selected_plan, selected_policy=proposal,
            predicted_actions=captured['modeled'][proposal][0]['actions']))
        world = after
        store.save(world, actor, rng, trace, checks, elapsed+time.monotonic()-start, terminal=g.terminal(world))
        print(json.dumps(dict(id=spec['id'], arm=arm, turn=world['tick'], complete=g.terminal(world))), flush=True)
        if store.stopping(len(trace)) and not g.terminal(world):
            return None
    actual_steps = replay(g, spec, trace, actor.record(), rng.bit_generator.state)
    result = dict(complete=True, configuration=config, final=world, score=g.goal(world).value,
        conditional_increment=sum(row['conditional_increment'] for row in trace), trace=trace,
        checks=checks, replayed_actual_steps=actual_steps,
        seconds=elapsed+time.monotonic()-start, final_owner=digest(actor.record()),
        final_rng=rng.bit_generator.state, runtime=store.contract['runtime'])
    (directory/'result.json').write_bytes(encoded(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--arm', choices=('stable', 'stratified', 'persistent', 'sample8'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-decisions', type=int)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    spec = next(s for s in plan['specs'] if s['id'] == args.case)
    registration = verify_registration(args.plan)
    run(args.output, spec, args.arm, max_decisions=args.max_decisions, registration=registration)


def verify_registration(path):
    path = Path(path)
    proof = json.loads((path.parent/'registration.json').read_bytes())
    assert proof['plan_sha256'] == sha(path.read_bytes()), 'registered plan changed'
    for name, expected in proof['sources'].items():
        assert sha((ROOT/name).read_bytes()) == expected, ('registered source changed', name)
    import os
    for name, expected in json.loads(path.read_bytes()).get('arithmetic', {}).items():
        assert os.environ.get(name) == expected, ('registered arithmetic profile changed', name)
    return dict(plan_sha256=proof['plan_sha256'], source_sha256=sha(encoded(proof['sources'])))


if __name__ == '__main__':
    main()
