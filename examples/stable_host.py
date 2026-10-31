"""Minimal host for the published v0.1.0 wheel; no research-runner imports.

This six-step world demonstrates the connection, not intelligence or speed.
Run with --save demo.json, or --stop-after 3 --save demo.json followed by
--resume demo.json --save resumed.json in a new Python process.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from reflex.adaptive_dynamics import PublicEvent
from reflex.core import FEATURES, Policy, character_key, digest
from reflex.cumulative_preference import CumulativePreference
from reflex.decision_loop import DecisionLoop, Request
from reflex.event_reader import EventReader
from reflex.examples import action, context, effect
from reflex.fixed_horizon import FixedHorizonOwner
from reflex.judgment import Binding, ReadControl
from reflex.model_observer import ModelFeedback
from reflex.planning import vector
from reflex.pressure import NeedPressure
from reflex.progress import Activity, ProgressWatch, PurposeFeedback, PurposeRequest
from reflex.purpose_plan import Goal

RULES = 'stable-host-example-v1'
TARGET = 'earned-score'
RESPONSES = ('a', 'b')
END = 6


def initial_world():
    return dict(tick=0, score=0., signal='a')


def outcome(observation, move, response):
    gain = .1 if move == response else -.05
    return effect(gain, needs={'growth': gain}, values={'achievement': gain})


def observe(world, intent=None):
    moves = [] if terminal(world) else [
        action(k, *[dict(outcome(None, k, z), p=.5) for z in RESPONSES])
        for k in RESPONSES]
    c = context('stable-host', moves, values={'achievement': .7})
    c.update(tick=world['tick'], facts={'public_world': json.dumps(world, sort_keys=True)})
    if intent is not None:
        c['state'] = copy.deepcopy(intent)
    return c


def models(observation):
    return {'a-usual': {'a': .9, 'b': .1}, 'b-usual': {'a': .1, 'b': .9}}


def request_for(c):
    world = json.loads(c['facts']['public_world'])
    activities = {a['id']: Activity('uncertain', 'public-rule', condition='same-rules')
                  for a in c['actions']}
    return Request(c, {k: Binding(k, 'same-rules', FEATURES) for k in activities},
                   {k: () for k in activities}, threatened=True,
                   purpose=PurposeRequest(world['score'], 0., activities))


def transition(world, move, response):
    flow = outcome(None, move, response)
    return dict(tick=world['tick']+1, score=world['score']+flow['objective'], signal=response), flow


def events(world, move):
    # Enumerate publicly possible responses, never use the actual world's RNG.
    return [PublicEvent(z, *transition(world, move, z)) for z in RESPONSES]


def terminal(world):
    return world['tick'] >= END


def assess(world):
    return Goal(TARGET, 'scored' if terminal(world) else 'running', world['score']).record()


def feedback(before, after, move, observation, flow):
    return ModelFeedback(
        PurposeFeedback(after['score'], 0., completed=terminal(after)),
        needs={'growth': flow['needs']['growth']}, observed=vector(flow),
        event={'revealed_action': after['signal']})


def make_controller(predictor=None, owner_type=FixedHorizonOwner):
    c = observe(initial_world())
    if predictor is None:
        predictor = EventReader(c['scope'], ('a-usual', 'b-usual'), models, outcome,
            legal_for=lambda c: RESPONSES, model_id=RULES, known_conditionals=True,
            learning='segmented')
    actor = DecisionLoop(c,
        policy=Policy(principle_priority='personal', commitments={character_key(c): .5}),
        predictor=predictor, read_control=ReadControl(max_nodes=64),
        pressure=NeedPressure(c['scope']), progress=ProgressWatch(c['scope']))
    return owner_type(actor, observe=observe, events=events, terminal=terminal,
        assess=assess, request_for=request_for, feedback=feedback, model_id=RULES,
        target=TARGET, cumulative_preference=CumulativePreference(
            TARGET, 'raw-score', (('values.achievement', 1.),)),
        horizon=32, seeds=(701, 702, 703), recovery_cache=64)


def main(controller_factory=make_controller):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after', type=int, choices=range(END+1), default=END)
    args = parser.parse_args()
    controller = controller_factory()
    world, trace = initial_world(), []
    rng = np.random.default_rng(987)
    if args.resume:
        saved = json.loads(args.resume.read_text(encoding='utf8'))
        if saved['format'] != RULES:
            raise ValueError('host save format changed')
        world, trace = saved['world'], saved['trace']
        controller = controller.restored(world, saved['owner'])
        rng.bit_generator.state = saved['rng']
    while not terminal(world) and world['tick'] < args.stop_after:
        result = controller.decide(world)
        move = result['decision']['action_id']
        # This private changing law belongs only to the real host, not the AI.
        response = 'a' if rng.random() < (.8 if world['tick'] < 3 else .2) else 'b'
        after, flow = transition(world, move, response)
        controller.resolve(result['ticket'], feedback(world, after, move, result['context'], flow))
        world = after
        trace.append(move)
    bundle = dict(format=RULES, world=world, rng=rng.bit_generator.state,
                  owner=controller.checkpoint(world), trace=trace)
    # One resolved boundary: the world, RNG and factual owner are saved together.
    temp = args.save.with_name(args.save.name+'.tmp')
    temp.write_text(json.dumps(bundle, sort_keys=True, allow_nan=False), encoding='utf8')
    temp.replace(args.save)
    print(json.dumps(dict(complete=terminal(world), tick=world['tick'], trace=trace,
                         world=world, owner_sha256=digest(controller.actor.record()))))


if __name__ == '__main__':
    main()
