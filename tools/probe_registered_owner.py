"""Replay the registered candidate's actual owner before a diagnostic forecast.

Run in a fresh process outside the frozen worker. The worker's actor factory,
controller, arithmetic and source registration are authoritative. Vary only a
predeclared diagnostic model bank; never continue an actual world or adopt.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inputs(path, expected_sha256, tick, start, samples):
    raw = Path(path).read_bytes()
    if sha(raw) != expected_sha256:
        raise ValueError('registered diagnostic input changed')
    row = json.loads(raw)
    cfg = row['configuration']
    if (not row['complete'] or cfg['horizon'] != 32
            or len(row['trace']) != cfg['spec']['turns']
            or row['replayed_actual_steps'] != cfg['spec']['turns']
            or [t['tick'] for t in row['trace']] != list(range(cfg['spec']['turns']))):
        raise ValueError('complete registered H32 history required')
    if type(tick) is not int or not 0 <= tick < len(row['trace']):
        raise ValueError('saved factual root required')
    if start not in (701, 9101, 9201) or samples not in (3, 8):
        raise ValueError('predeclared diagnostic bank required')
    return row, tuple(range(start, start + samples))


def personal_ranking(row):
    """Reproduce selection from its retained numeric packet, without forecasting."""
    from reflex.core import Policy, NEEDS
    from reflex.deliberation import JointForecast, select
    from reflex.valuation import ReturnFrame
    from reflex.return_means import ReturnMeans
    packet, audit = row['selection_inputs'], row['deliberation']
    if any(packet['extra_selectors'].values()):
        raise ValueError('unsupported selection extension; never infer a personal ranking')
    metadata = audit['metadata']
    frame = metadata['return_frame']
    means = metadata.get('cumulative_mean_correction')
    assert frame['format'] == 'return-frame-v1'
    frame = ReturnFrame(tuple(frame['owners']), tuple(map(tuple, frame['effects'])),
                        tuple(frame['root_terms']), frame['unit'])
    if means is not None:
        assert means['format'] == 'return-means-v1'
        means = ReturnMeans(tuple(means['owners']), tuple(means['plans']),
            tuple(tuple(map(tuple, owner)) for owner in means['corrections']))

    class Capture(Policy):
        def decide(self, batch, *args, **kwargs):
            result = super().decide(batch, *args, **kwargs)
            self.captured = batch, result
            return result

    policy = Capture(**packet['policy'])
    forecast = JointForecast(tuple(packet['forecast_contexts']),
        {k: tuple(v) for k, v in row['roots'].items()}, row['purpose_by_plan'],
        audit['horizon'], audit['max_regret'], metadata, packet['target'],
        return_frame=frame, return_means=means, outcome_limit=packet['outcome_limit'])
    base = packet['base_context']
    allowed = {a['id'] for a in base['actions'] if a['legal'] and not a['known_failure']}
    progress = packet['progress']
    if progress and progress['applied']:
        allowed -= set(progress['blocked'])
    decisions, reproduced = select([base], forecast, policy, [allowed])
    assert decisions[0] == row['decision']
    for key in ('selected_plan', 'selected_purpose', 'best_purpose', 'max_regret',
                'progress_excluded', 'purpose_excluded', 'subjective_regret'):
        assert reproduced[key] == audit[key], ('selection replay differs', key)
    batch, decision = policy.captured
    names = ('objective_mean', 'primary_need_effect', 'top_value_effect', 'downside_loss',
             'cost_mean', 'familiarity', 'switch_cost', 'confidence')
    return dict(exact_selection_reproduced=True,
        primary_need=None if decision.primary[0] < 0 else NEEDS[int(decision.primary[0])],
        mode='principle' if decision.mode[0] else 'need',
        plans={key: dict(personal_score=float(decision.scores[0, i]),
            eligible=bool(decision.eligible[0, i]),
            features=dict(zip(names, decision.features[0, i].tolist())))
            for i, key in enumerate(batch.ids[0])},
        scope='Exact personal ranking under this modeled future; not proof of true future value or intentional real-world loss.')


def main(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('do not replace a diagnostic result')
    saved, seeds = inputs(args.input, args.input_sha256, args.tick, args.start, args.samples)
    worker = Path(args.worker).resolve()
    plan_path = worker / args.plan
    plan = json.loads(plan_path.read_bytes())
    cfg = saved['configuration']
    if cfg['arm'] != plan['candidate_arm'] or cfg['spec'] not in plan['specs']:
        raise ValueError('candidate and condition must belong to the frozen study')
    sys.path.insert(0, str(worker))
    import numpy as np
    from reflex.core import character_key, digest
    from reflex.decision_loop import DecisionLoop
    from reflex.model_observer import resolve_feedback
    from tools.run_adverse_recovery import actor_for, decide, game_for, verify_registration
    from tools.run_owned_value import law, sample
    from tools.owned_checkpoint import encoded
    registration = verify_registration(plan_path)
    if registration != cfg['registration']:
        raise ValueError('input has a different source registration')
    runtime = dict(python=sys.version, numpy=np.__version__, system=platform.system(), machine=platform.machine())
    if runtime != saved['runtime']:
        raise ValueError('original runtime required; no tolerance-based prefix recovery')
    if sha((worker / 'tools/run_adverse_recovery.py').read_bytes()) != cfg['runner_sha256']:
        raise ValueError('registered actor factory and controller required')

    started = time.monotonic()
    spec, arm = cfg['spec'], cfg['arm']
    game = game_for(spec)
    world = game.start()
    actor = actor_for(game, game.context(world), arm)
    persona = character_key(game.context(world))
    rng = np.random.default_rng(spec['seed'])
    for row in saved['trace'][:args.tick]:
        assert world == row['before'] and digest(actor.record()) == row['before_owner']
        context = game.context(world, actor.state)
        result = DecisionLoop.decide_batch([(actor, game.request(context))], False,
            _counterfactual_roots=(row['action'],))[0]
        # The evaluator's law validates the historical draws only. It is never
        # passed to the actor or its prospective controller.
        truth = law(world['tick'], spec['turns'], spec['regime'])
        assert truth == row['truth'] and sample(truth, rng.random()) == row['response']
        after, flow = game.transition(world, row['action'], row['response'])
        resolve_feedback(actor, result['ticket'], game.feedback(world, after, row['action'], context, flow))
        assert after == row['after'] and digest(actor.record()) == row['owner']
        assert character_key(game.context(after)) == persona
        world = after
    original = saved['trace'][args.tick]
    owner = digest(actor.record())
    assert world == original['before'] and owner == original['before_owner']
    result, captured = decide(actor, game, world, arm, seeds=seeds)
    assert character_key(result['context']) == persona
    forecast = captured['forecast']
    control = list(seeds) == cfg['seeds']
    if control:
        assert result['decision']['action_id'] == original['action']
        assert result['deliberation']['selected_plan'] == original['selected_plan']
        assert result['deliberation']['selected_purpose'] == original['selected_purpose']
        assert forecast.purpose == original['purpose_by_plan']
    best = {}
    for name, roots in forecast.roots.items():
        best[roots[0]] = max(best.get(roots[0], -float('inf')), forecast.purpose[name])
    row = dict(id=spec['id'], tick=args.tick, arm=arm, seeds=list(seeds),
        input_sha256=args.input_sha256, registration=registration, runtime=runtime,
        factual_owner=owner, persona=persona, exact_factual_prefix_steps=args.tick,
        exact_registered_forecast_reproduced=control,
        action=result['decision']['action_id'], selected_purpose=result['deliberation']['selected_purpose'],
        estimated_purpose_by_root=best, purpose_by_plan=forecast.purpose, roots=forecast.roots,
        decision=result['decision'], deliberation=result['deliberation'],
        # Retain the actual numeric selection packet. Purpose alone cannot
        # distinguish a personal tradeoff from a bad forecast. This is captured
        # after the original decision and never fed back into its computation.
        selection_inputs=dict(base_context=result['context'], forecast_contexts=forecast.contexts,
            target=forecast.target,
            extra_selectors={name: getattr(forecast, name) is not None for name in
                ('continuity', 'stability_reference', 'replan_check', 'purpose_sampling', 'goal_ties')},
            outcome_limit=forecast.outcome_limit, progress=result.get('progress'),
            policy=dict(principle_priority=actor.policy.principle_priority,
                        commitments=actor.policy.commitments, residual=actor.policy.residual.tolist())),
        simulated_steps=captured['simulated_steps'], total_seconds=time.monotonic()-started,
        diagnostic_source_sha256=sha(Path(__file__).read_bytes()), adoption=False)
    row['personal_ranking'] = personal_ranking(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(encoded(row))
    print(json.dumps({k: row[k] for k in ('id','tick','arm','seeds','action',
        'exact_registered_forecast_reproduced','total_seconds')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--input-sha256', required=True)
    parser.add_argument('--tick', type=int, required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--samples', type=int, required=True)
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
