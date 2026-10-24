"""Compare frozen research execution and an isolated installed candidate.

Run twice in fresh Python -I processes with the same registered plan. Each run
writes complete per-case behavioral hashes and factual trajectories. This is
integration parity, never replacement independent quality evidence.
"""
import argparse
import copy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys


def encoded(value):
    def convert(x):
        if is_dataclass(x): return asdict(x)
        if hasattr(x, 'tolist'): return x.tolist()
        raise TypeError(type(x).__name__)
    return json.dumps(value, default=convert, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def write(path, value):
    temp = path.with_suffix('.new')
    temp.write_bytes(encoded(value)); temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--implementation', choices=('frozen', 'package'), required=True)
    args = parser.parse_args(); root = args.root.resolve()
    plan_bytes = (root/'plan.json').read_bytes(); plan = json.loads(plan_bytes)
    library = Path(plan['libraries'][args.implementation]['path'])
    def verify_source():
        for name, expected in plan['libraries'][args.implementation]['files'].items():
            if sha((library/name).read_bytes()) != expected:
                raise ValueError('changed library source: '+name)
        if sha(Path(__file__).read_bytes()) != plan['worker_sha256']:
            raise ValueError('changed verifier')
    verify_source()
    archive = Path(plan['known_worlds']['path'])
    if sha(archive.read_bytes()) != plan['known_worlds']['sha256']:
        raise ValueError('changed known-world history')
    sys.path.insert(0, str(library))
    if args.implementation == 'package':
        sys.path.append(plan['libraries']['frozen']['path'])  # Host adapters only.
    import numpy as np
    import reflex
    if not Path(reflex.__file__).resolve().is_relative_to(library):
        raise ValueError('wrong library imported')
    from reflex.core import digest, character_key
    from reflex.decision_loop import DecisionLoop
    from reflex.model_observer import resolve_feedback
    from tools.continuing_games import ContinuingGame
    from tools.run_continuation_portfolio import game_for
    from tools.run_owned_value import cumulative_basis, law, sample
    from reflex.laboratory import PROFILES
    if args.implementation == 'package':
        from reflex.fixed_horizon import FixedHorizonOwner
        class BlockResearch:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'tools' or fullname.startswith('tools.'):
                    raise ImportError('candidate may not import research tools')
        sys.meta_path.insert(0, BlockResearch())
    else:
        from tools.horizon_consensus import consensus_planner
    known = [json.loads(line) for line in gzip.decompress(archive.read_bytes()).splitlines()]
    histories = {row['spec']['id']: row for row in known if row.get('retry') is True}
    output = root/args.implementation; output.mkdir(exist_ok=True)
    contract = dict(plan_sha256=sha(plan_bytes), implementation=args.implementation,
                    worker_sha256=plan['worker_sha256'])
    for case in plan['cases']:
        path = output/(case['id']+'.json')
        if path.exists():
            saved = json.loads(path.read_bytes())
            if saved['contract'] != contract or not saved['complete']:
                raise ValueError('invalid prior parity result')
            continue
        write(output/'progress.json', dict(case=case['id'], phase='replay', at=datetime.now(timezone.utc).isoformat()))
        if case['kind'] == 'known-root':
            original = histories[case['history']]
            spec = original['spec']; game = game_for(spec)
        else:
            spec = case; game = ContinuingGame(case['game'], PROFILES[case['profile']], case['turns'], case['seed'] % 2)
        world = game.start(); actor = game.actor(game.context(world), learning='segmented')
        rng = np.random.default_rng(spec['seed']); replayed = 0
        if case['kind'] == 'known-root':
            for entry in original['result']['trace'][:case['tick']]:
                assert world == entry['before'] and digest(actor.record()) == entry['before_owner']
                context = game.context(world, actor.state); request = game.request(context)
                result = DecisionLoop.decide_batch([(actor, request)], False, _counterfactual_roots=(entry['action'],))[0]
                response = sample(law(world['tick'], spec['turns'], spec['regime']), rng.random())
                assert response == entry['response']
                after, flow = game.transition(world, entry['action'], response)
                resolve_feedback(actor, result['ticket'], game.feedback(world, after, entry['action'], context, flow))
                assert after == entry['after'] and digest(actor.record()) == entry['owner']
                world = after; replayed += 1
        settings = dict(observe=game.context, events=game.events, terminal=game.terminal,
            assess=lambda w: game.goal(w).record(), request_for=game.request, feedback=game.feedback,
            model_id=game.model_id, target=game.target, cumulative_preference=cumulative_basis(game, True),
            horizon=32, seeds=(701,702,703), recovery_cache=64)
        if args.implementation == 'package':
            controller = FixedHorizonOwner(actor, **settings)
        trace = []; axes = character_key(game.context(world))
        for step in range(case['steps']):
            if game.terminal(world): break
            write(output/'progress.json', dict(case=case['id'], phase='decide', step=step,
                world_tick=world['tick'], replayed=replayed, at=datetime.now(timezone.utc).isoformat()))
            context = game.context(world, actor.state); request = game.request(context)
            before = digest(actor.record())
            if args.implementation == 'package':
                result = controller.decide(world); cap = controller.last
            else:
                cap = {}
                planner = consensus_planner(actor, request, world, captured=cap, minimum_horizon=32,
                    include_retry=True, coherent_cumulative_mean=True, purpose_estimator='conditional',
                    tail_recovery=2, **settings)
                result = DecisionLoop.decide_batch([(actor, request)], False, planner)[0]
            assert character_key(result['context']) == axes
            key = result['decision']['action_id']
            response = sample(law(world['tick'], spec['turns'], spec['regime']), rng.random())
            after, flow = game.transition(world, key, response)
            row = dict(tick=world['tick'], action=key, before_owner=before,
                decision_sha256=sha(encoded(result)), forecast_sha256=sha(encoded(cap['forecast'])),
                paths_sha256=sha(encoded(cap['paths'])), modeled_sha256=sha(encoded(cap['modeled'])),
                response=response, purpose=result['deliberation']['selected_purpose'],
                simulated_steps=cap['simulated_steps'], cutoff=cap['horizon_consensus'])
            resolve_feedback(actor, result['ticket'], game.feedback(world, after, key, context, flow))
            row['after_owner'] = digest(actor.record()); row['world_sha256'] = digest(after)
            trace.append(row); world = after
            if args.implementation == 'package':
                bundle = json.loads(encoded(dict(world=world, rng=rng.bit_generator.state,
                    owner=controller.checkpoint(world))))
                template = FixedHorizonOwner(game.actor(game.context(game.start()), learning='segmented'), **settings)
                controller = template.restored(bundle['world'], bundle['owner']); actor = controller.actor
                world = bundle['world']; rng = np.random.default_rng(); rng.bit_generator.state = bundle['rng']
                assert digest(actor.record()) == row['after_owner']
        if len(trace) != case['steps']:
            raise ValueError('parity case ended earlier than registered')
        write(path, dict(complete=True, contract=contract, case=case, trace=trace,
            final=world, rng=rng.bit_generator.state, owner_sha256=digest(actor.record()), replayed=replayed))
        print(json.dumps(dict(implementation=args.implementation, case=case['id'], complete=True)), flush=True)
    verify_source()
    write(output/'progress.json', dict(complete=True, cases=len(plan['cases']), at=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__': main()
