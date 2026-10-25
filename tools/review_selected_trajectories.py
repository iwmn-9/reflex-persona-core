"""Final read-only account of the complete registered long-play comparison.

Consumes the published factual audit, never chooses new seeds or reruns play.
True-law reference values are evaluator-only and omit persona/information
constraints. Conditional increments retain the effects of earlier outcomes;
neither account is a pure intelligence or personality-utility measurement.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    folder = Path(args.evidence).resolve()
    read = lambda name: json.loads((folder / name).read_bytes())
    proof = read('verification.json')
    assert proof['complete'] and proof['source_verified'] and proof['actual_owner_replayed']
    for name, expected in proof['artifacts'].items():
        assert sha((folder / name).read_bytes()) == expected, name
    plan, summary, quality = (read(name) for name in ('plan.json', 'summary.json', 'quality_review.json'))
    assert summary['complete'] and quality['complete']
    assert len(plan['specs']) == summary['completed_pairs'] == summary['registered_pairs'] == proof['pairs'] == quality['pairs'] == 40
    assert quality['source'] == plan['source']
    assert quality['summary_sha256'] == sha((folder / 'summary.json').read_bytes())
    runs = [json.loads(line) for line in gzip.decompress((folder / 'runs.jsonl.gz').read_bytes()).splitlines()]
    assert len(runs) == 80 and all('reference' not in item for item in runs)
    indexed = {(item['spec']['id'], item['retry']): item['result'] for item in runs}
    assert len(indexed) == 80
    rows = {row['id']: row for row in summary['rows']}
    assert set(rows) == {spec['id'] for spec in plan['specs']}
    receipts = {row['id']: row for row in quality['results']}
    manifest = read('source_snapshot.json')
    assert sha(encoded(manifest)) == plan['source']
    with tempfile.TemporaryDirectory(prefix='selected-world-account-') as temporary:
        source = Path(temporary)
        with zipfile.ZipFile(folder / 'source.zip') as archive:
            for name in archive.namelist():
                assert (source / name).resolve().is_relative_to(source)
            archive.extractall(source)
        for name, expected in manifest.items():
            assert sha((source / name).read_bytes()) == expected, name
        sys.path.insert(0, str(source))
        from reflex.laboratory import PROFILES
        from reflex.core import character_key
        from tools.run_continuation_portfolio import game_for
        from tools.summarize_horizon_study import realized_account
        pairs = []
        for spec in plan['specs']:
            identifier = spec['id']
            arms = []
            for retry in (False, True):
                result = indexed[identifier, retry]
                assert result['configuration']['spec'] == spec and result['configuration']['retry'] == retry
                assert result['configuration']['horizon'] == (32 if retry else 16)
                assert result['configuration']['seeds'] == [701, 702, 703]
                if retry:
                    assert all(len(step['horizon_consensus']['ballots']) == 1 for step in result['trace'])
                else:
                    assert result['configuration']['reference'] == 'adopted-owner-continuity-H16'
                assert result['complete'] and len(result['trace']) == spec['turns'] == 128
                assert sha(encoded(result)) == receipts[identifier]['results'][int(retry)]['sha256']
                assert result['checks']['fixed_axes'] == result['checks']['owner_parity'] == 128
                game = game_for(spec)
                account = realized_account(dict(game=spec['game'], profile=PROFILES[spec['profile']]['id'],
                    turns=spec['turns'], seed=spec['seed'], regime=spec['regime'],
                    trace=result['trace'], score=result['score']))
                arms.append(dict(score=result['score'], account=account,
                    actions=dict(Counter(step['action'] for step in result['trace'])),
                    policy_choices=dict(Counter(step['selected_policy'] for step in result['trace'])),
                    elapsed_seconds=result['seconds'],
                    simulated_steps=(sum(step['simulated_steps'] for step in result['trace'])
                        if all(step.get('simulated_steps') is not None for step in result['trace']) else None),
                    simulated_steps_reported_decisions=sum(step.get('simulated_steps') is not None for step in result['trace']),
                    fixed_persona=character_key(game.context(game.start())),
                    saved_restores=result['checks']['restored'],
                    final=result['final']))
            left, right = (indexed[identifier, retry] for retry in (False, True))
            assert left['runtime'] == right['runtime']
            assert arms[0]['fixed_persona'] == arms[1]['fixed_persona']
            assert all(a['truth'] == b['truth'] and a['response'] == b['response']
                       for a, b in zip(left['trace'], right['trace']))
            delta = right['score'] - left['score']
            conditional = sum(s['conditional_increment'] for s in right['trace']) - sum(s['conditional_increment'] for s in left['trace'])
            assert delta == rows[identifier]['delta'] and conditional == rows[identifier]['conditional_delta']
            difference = dict(score=delta, conditional=conditional, residual=delta-conditional)
            if 'bellman' in arms[0]['account']:
                a, b = (arm['account']['bellman'] for arm in arms)
                difference.update(reference_choice=b['residual_removed']-a['residual_removed'],
                                  reference_residual=b['residual']-a['residual'])
                assert abs(sum(difference[k] for k in ('reference_choice', 'reference_residual'))-delta) < 1e-10
            phases = []
            game = game_for(spec)
            for start in (0, 32, 64, 96):
                parts = [result['trace'][start:start+32] for result in (left, right)]
                phases.append(dict(start=start, end=start+32,
                    final_delta=game.goal(parts[1][-1]['after']).value-game.goal(parts[0][-1]['after']).value,
                    conditional_delta=sum(s['conditional_increment'] for s in parts[1])-sum(s['conditional_increment'] for s in parts[0]),
                    actions=[dict(Counter(s['action'] for s in part)) for part in parts]))
            pair = dict(id=identifier, spec=spec, profile=PROFILES[spec['profile']]['id'],
                difference=difference, arms=arms, phases=phases,
                action_disagreements=sum(a['action'] != b['action'] for a, b in zip(left['trace'], right['trace'])))
            pairs.append(pair)
            print(json.dumps(dict(id=identifier, difference=difference)), flush=True)
    by_persona = defaultdict(list)
    by_world = defaultdict(list)
    for pair in pairs:
        by_persona[pair['spec']['game'], pair['profile']].append(pair)
        by_world[pair['spec']['game'], pair['spec']['seed']].append(pair)
    personas = []
    for (game, persona), items in sorted(by_persona.items()):
        personas.append(dict(game=game, persona=persona, conditions=len(items),
            mean_delta=sum(row['difference']['score'] for row in items)/len(items),
            conditional_delta=sum(row['difference']['conditional'] for row in items)/len(items),
            actions=[dict(sum((Counter(row['arms'][side]['actions']) for row in items), Counter())) for side in (0, 1)]))
    worlds = []
    for (game, seed), items in sorted(by_world.items()):
        assert sorted(row['spec']['profile'] for row in items) == [0, 1, 2, 3]
        series = [tuple(step['action'] for step in indexed[row['id'], True]['trace']) for row in items]
        first = [indexed[row['id'], True]['trace'][0] for row in items]
        assert all(entry['before'] == first[0]['before'] for entry in first)
        worlds.append(dict(game=game, seed=seed, distinct_action_sequences=len(set(series)),
            same_initial_world_choices={row['profile']: entry['action'] for row, entry in zip(items, first)}))
    output = dict(complete=True, adopted=False, pairs=pairs, personas=personas, worlds=worlds,
        reviewed_pairs=40, audited_actual_decisions=10240,
        source=plan['source'], plan_sha256=sha((folder/'plan.json').read_bytes()),
        evidence_verification_sha256=sha((folder/'verification.json').read_bytes()),
        quality_review_sha256=sha((folder/'quality_review.json').read_bytes()),
        reviewer_sha256=sha(Path(__file__).read_bytes()),
        inherited_quality_requirements=quality['inherited_quality_requirements'],
        adverse_ids=[row['id'] for row in pairs if any(row['difference'].get(k, 0)<-1e-12
                      for k in ('score', 'conditional', 'reference_choice'))],
        action_disagreements=sum(row['action_disagreements'] for row in pairs),
        limits=__doc__, cost='Recorded elapsed seconds include contention and resume wall time; not CPU time or game-frame latency. Personas sharing a world seed are not independent world samples.')
    Path(args.output).write_bytes(encoded(output))
    print(json.dumps(dict(complete=True, pairs=40, adverse_ids=output['adverse_ids'],
        action_disagreements=output['action_disagreements'], adoption=False)), flush=True)


if __name__ == '__main__':
    main()
