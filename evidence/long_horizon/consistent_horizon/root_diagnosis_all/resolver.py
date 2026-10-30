"""Resolve the already registered loss-root protocol from a complete archive.

No policy execution, source mutation or adoption. Prior probes are reusable
only when their original input, factual owner and frozen registration match.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare(archive, previous):
    archive = Path(archive)
    proof = json.loads((archive / 'verification.json').read_bytes())
    if not proof['complete']:
        raise ValueError('complete audited study required')
    data = {}
    for name in ('plan.json', 'outcome_review.json', 'runs.jsonl.gz', 'checkpoints.zip'):
        raw = (archive / name).read_bytes()
        if sha(raw) != proof['artifacts'][name]:
            raise ValueError('audited input changed: ' + name)
        data[name] = raw
    plan = json.loads(data['plan.json'])
    review = json.loads(data['outcome_review.json'])
    if not review['complete']:
        raise ValueError('complete outcome review required')
    rows = [json.loads(line) for line in gzip.decompress(data['runs.jsonl.gz']).splitlines()]
    records = {(row['spec']['id'], row['arm']): row for row in rows}
    expected = {(spec['id'], arm) for spec in plan['specs'] for arm in plan['arms']}
    if len(records) != len(rows) or set(records) != expected:
        raise ValueError('all registered executions exactly once required')
    outcomes = {row['id']: row for row in review['rows']}
    if len(outcomes) != len(review['rows']) or set(outcomes) != {s['id'] for s in plan['specs']}:
        raise ValueError('all registered outcomes exactly once required')
    if (proof['pairs'] != len(plan['specs']) or proof['executions'] != len(rows)
            or proof['actual_steps'] != sum(len(row['result']['trace']) for row in rows)):
        raise ValueError('completion counts differ')
    ref, candidate = plan['reference_arm'], plan['candidate_arm']
    previous = Path(previous) if previous else None
    prior = []
    prior_inputs = None
    if previous:
        p = json.loads((previous / 'verification.json').read_bytes())
        registration = json.loads((previous / 'registration.json').read_bytes())
        if registration['worker_ref'] != proof['source_ref']:
            raise ValueError('previous probes used another frozen worker')
        raw = (previous / 'rows.json.gz').read_bytes()
        if not p['complete'] or sha(raw) != p['rows_sha256']:
            raise ValueError('previous root diagnosis corrupted or incomplete')
        prior = json.loads(gzip.decompress(raw))
        if len(prior) != p['evaluations']:
            raise ValueError('previous evaluation count changed')
        if any(r['diagnostic_source_sha256'] != registration['sources']['tools/probe_frozen_sample_budget.py'] for r in prior):
            raise ValueError('previous diagnostic implementation differs')
        keys = [(r['id'], r['tick'], tuple(r['seeds'])) for r in prior]
        if len(set(keys)) != len(keys):
            raise ValueError('duplicate previous root evaluation')
        prior_inputs = dict(path=str(previous), rows_sha256=sha(raw),
            registration_sha256=sha((previous / 'registration.json').read_bytes()))
    losses = []
    jobs = []
    with zipfile.ZipFile(io.BytesIO(data['checkpoints.zip'])) as checkpoints:
        for spec in plan['specs']:
            cid = spec['id']
            pair = {}
            for arm in plan['arms']:
                row = records[cid, arm]
                result = row['result']
                if (row['spec'] != spec or not result['complete']
                        or result['replayed_actual_steps'] != spec['turns']
                        or result['configuration']['spec'] != spec
                        or result['configuration']['arm'] != arm
                        or result['configuration']['registration'] != proof['registration']
                        or [t['tick'] for t in result['trace']] != list(range(spec['turns']))):
                    raise ValueError('complete registered history required')
                raw = checkpoints.read(f'{cid}/{arm}/result.json')
                if (json.loads(raw) != result
                        or sha(raw) != review['source_results'][f'adverse-{cid}-{arm}']):
                    raise ValueError('result and reviewed history differ')
                pair[arm] = (result, sha(raw))
            d = outcomes[cid]['differences']
            negative = {k: d[k] for k in ('score', 'conditional', 'reference_choice')
                        if k in d and d[k] < -1e-12}
            if not negative:
                continue
            left, right = pair[ref][0]['trace'], pair[candidate][0]['trace']
            divergence = next((a['tick'] for a, b in zip(left, right) if a['action'] != b['action']), None)
            phase = min(outcomes[cid]['phases'], key=lambda p: (p['differences']['score_gain'], p['start']))
            roots = {phase['start']: ['worst_32_turn_realized_gain_phase']}
            if divergence is not None:
                roots.setdefault(divergence, []).append('first_action_divergence')
            loss = dict(id=cid, losses=negative, roots=[])
            for tick, reasons in sorted(roots.items()):
                saved = right[tick]
                loss['roots'].append(dict(tick=tick, reasons=reasons, factual_owner=saved['before_owner']))
                for start in (701, 9101, 9201):
                    for samples in (3, 8):
                        seeds = list(range(start, start + samples))
                        matches = [r for r in prior if (r['id'], r['tick'], r['seeds']) == (cid, tick, seeds)]
                        if matches:
                            old = matches[0]
                            if (old['input_sha256'] != pair[candidate][1]
                                    or old['factual_owner'] != saved['before_owner']
                                    or old['registration'] != proof['registration']):
                                raise ValueError('previous evaluation has a different factual root')
                            if start == 701 and samples == 8 and (
                                    not old['exact_registered_forecast_reproduced']
                                    or old['action'] != saved['action']
                                    or old['selected_purpose'] != saved['selected_purpose']):
                                raise ValueError('previous exact forecast control differs')
                        jobs.append(dict(id=cid, tick=tick, start=start, samples=samples,
                            factual_owner=saved['before_owner'], input_sha256=pair[candidate][1],
                            reuse=bool(matches), input_archive_member=f'{cid}/{candidate}/result.json'))
            losses.append(loss)
    return dict(complete_archive=True, adoption=False, candidate=candidate,
        source_ref=proof['source_ref'], registration=proof['registration'],
        archive=str(archive.resolve()), inputs={name: sha(raw) for name, raw in data.items()},
        prior_diagnosis=prior_inputs, losses=losses, jobs=jobs,
        required_evaluations=len(jobs), reused_evaluations=sum(j['reuse'] for j in jobs),
        pending_evaluations=sum(not j['reuse'] for j in jobs),
        resolver_sha256=sha(Path(__file__).read_bytes()),
        scope='Predeclared diagnostic roots only; no new actual worlds, seed replacement or adoption.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--previous')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = prepare(args.archive, args.previous)
    with Path(args.output).open('x', encoding='utf8') as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
    print(json.dumps({k: result[k] for k in ('required_evaluations', 'reused_evaluations', 'pending_evaluations')}))
