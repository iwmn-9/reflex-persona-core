"""Offline C2 root diagnosis; incompatible roots cannot certify a numeric gate."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile


def aggregate(rows, plan):
    expected = {(s['id'], arm) for s in plan['specs'] for arm in plan['arms']}
    actual = [(r['spec']['id'], r['arm']) for r in rows]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError('every registered root required exactly once')
    specs = {s['id']: s for s in plan['specs']}
    for row in rows:
        r = row['result']
        if row['spec'] != specs[row['spec']['id']] or r['complete'] is not True:
            raise ValueError('registered specification and settled result required')
        value = r['normalized_regret']
        if type(r['reference_admissible']) is not bool:
            raise ValueError('explicit reference admissibility required')
        if r['reference_admissible']:
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1.:
                raise ValueError('finite normalized regret required for comparable roots')
        elif value is not None:
            raise ValueError('incompatible roots must not carry a numeric regret')

    def group(selected):
        values = sorted(r['result']['normalized_regret'] for r in selected
                        if r['result']['reference_admissible'])
        incomparable = [r['spec']['id'] for r in selected
                        if not r['result']['reference_admissible']]
        stats = dict(mean=math.fsum(values) / len(values),
                     p90=values[math.ceil(.9 * len(values)) - 1], max=max(values)) if values else None
        full = stats if len(values) == len(selected) and selected else None
        passed = (full['mean'] <= .05 and full['p90'] <= .15 and full['max'] <= .5) if full else None
        return dict(registered_roots=len(selected), comparable=len(values),
                    incompatible_roots=incomparable, full_group=full,
                    comparable_subset=stats, numeric_gate=passed,
                    status='unmeasured' if full is None else 'pass' if passed else 'fail')

    groups = {}
    for arm in plan['arms']:
        selected = [r for r in rows if r['arm'] == arm]
        groups[arm] = dict(all=group(selected),
            by_kind={k: group([r for r in selected if r['spec']['kind'] == k])
                     for k in sorted({s['kind'] for s in plan['specs']})},
            by_persona={str(p): group([r for r in selected if r['spec']['profile'] == p])
                        for p in sorted({s['profile'] for s in plan['specs']})})
    return dict(complete=True, registered_roots=len(expected), groups=groups,
                adoption=False,
                overall_C2='unmeasured: known first-action diagnosis; long-world and transfer gates separate')


def review(archive, source, output):
    with zipfile.ZipFile(archive) as z, zipfile.ZipFile(source) as src:
        if json.loads(z.read('done.json'))['complete'] is not True:
            raise ValueError('completed reference execution required')
        plan_bytes = z.read('cohort/plan.json'); plan = json.loads(plan_bytes)
        for name, digest in plan['sources'].items():
            if hashlib.sha256(src.read(name)).hexdigest() != digest:
                raise ValueError('frozen source mismatch: ' + name)
        if hashlib.sha256(src.read('docs/COMPLETION.md')).hexdigest() != plan['completion_contract_sha256']:
            raise ValueError('completion contract mismatch')
        old = json.loads(z.read('cohort/summary.json'))
        if old['source_unchanged'] is not True or old['plan_sha256'] != hashlib.sha256(plan_bytes).hexdigest():
            raise ValueError('execution integrity proof missing')
        rows = [json.loads(z.read('cohort/' + s['id'] + '-' + arm + '.json'))
                for s in plan['specs'] for arm in plan['arms']]
    result = aggregate(rows, plan)
    result['provenance'] = {label: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                            for label, path in dict(raw_results=archive, source_bundle=source,
                                                    aggregator=__file__).items()}
    result['aggregation_change'] = 'Incompatible roots retained as unmeasured; kind/persona groups added. No simulation or thresholds changed.'
    Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', required=True); p.add_argument('--source', required=True)
    p.add_argument('--output', required=True); a = p.parse_args()
    review(a.archive, a.source, a.output)
