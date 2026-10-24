"""Review every registered hypothetical seed bank without selecting a winner.

These deliberately selected disagreement states are exploratory diagnoses, not
independent games. Stability across three banks is not calibrated reliability.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def review(root, references):
    plan = json.loads((root / 'plan.json').read_bytes())
    result = json.loads((root / 'summary.json').read_bytes())
    links = json.loads((references / 'input_references.json').read_bytes())
    source = references / links['reference_choices']['path']
    assert sha(source.read_bytes()) == links['reference_choices']['sha256']
    original = json.loads(source.read_bytes())
    assert result['complete'] and original['complete']
    assert result['source'] == original['source'] == plan['source']
    assert result['plan_sha256'] == sha((root / 'plan.json').read_bytes())
    assert (root / 'plan.json').read_bytes() == (references / 'registration.json').read_bytes()
    cases = {case['id']: case for case in plan['cases']}
    banks = {bank['id']: bank['seeds'] for bank in plan['banks']}
    expected = {(case, bank) for case in cases for bank in banks}
    observed = {(row['case']['id'], row['bank']) for row in result['rows']}
    assert expected == observed and len(result['rows']) == len(expected)
    prior = {row['case']['id']: row for row in original['rows']}
    assert set(prior) == set(cases) and len(original['rows']) == len(cases)
    indexed = {}
    for row in result['rows']:
        key = (row['case']['id'], row['bank']); case = cases[key[0]]
        assert row['case'] == case and row['seeds'] == banks[key[1]]
        assert row['source_owner'] == case['actual_fixed32_owner']
        assert row['factual_owner_unchanged'] and row['factual_world_unchanged']
        assert row == json.loads((root / (key[0] + '-' + key[1] + '.json')).read_bytes())
        indexed[key] = row
    rows = []
    for case_id, case in cases.items():
        first = prior[case_id]
        assert first['case'] == case and first['full_saved_choices_reproduced']
        assert first['factual_owner_unchanged'] and first['source_owner'] == case['actual_fixed32_owner']
        samples = [dict(bank='original', seeds=plan['original_model_seeds'],
            ballots=first['choices'], fixed32_selected=first['fixed32_selected'],
            consensus_selected=first['consensus_selected'], votes=first['votes'])]
        samples.extend(indexed[case_id, bank] for bank in banks)
        observations = []
        for sample in samples:
            ballots = sample['ballots']
            assert [b['horizon'] for b in ballots] == list(range(16, 33))
            assert sample['fixed32_selected']['action'] == ballots[-1]['action']
            votes = Counter(b['action'] for b in ballots)
            assert dict(votes) == sample['votes']
            chosen = next(b for b in reversed(ballots) if votes[b['action']] == max(votes.values()))
            assert all(sample['consensus_selected'][k] == chosen[k] for k in ('action', 'horizon', 'purpose'))
            observations.append(dict(bank=sample['bank'], seeds=sample['seeds'],
                fixed32=ballots[-1]['action'], consensus=chosen['action'],
                cutoff_switches=sum(a['action'] != b['action'] for a, b in zip(ballots, ballots[1:])),
                votes=dict(votes), ballots=[dict(horizon=b['horizon'], action=b['action']) for b in ballots]))
        rows.append(dict(case=case_id, tick=case['tick'], source_owner=case['actual_fixed32_owner'],
            samples=observations, fixed32_varies=len({s['fixed32'] for s in observations}) > 1,
            consensus_varies=len({s['consensus'] for s in observations}) > 1))
    return dict(complete=True, adoption=False, source=plan['source'],
        registered_probes=len(expected), reviewed_cutoff_choices=17 * (len(expected) + len(cases)),
        fixed32_varying_states=sum(r['fixed32_varies'] for r in rows),
        consensus_varying_states=sum(r['consensus_varies'] for r in rows), rows=rows,
        inputs=dict(summary_sha256=sha((root / 'summary.json').read_bytes()),
            original_sha256=sha(source.read_bytes()), plan_sha256=sha((root / 'plan.json').read_bytes())),
        limit=__doc__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--references', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = review(Path(args.root), Path(args.references))
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
