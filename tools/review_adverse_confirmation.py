"""Unchanged reserved forty-world gates against both adopted controls.

Numeric passage does not adopt a candidate: individual losses, personality,
source/history audits and original adverse evidence still require review.
"""
from collections import defaultdict
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.review_adverse_recovery import summarize
from tools.owned_checkpoint import encoded


REGIMES = {'auction': ('outside','shift-return'),
           'combat': ('outside','shift-return'),
           'gambling': ('outside','shift-return','irregular','steady'),
           'resources': ('outside','shift-return')}
WORLD_SEEDS = dict(zip(('outside','shift-return','irregular','steady'),
                      (2026102701,2026102702,2026102703,2026102704)))


def reserved_specs():
    return [dict(id=f'{game}-{profile}-{WORLD_SEEDS[regime]}-{regime}',
                 mechanism='world', game=game, profile=profile, seed=WORLD_SEEDS[regime],
                 regime=regime, turns=128)
            for game, regimes in REGIMES.items() for profile in range(4) for regime in regimes]


def numeric_gates(plan, comparisons):
    assert plan['specs'] == reserved_specs(), 'reserved conditions, seeds and ordering must be retained'
    assert set(comparisons) == {'original16','stable'}
    expected_ids = {s['id'] for s in plan['specs']}
    checks = {}
    for reference, summary in comparisons.items():
        complete = (summary['complete'] and summary['pairs_complete'] == 40
                    and len(summary['pairs']) == 40
                    and {p['id'] for p in summary['pairs']} == expected_ids)
        checks[reference+'_complete'] = complete
        checks[reference+'_family_nonnegative'] = (
            set(summary['families']) == set(REGIMES)
            and all(value >= -1e-12 for family in summary['families'].values() for value in family.values()))
    stable = comparisons['stable']
    checks['strict_gain_vs_stable'] = any(p['score_delta'] > 1e-12 and p['conditional_delta'] > 1e-12
                                          for p in stable['pairs'])
    original = comparisons['original16']
    for game in ('gambling','resources'):
        checks[game+'_both_means_positive_vs_original16'] = (
            game in original['families']
            and all(v > 1e-12 for v in original['families'][game].values()))
    groups = defaultdict(list)
    lookup = {s['id']: s for s in plan['specs']}
    for pair in original['pairs']:
        spec = lookup[pair['id']]
        if spec['game'] == 'gambling':
            groups[spec['regime']].append(pair)
    worlds = {regime: dict(profiles=sorted(p['profile'] for p in rows),
                           score_delta=sum(p['score_delta'] for p in rows)/len(rows),
                           conditional_delta=sum(p['conditional_delta'] for p in rows)/len(rows))
              for regime, rows in groups.items()}
    checks['all_four_gambling_worlds'] = (set(worlds) == set(REGIMES['gambling'])
                                        and all(w['profiles'] == [0,1,2,3] for w in worlds.values()))
    checks['three_gambling_worlds_improve_vs_original16'] = sum(w['score_delta'] > 1e-12 for w in worlds.values()) >= 3
    return dict(checks=checks, numeric_gate_passed=all(checks.values()),
                gambling_worlds=worlds, adoption=False,
                remaining_review='source/history integrity, every individual loss, persona differences, original failed studies')


def review(root, plan):
    comparisons = {}
    for reference in ('original16','stable'):
        compared = dict(plan, reference_arm=reference, concordant_adverse=[])
        comparisons[reference] = summarize(root, compared)
    return dict(comparisons=comparisons, **numeric_gates(plan, comparisons))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True); p.add_argument('--root', required=True)
    p.add_argument('--output', required=True); a = p.parse_args()
    Path(a.output).write_bytes(encoded(review(a.root, json.loads(Path(a.plan).read_bytes()))))
