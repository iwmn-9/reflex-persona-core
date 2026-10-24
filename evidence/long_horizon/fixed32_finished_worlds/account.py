"""Account complete audited arms with existing evaluator-only references.

Perfect-information opportunity loss includes personality and unavailable
information, not just mistakes. These values never enter the playing policy.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser(); parser.add_argument('--id', required=True)
args = parser.parse_args(); root = Path('results/continuation_horizon_ablation')
sha = lambda b: hashlib.sha256(b).hexdigest()
manifest = json.loads((root / 'source_snapshot.json').read_bytes())
for name, h in manifest.items():
    assert sha((root / '_source' / name).read_bytes()) == h
sys.path.insert(0, str((root / '_source').resolve()))
from tools.run_continuation_portfolio import game_for
from tools.run_owned_value import law
from tools.combat_reference import account as combat_account
from tools.diagnose_auction_continuity import reference as auction_reference


def auction_account(row, spec):
    game = game_for(spec); turns = spec['turns']
    q = auction_reference(turns, spec['seed'] % 2,
        lambda t: tuple(law(t, turns, spec['regime']).values()))
    def choices(w):
        offset = (w['score'] - w['rival']) / (8 * turns)
        return {k: v + offset for k, v in q(w['tick'], int(w['cash'])).items()}
    def value(w):
        return game.goal(w).value if game.terminal(w) else max(choices(w).values())
    initial = value(row['trace'][0]['before']); loss = residual = 0.; steps = []; checks = 0
    for t in row['trace']:
        values = choices(t['before']); truth = law(t['tick'], turns, spec['regime'])
        assert set(values) == set(game.keys(t['before']))
        for key, expected in values.items():
            actual = sum(p * value(game.transition(t['before'], key, z)[0]) for z, p in truth.items())
            assert abs(actual - expected) < 1e-10
            checks += 1
        regret = max(values.values()) - values[t['action']]
        noise = value(t['after']) - values[t['action']]
        loss += regret; residual += noise
        steps.append(dict(tick=t['tick'], purpose_opportunity_loss=regret, residual=noise))
    error = initial - loss + residual - row['score']
    assert abs(error) < 1e-10
    return dict(initial_perfect_information=initial, purpose_opportunity_loss=loss,
        residual=residual, residual_removed=initial-loss, reconstruction_error=error,
        bellman_checks=checks, increments=steps)


case = args.id; out = Path('evidence/long_horizon/fixed32_finished_worlds') / case
proof = json.loads((out / 'verification.json').read_bytes()); assert proof['arm_audit_complete']
paths = dict(h16=root / 'references' / (case + '.json'), fixed32=root / case / 'candidate/result.json',
    consensus=Path('results/continuation_horizon_consensus/worlds') / case / 'candidate/result.json')
rows = {}
for side, path in paths.items():
    data = path.read_bytes(); row = json.loads(data); spec = row['configuration']['spec']
    assert row['complete'] and game_for(spec).terminal(row['final'])
    if spec['game'] == 'combat':
        reference_file = 'tools/combat_reference.py'
        account = combat_account(dict(game=spec['game'], turns=spec['turns'], regime=spec['regime'],
            trace=row['trace'], score=row['score']))
    elif spec['game'] == 'auction':
        reference_file = 'tools/diagnose_auction_continuity.py'; account = auction_account(row, spec)
    else:
        raise ValueError('This offline reviewer supports the existing auction/combat references only')
    rows[side] = dict(input_sha256=sha(data), score=row['score'], account=account)
for side, key in [('h16','control_result_sha256'),('fixed32','result_sha256'),('consensus','consensus_result_sha256')]:
    assert rows[side]['input_sha256'] == proof[key]
comparisons = []
for other in ['h16', 'consensus']:
    a = rows[other]; b = rows['fixed32']
    choice = b['account']['residual_removed'] - a['account']['residual_removed']
    noise = b['account']['residual'] - a['account']['residual']
    delta = b['score'] - a['score']; assert abs(delta - choice - noise) < 1e-10
    comparisons.append(dict(reference=other, score_delta=delta,
        reference_choice_contribution=choice, realized_residual_contribution=noise))
result = dict(complete=True, adoption=False, source=proof['source'], case=case,
    reference_source_sha256=sha((root / '_source' / reference_file).read_bytes()),
    accounting_script_sha256=sha(Path(__file__).read_bytes()), rows=rows, comparisons=comparisons,
    limit=__doc__)
(out / 'account.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(out.parent / 'account.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(comparisons, indent=2))
