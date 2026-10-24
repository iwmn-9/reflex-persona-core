"""Offline depth sensitivity of existing exact references at all five saved disagreements.

The reference ignores personality and future belief learning. It is not an NPC
input, a personality-loss estimate, or a replacement for the registered worlds.
"""
import hashlib
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

base = Path.cwd()
root = base / 'results/continuation_cutoff_tradeoffs'
target = base / 'results/cutoff_reference_depths'
target.mkdir(exist_ok=True)
sha = lambda b: hashlib.sha256(b).hexdigest()
manifest = json.loads((root / 'source_snapshot.json').read_bytes())
for name, expected in manifest.items():
    assert sha((root / '_source' / name).read_bytes()) == expected
sys.path.insert(0, str(root / '_source'))
from reflex.core import digest
from reflex.decision_loop import DecisionLoop
from reflex.model_observer import resolve_feedback
from tools.owned_checkpoint import encoded
from tools.run_continuation_portfolio import game_for
from tools.run_owned_value import law
from tools.diagnose_auction_continuity import reference as auction_reference
from tools.combat_reference import reference as combat_reference
from tools.resource_reference import reference as resource_reference

plan = json.loads((root / 'plan.json').read_bytes())
assert sha(encoded(manifest)) == plan['source']
old_oracles = {r['case']:r for r in json.loads((base / 'evidence/long_horizon/continuation_cutoff_tradeoffs/oracle_reference.json').read_bytes())['rows']}
rows = []
bellman_checks = 0
for case in plan['cases']:
    data = (root / case['input']).read_bytes()
    assert sha(data) == case['sha256']
    envelope = json.loads(data); saved = envelope['payload']
    assert sha(encoded(saved)) == envelope['sha256']
    spec = saved['contract']['configuration']['spec']; game = game_for(spec)
    actor = game.actor(game.context(game.start()), learning='segmented')
    for entry in saved['trace'][:case['tick']]:
        assert digest(actor.record()) == entry['before_owner']
        context = game.context(entry['before'], actor.state)
        decision = DecisionLoop.decide_batch([(actor, game.request(context))], False,
            _counterfactual_roots=(entry['action'],))[0]
        after, flow = game.transition(entry['before'], entry['action'], entry['response'])
        resolve_feedback(actor, decision['ticket'], game.feedback(entry['before'], after, entry['action'], context, flow))
        assert after == entry['after'] and digest(actor.record()) == entry['owner']
    entry = saved['trace'][case['tick']]; w = entry['before']; owner = digest(actor.record())
    assert owner == entry['before_owner'] == case['actual_fixed32_owner']
    probability = actor.predictor.predict(game.context(w, actor.state))
    if case['id'] in old_oracles:
        assert probability == old_oracles[case['id']]['root_prediction']
    first = json.loads((root / (case['id'] + '.json')).read_bytes())
    modeled = {r['horizon']: r for r in first['choices']}
    fixed = first['fixed32_selected']['action']; consensus = first['consensus_selected']['action']
    depth_rows = []
    for depth in sorted({*range(16, 33), spec['turns'] - w['tick']}):
        end = min(spec['turns'], w['tick'] + depth); scale = end / spec['turns']
        sources = {}
        for label in ('true_law', 'stationary_root_belief'):
            probabilities = (lambda t: tuple(law(t, spec['turns'], spec['regime']).values())) if label == 'true_law' else (lambda t: tuple(probability.values()))
            if spec['game'] == 'auction':
                options = auction_reference(end, spec['seed'] % 2, probabilities)
                def choices(state):
                    offset = (state['score'] - state['rival']) / (8 * spec['turns'])
                    return {k: offset + v * scale for k,v in options(state['tick'], int(state['cash'])).items()}
                def value(state):
                    return game.goal(state).value if state['tick'] == end else max(choices(state).values())
            else:
                factory = combat_reference if spec['game'] == 'combat' else resource_reference
                raw_value, raw_choices = factory(end, probabilities)
                value = lambda state: raw_value(state) * scale
                choices = lambda state: {k: v * scale for k,v in raw_choices(state).items()}
            q = choices(w)
            assert set(q) == set(game.keys(w))
            for action in game.keys(w):
                expected = sum(p * value(game.transition(w, action, str(z))[0])
                    for z,p in enumerate(probabilities(w['tick'])))
                assert abs(q[action] - expected) < 1e-10
                bellman_checks += 1
            best = max(q.values())
            sources[label] = dict(q=q, best_actions=[k for k,v in q.items() if best-v < 1e-12],
                fixed32_gap=best-q[fixed], consensus_gap=best-q[consensus],
                fixed32_advantage=q[fixed]-q[consensus])
        depth_rows.append(dict(depth=depth, end=end, references=sources,
            modeled_choice=modeled.get(depth, {}).get('action'),
            modeled_best_by_root=modeled.get(depth, {}).get('best_by_root')))
    assert digest(actor.record()) == owner
    rows.append(dict(id=case['id'], tick=case['tick'], owner=owner,
        input_sha256=case['sha256'], root_prediction=probability,
        fixed32=fixed, consensus=consensus, depths=depth_rows))
    print(json.dumps(dict(id=case['id'], complete=True, depths=len(depth_rows))), flush=True)
out = dict(complete=True, adoption=False, checked_at=datetime.now(timezone.utc).isoformat(),
    source=plan['source'], input_plan_sha256=sha((root/'plan.json').read_bytes()),
    script_sha256=sha(Path(__file__).read_bytes()), reference_sources={n:manifest[n] for n in (
        'tools/diagnose_auction_continuity.py','tools/combat_reference.py','tools/resource_reference.py',
        'tools/continuing_games.py','tools/run_owned_value.py')},
    actual_past_replayed=sum(c['tick'] for c in plan['cases']), bellman_checks=bellman_checks,
    rows=rows, limit=__doc__, normalization='Every depth is normalized to the original full-world public score, including already accumulated score. Depth is the number of future decisions, not the absolute terminal tick.')
(target / 'result.json').write_bytes(encoded(out))
(target / 'analysis.py').write_bytes(Path(__file__).read_bytes())
