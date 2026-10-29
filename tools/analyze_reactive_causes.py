"""Evaluator-only attribution of completed immediate-policy trajectories.

The actor is replayed from factual experience. At every paired action
disagreement, detached copies receive the other predictor or the true current
response law. Nothing from these counterfactuals reaches factual learning.
This diagnoses immediate choice, not long-run optimality or justified motives.
"""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import gzip
import json
from pathlib import Path
import sys
from collections import Counter
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reflex.core import character_key, compile_batch, digest
from reflex.decision_loop import DecisionLoop, Reading
from reflex.model_observer import resolve_feedback
from tools.run_adverse_recovery import actor_for, game_for
from tools.owned_checkpoint import encoded, sha


def reader_for(game, probabilities):
    def read(context, cap):
        out = copy.deepcopy(context)
        nodes = len(out['actions']) * len(probabilities)
        assert nodes <= cap
        for action in out['actions']:
            action['outcomes'] = [dict(game.outcome(context, action['id'], z), p=p)
                                  for z, p in probabilities.items()]
        return Reading(out, nodes, 'evaluator-only-response', responses=probabilities,
                       known_conditionals=True)
    return read


def one(path):
    raw = Path(path).read_bytes()
    row = json.loads(gzip.decompress(raw)); spec = row['spec']
    game = game_for(spec); traces = {arm: v['trace'] for arm, v in row['arms'].items()}
    changed = {i for i, (a, b) in enumerate(zip(traces['stable'], traces['transfer']))
               if a['action'] != b['action']}
    out = []
    for arm, trace in traces.items():
        actor = actor_for(game, game.context(game.start()), arm)
        other = 'transfer' if arm == 'stable' else 'stable'
        for t in trace:
            assert digest(actor.record()) == t['before_owner']
            request = game.request(game.context(t['before'], actor.state))
            if t['tick'] in changed:
                fixed = copy.deepcopy(actor.record())
                choices = {}
                for label, probabilities in [('actual_prediction', t['response_prediction']),
                                             ('other_prediction', traces[other][t['tick']]['response_prediction']),
                                             ('true_current_law', t['truth'])]:
                    detached = copy.deepcopy(actor)
                    result = detached.decide(replace(request, reader=reader_for(game, probabilities)), False)
                    chosen = result['decision']['action_id']
                    # Also expose the score-maximizing action and personal
                    # policy without reader-switch hysteresis; do not call a
                    # difference a mistake or alter the actor's preferences.
                    prepared = actor._prepare(request)
                    exact_context = reader_for(game, probabilities)(prepared['effective'], 64).context
                    batch = compile_batch([exact_context])
                    numeric = actor.policy.decide(batch, False)
                    personal = numeric.records(batch)[0]['action_id']
                    choices[label] = dict(action=chosen, pure_policy=personal,
                                          reading=result['reading'], progress=result.get('progress'),
                                          personal_scores=dict(zip(batch.ids[0], map(float, numeric.scores[0]))))
                assert choices['actual_prediction']['action'] == t['action']
                assert actor.record() == fixed
                assert character_key(request.context) == actor.character
                objective = {key: sum(p * (game.goal(game.transition(t['before'], key, z)[0]).value
                                           - game.goal(t['before']).value) for z, p in t['truth'].items())
                             for key in game.keys(t['before'])}
                truth_choice = choices['true_current_law']['action']
                out.append(dict(tick=t['tick'], arm=arm, before=t['before'], factual=t['action'],
                                choices=choices, true_immediate_increments=objective,
                                prediction_change=choices['other_prediction']['action'] != t['action'],
                                true_law_changes_choice=truth_choice != t['action'],
                                correction_immediate_delta=objective[truth_choice]-objective[t['action']],
                                oracle_persona_gap=max(objective.values())-objective[truth_choice]))
            result = DecisionLoop.decide_batch([(actor, request)], False,
                                               _counterfactual_roots=(t['action'],))[0]
            flow = game.transition(t['before'], t['action'], t['response'])[1]
            resolve_feedback(actor, result['ticket'], game.feedback(t['before'], t['after'],
                             t['action'], result['context'], flow))
            assert digest(actor.record()) == t['owner']
        assert actor.record() == row['arms'][arm]['final_owner']
    return dict(spec=spec, source_sha256=sha(raw), delta=row['delta'], disagreements=len(changed),
                audited_steps=sum(map(len, traces.values())), probes=out)


def run(root, output, workers):
    root, output = Path(root), Path(output)
    audit = json.loads((root/'summary.json').read_bytes())
    assert audit['complete'] and audit['pairs'] == 256
    if output.exists():
        raise FileExistsError('retain existing attribution')
    output.mkdir(parents=True)
    rows = []
    paths = [root/name for name in sorted(audit['artifacts'])]
    for p in paths: assert sha(p.read_bytes()) == audit['artifacts'][p.name]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed(pool.submit(one, str(p)) for p in paths):
            row = future.result(); rows.append(row)
            (output/(row['spec']['id']+'.json.gz')).write_bytes(gzip.compress(encoded(row), mtime=0))
            (output/'status.json').write_bytes(encoded(dict(complete=len(rows)==len(paths), pairs=len(rows))))
            if len(rows)%16 == 0: print(json.dumps(dict(pairs=len(rows), total=len(paths))), flush=True)
    summaries = {}
    for game in sorted({r['spec']['game'] for r in rows}):
        summaries[game] = {}
        for arm in ('stable','transfer'):
            entries = [e for r in rows if r['spec']['game']==game for e in r['probes'] if e['arm']==arm]
            summaries[game][arm] = dict(probes=len(entries),
                changed_by_true_law=sum(e['true_law_changes_choice'] for e in entries),
                unchanged_by_true_law=sum(not e['true_law_changes_choice'] for e in entries),
                changed_by_other_prediction=sum(e['prediction_change'] for e in entries),
                oracle_policy_score_cost=sum(e['oracle_persona_gap'] for e in entries),
                correction_immediate_delta=sum(e['correction_immediate_delta'] for e in entries))
    report = dict(complete=True,pairs=len(rows),audited_steps=sum(r['audited_steps'] for r in rows),
                  source_sha256=sha(Path(__file__).read_bytes()), summary=summaries,
                  scope='Post-hoc local attribution on all action disagreements; no hidden information supplied to factual actors. Not a long-run counterfactual or a personality-quality gate.',
                  artifacts={p.name:sha(p.read_bytes()) for p in output.glob('*.json.gz')})
    (output/'review.json').write_bytes(encoded(report))
    print(json.dumps({k:v for k,v in report.items() if k!='artifacts'}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a.input,a.output,a.workers)
