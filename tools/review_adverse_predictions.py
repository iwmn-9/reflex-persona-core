"""Separate observed-belief error from rollout sampling, using complete evidence.

Hidden response probabilities are offline scoring targets only. This report
cannot identify sampling error from one fixed model bank, and cannot adopt a
controller. It never imports a controller or executes a game.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path


def score_prediction(prediction, truth, response):
    if (set(prediction) != set(truth) or response not in truth
            or any(not math.isfinite(x) or x <= 0 for x in prediction.values())
            or any(not math.isfinite(x) or x < 0 for x in truth.values())
            or abs(sum(prediction.values())-1) > 1e-12
            or abs(sum(truth.values())-1) > 1e-12):
        raise ValueError('normalized predictions with common response support required')
    return dict(expected_brier_excess=sum((prediction[k]-q)**2 for k,q in truth.items()),
        expected_log_regret=sum(q*math.log(q/prediction[k]) for k,q in truth.items() if q),
        observed_log_loss=-math.log(prediction[response]),
        observed_brier=sum((p-float(k==response))**2 for k,p in prediction.items()))


def mean_scores(rows):
    return {key:sum(row[key] for row in rows)/len(rows) for key in rows[0]}


def review(plan, rows):
    expected={(s['id'],a) for s in plan['specs'] for a in plan['arms']}
    records={}
    for row in rows:
        key=(row['spec']['id'],row['arm'])
        if key in records:raise ValueError('duplicate execution')
        records[key]=row
    if set(records)!=expected:raise ValueError('all registered executions required')
    ref=plan.get('reference_arm','stable');candidate=plan.get('candidate_arm','stratified');pairs=[]
    for spec in plan['specs']:
        traces={};scores={}
        for arm in plan['arms']:
            item=records[(spec['id'],arm)];r=item['result'];tr=r['trace']
            if (item['spec']!=spec or not r['complete'] or len(tr)!=spec['turns']
                    or r['configuration']['spec']!=spec or r['configuration']['arm']!=arm):
                raise ValueError('complete registered history required')
            if [t['tick'] for t in tr]!=list(range(spec['turns'])):
                raise ValueError('ordered complete actual history required')
            traces[arm]=tr
            scores[arm]=[score_prediction(t['response_prediction'],t['truth'],t['response']) for t in tr]
        left,right=traces[ref],traces[candidate]
        if [(t['truth'],t['response']) for t in left]!=[(t['truth'],t['response']) for t in right]:
            raise ValueError('paired actual worlds changed')
        changed=[i for i,(x,y) in enumerate(zip(left,right)) if x['action']!=y['action']]
        divergence=None
        if changed:
            i=changed[0];x,y=left[i],right[i]
            divergence=dict(tick=i,actions={ref:x['action'],candidate:y['action']},
                same_world=x['before']==y['before'],same_owner=x['before_owner']==y['before_owner'],
                prediction={ref:x['response_prediction'],candidate:y['response_prediction']},
                evaluator_truth=x['truth'])
        paired_difference=max(abs(x['response_prediction'][k]-y['response_prediction'][k])
            for x,y in zip(left,right) for k in x['response_prediction'])
        pairs.append(dict(id=spec['id'],game=spec['game'],profile=spec['profile'],
            actual_prediction_max_difference=paired_difference,
            means={arm:mean_scores(values) for arm,values in scores.items()},
            phases=[dict(start=start,end=min(start+32,spec['turns']),
                means={arm:mean_scores(values[start:start+32]) for arm,values in scores.items()})
                for start in range(0,spec['turns'],32)],first_action_divergence=divergence))
    groups=defaultdict(list)
    for row in pairs:groups[row['game']].append(row)
    return dict(complete=True,conditions=len(pairs),pairs=pairs,
        families={game:{arm:mean_scores([row['means'][arm] for row in group])
            for arm in plan['arms']} for game,group in groups.items()},adoption=False,
        sampling_attribution='Not identifiable from one fixed-bank long-play comparison. Require same-world/same-owner forecasts over predeclared paired model banks.',
        limits=['True response laws are evaluator-only; their errors do not imply that the actor could have known the future.',
            'These scores measure actual one-step response beliefs, not the future-world rollout distribution or long-horizon action quality.',
            'Equal response beliefs do not imply equal future states, learned needs, plans or action values.',
            'Repeated personas and overlapping phases share observations; do not count them as independent samples.',
            'This diagnostic does not change any acceptance threshold, personality parameter, source or history.'])


def from_archive(directory):
    directory=Path(directory)
    proof=json.loads((directory/'verification.json').read_bytes())
    if not proof['complete']:raise ValueError('a complete audited archive is required')
    contents={}
    for name in ('plan.json','runs.jsonl.gz'):
        contents[name]=(directory/name).read_bytes()
        if hashlib.sha256(contents[name]).hexdigest()!=proof['artifacts'][name]:
            raise ValueError('audited input checksum changed: '+name)
    plan=json.loads(contents['plan.json'])
    rows=[json.loads(line) for line in gzip.decompress(contents['runs.jsonl.gz']).splitlines()]
    report=review(plan,rows)
    if (proof['pairs']!=report['conditions'] or proof['executions']!=len(rows)
            or proof['actual_steps']!=sum(len(row['result']['trace']) for row in rows)):
        raise ValueError('archive completion counts changed')
    report.update(source_ref=proof['source_ref'],inputs={name:proof['artifacts'][name] for name in contents},
        reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();report=from_archive(args.evidence)
    with Path(args.output).open('x',encoding='utf8') as stream:
        json.dump(report,stream,sort_keys=True,separators=(',',':'));stream.write('\n')
    print(json.dumps(dict(complete=True,conditions=report['conditions'],adoption=False,
        maximum_paired_prediction_difference=max(r['actual_prediction_max_difference'] for r in report['pairs']))))
