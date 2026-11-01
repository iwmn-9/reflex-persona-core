"""Complete-pair long-play accounting; no quality threshold or policy changes."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reflex.laboratory import PROFILES
from tools.run_adverse_recovery import game_for
from tools.summarize_horizon_study import realized_account
from tools.owned_checkpoint import encoded,sha
from tools.review_adverse_recovery import summarize


def account_pair(spec, reference, candidate):
    game=game_for(spec);accounts={};behaviors={}
    for name,row in (('reference',reference),('candidate',candidate)):
        assert row['complete'] and game.terminal(row['final'])
        flat=dict(spec,profile=PROFILES[spec['profile']]['id'],trace=row['trace'],score=row['score'])
        accounts[name]=realized_account(flat)
        horizon=row['configuration']['horizon'];errors=[];tr=row['trace']
        for t in tr:
            target=min(t['tick']+horizon,len(tr))
            errors.append(t['selected_purpose']-game.goal(tr[target-1]['after']).value)
        behaviors[name]=dict(final=row['final'],actions=dict(Counter(t['action'] for t in tr)),
            selected_policies=dict(Counter(t['selected_policy'] for t in tr)),
            forecast_horizon=horizon,forecast_bias=sum(errors)/len(errors),
            forecast_rmse=math.sqrt(sum(x*x for x in errors)/len(errors)))
    phases=[]
    for start in range(0,spec['turns'],32):
        local={}
        for name,row in (('reference',reference),('candidate',candidate)):
            tr=row['trace'];a=accounts[name]
            pieces=a['increments'][start:start+32]
            if not pieces:local[name]=dict(score_gain=0.,conditional=0.,residual=0.);continue
            gain=game.goal(tr[start+len(pieces)-1]['after']).value-game.goal(tr[start]['before']).value
            local[name]=dict(score_gain=gain,conditional=sum(x['conditional'] for x in pieces),residual=sum(x['residual'] for x in pieces))
        phases.append(dict(start=start,end=min(start+32,spec['turns']),differences={k:local['candidate'][k]-local['reference'][k] for k in local['reference']}))
    a,b=accounts['reference'],accounts['candidate']
    delta=dict(score=candidate['score']-reference['score'],conditional=b['conditional_increments']-a['conditional_increments'],residual=b['residual']-a['residual'])
    if 'bellman' in a:
        delta['reference_choice']=b['bellman']['residual_removed']-a['bellman']['residual_removed']
        delta['reference_residual']=b['bellman']['residual']-a['bellman']['residual']
    assert abs(delta['score']-delta['conditional']-delta['residual'])<1e-10
    common=min(len(reference['trace']),len(candidate['trace']))
    has_predictions=all('response_prediction' in t for row in (reference,candidate) for t in row['trace'][:common])
    prediction_difference=(max(abs(reference['trace'][i]['response_prediction'][k]-candidate['trace'][i]['response_prediction'][k]) for i in range(common) for k in reference['trace'][i]['response_prediction']) if has_predictions else None)
    return dict(id=spec['id'],differences=delta,phases=phases,behaviors=behaviors,
        actual_prediction_max_difference=prediction_difference,accounts=accounts,
        realized_loss=delta['score'] < -1e-12,conditional_loss=delta['conditional'] < -1e-12)


def analyze(root,plan):
    root=Path(root);verified=summarize(root,plan)
    if not verified['complete']:raise ValueError('all registered pairs must be complete and audited before this review')
    rows=[];files={}
    for spec in plan['specs']:
        arms={}
        for arm in plan['arms']:
            path=root/f"adverse-{spec['id']}-{arm}"/'result.json'
            arms[arm]=json.loads(path.read_bytes());files[path.parent.name]=sha(path.read_bytes())
        rows.append(account_pair(spec,arms[plan.get('reference_arm','stable')],arms[plan.get('candidate_arm','stratified')]))
    return dict(complete=True,source_results=files,rows=rows,adoption=False,
        development_gate_passed=verified['development_gate_passed'],
        limits=['Conditional increments include acquired resources and personality, not pure intelligence.',
                'Perfect-information Bellman accounts include unknown future changes and personality sacrifice; never supplied to the controller.',
                'Forecast error includes subsequent replanning and realized randomness, not just estimator bias.',
                'Recorded predicted_actions is one illustrative branch, not the plan that the future person committed to execute.',
                'Complete development results are insufficient for adoption.'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=analyze(a.root,json.loads(Path(a.plan).read_bytes()))
    Path(a.output).write_bytes(encoded(result));print(json.dumps(dict(complete=True,pairs=len(result['rows']),adoption=False)))
