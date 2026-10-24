"""Recompute a complete fresh comparison before applying inherited criteria.

Only an offline reviewer sees the true world law. Conditional gains retain
personality, acquired state and effects of earlier luck; they are not a pure
intelligence metric. This program cannot promote a candidate or change a run.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def quality_requirements(rows):
    families=[]
    for name in sorted({r['spec']['game'] for r in rows}):
        group=[r for r in rows if r['spec']['game']==name]
        families.append(dict(game=name,count=len(group),
            mean_delta=sum(r['delta'] for r in group)/len(group),
            conditional_delta=sum(r['conditional_delta'] for r in group)/len(group),
            worse=sum(r['delta'] < -1e-12 for r in group)))
    by_name={r['game']:r for r in families}
    gambling=[r for r in rows if r['spec']['game']=='gambling']
    worlds=[]
    for seed in sorted({r['spec']['seed'] for r in gambling}):
        group=[r for r in gambling if r['spec']['seed']==seed]
        worlds.append(dict(seed=seed,profiles=sorted(r['spec']['profile'] for r in group),
            mean_delta=sum(r['delta'] for r in group)/len(group)))
    checks=dict(
        required_families=set(by_name)=={'auction','combat','resources','gambling'},
        no_family_loss=bool(families) and all(r['mean_delta']>=-1e-12 and r['conditional_delta']>=-1e-12 for r in families),
        positive_gambling=all(by_name.get('gambling',{}).get(k,0)>1e-12 for k in ('mean_delta','conditional_delta')),
        positive_resources=all(by_name.get('resources',{}).get(k,0)>1e-12 for k in ('mean_delta','conditional_delta')),
        four_complete_gambling_worlds=len(worlds)==4 and all(r['profiles']==[0,1,2,3] for r in worlds),
        three_gambling_worlds_improve=sum(r['mean_delta']>1e-12 for r in worlds)>=3)
    return dict(passed=all(checks.values()),checks=checks,families=families,gambling_worlds=worlds)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output')
    args=p.parse_args();root=Path(args.root).resolve()
    sha=lambda data:hashlib.sha256(data).hexdigest()
    manifest=json.loads((root/'source_snapshot.json').read_bytes())
    for name,value in manifest.items():assert sha((root/'_source'/name).read_bytes())==value
    # Import the exact rules that generated this study, never the current rules.
    sys.path.insert(0,str(root/'_source'))
    from tools.run_continuation_portfolio import game_for
    from tools.run_owned_value import law
    from tools.owned_checkpoint import encoded
    plan=json.loads((root/'plan.json').read_bytes())
    proof=json.loads((root/'verification.json').read_bytes())
    summary=json.loads((root/'summary.json').read_bytes())
    assert plan['source']==sha(encoded(manifest))
    assert proof['complete'] and proof['source_verified'] and proof['actual_owner_replayed']
    assert summary['complete'] and summary['completed_pairs']==summary['registered_pairs']==proof['pairs']==len(plan['specs'])
    rows={r['id']:r for r in summary['rows']}
    assert len(rows)==len(summary['rows'])==len(plan['specs'])
    assert set(rows)=={s['id'] for s in plan['specs']}
    recomputed=[];receipts=[];decisions=0
    for spec in plan['specs']:
        assert spec['mechanism']=='world', 'world comparison required'
        row=rows[spec['id']];assert row['spec']==spec
        results=[];conditional=[];paths=[]
        for side in ('baseline','candidate'):
            reference=spec.get('reference') if side=='baseline' else None
            path=root/reference['archive'] if reference else root/spec['id']/side/'result.json'
            data=path.read_bytes();result=json.loads(data)
            expected=spec
            if reference:
                assert sha(data)==reference['sha256']
                control_proof=json.loads((root/'controls_verification.json').read_bytes())
                receipt=next(r for r in control_proof['rows'] if r['id']==spec['id'])
                assert control_proof['complete'] and control_proof['actual_owner_replayed'] and control_proof['source_verified']
                assert receipt['result_sha256']==sha(data)
                assert control_proof['plan_sha256']==reference['audited_development_plan_sha256']
                assert control_proof['source']==reference['original_source']==plan['source']
                expected=result['configuration']['spec']
                # A control may differ only in the tested controller setting;
                # world rules, persona, random seed and duration must match.
                assert {k:v for k,v in expected.items() if k!='controller'}=={
                    k:v for k,v in spec.items() if k not in ('controller','reference')}
            assert result['complete'] and result['configuration']['spec']==expected
            assert result['configuration']['retry']==(side=='candidate')
            game=game_for(spec);world=game.start();increments=[]
            for step in result['trace']:
                assert step['before']==world
                truth=law(world['tick'],spec['turns'],spec['regime']);assert step['truth']==truth
                value=game.goal(world).value
                increment=sum(prob*(game.goal(game.transition(world,step['action'],symbol)[0]).value-value)
                    for symbol,prob in truth.items())
                assert increment==step['conditional_increment'], 'stored one-step expectation drift'
                increments.append(increment)
                world,_=game.transition(world,step['action'],step['response'])
                assert world==step['after'];decisions+=1
            assert world==result['final'] and game.terminal(world)
            assert game.goal(world).value==result['score']
            # The runner uses Python sum (compensated on Python >=3.12),
            # whereas repeated += rounds after every addition. Recompute the
            # same defined statistic; keep exact equality, not a tolerance.
            results.append(result);conditional.append(sum(increments))
            paths.append(dict(path=path.relative_to(root).as_posix(),sha256=sha(data)))
        assert results[0]['runtime']==results[1]['runtime']
        delta=results[1]['score']-results[0]['score'];cd=conditional[1]-conditional[0]
        assert row['scores']==[r['score'] for r in results]
        assert row['executed']==[len(r['trace']) for r in results]
        assert row['delta']==delta and row['conditional_delta']==cd, 'summary differs from actual results'
        recomputed.append(dict(id=spec['id'],spec=spec,delta=delta,conditional_delta=cd))
        receipts.append(dict(id=spec['id'],results=paths))
    out=dict(complete=True,pairs=len(recomputed),recomputed_actual_decisions=decisions,
        source=plan['source'],plan_sha256=sha((root/'plan.json').read_bytes()),
        summary_sha256=sha((root/'summary.json').read_bytes()),results=receipts,
        inherited_quality_requirements=quality_requirements(recomputed),adoption=False,
        limit='Numerical quality gate only. Original 40/8/8, individual losses and personality behavior still require review; no automatic adoption.')
    out['reused_controls']=sum('reference' in spec for spec in plan['specs'])
    target=Path(args.output) if args.output else root/'quality_review.json'
    target.write_bytes(encoded(out))
    print(json.dumps({k:v for k,v in out.items() if k!='results'}),flush=True)


if __name__=='__main__':main()
