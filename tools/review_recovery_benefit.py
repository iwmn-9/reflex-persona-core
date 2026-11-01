"""Offline C3 paired scoring. Missing, censored and unavailable cases cannot pass."""
from collections import defaultdict
import json
import math
from pathlib import Path

from tools.completion_metrics import summarize_trace
from tools.owned_checkpoint import encoded, sha
from tools.recovery_benefit import truth_after
from reflex.core import digest
from tools.recovery_benefit import load_origin, audit_branch
from tools.run_adverse_recovery import verify_registration
from tools.run_continuation_portfolio import game_for


def aggregate(specs, rows, *, window=32):
    expected = {(s['id'],m,e) for s in specs for m in ('changed','steady') for e in ('learning','frozen')}
    keys = [(r['id'],r['world_mode'],r['experience']) for r in rows]
    if len(keys) != len(set(keys)) or not set(keys) <= expected:
        raise ValueError('registered branches exactly once; no substitute cases')
    lookup=dict(zip(keys,rows)); pairs=[]; groups=defaultdict(list); missing=sorted(expected-set(keys))
    for spec in specs:
        for mode in ('changed','steady'):
            arms=[lookup.get((spec['id'],mode,e)) for e in ('learning','frozen')]
            full=all(r is not None and r['window_complete'] and r['actual_steps']==window for r in arms)
            for row in arms:
                if row is None or row.get('unavailable'):continue
                for name in ('score_increment','conditional_increment'):
                    if type(row[name]) not in (int,float) or not math.isfinite(row[name]):
                        raise ValueError('finite observed increments required')
            if all(r is not None and not r.get('unavailable') for r in arms):
                a,b=arms
                if a['origin_sha256'] != b['origin_sha256'] or a['uniforms'][:min(len(a['uniforms']),len(b['uniforms']))] != b['uniforms'][:min(len(a['uniforms']),len(b['uniforms']))]:
                    raise ValueError('same actual origin and exogenous random stream required')
                delta={k:a[k]-b[k] for k in ('score_increment','conditional_increment')}
            else:delta=None
            pair=dict(id=spec['id'],game=spec['game'],profile=spec['profile'],world_mode=mode,
                window_complete=full,delta=delta,
                prediction_recovered=bool(arms[0] and arms[0]['window_complete']
                    and arms[0]['actual_steps']==window and arms[0]['recovered_within_16']))
            pairs.append(pair);groups[mode,spec['game']].append(pair)
    families={mode:{} for mode in ('changed','steady')}
    for (mode,family), members in groups.items():
        full=all(r['window_complete'] and r['delta'] is not None for r in members)
        families[mode][family]=dict(registered=len(members),complete=full,
            mean_delta={k:math.fsum(r['delta'][k] for r in members)/len(members)
                        for k in ('score_increment','conditional_increment')} if full else None)
    all_full=not missing and all(p['window_complete'] for p in pairs)
    changed=[p for p in pairs if p['world_mode']=='changed']
    recovered=sum(p['prediction_recovered'] for p in changed)
    rate=recovered/len(specs) if specs else None
    gates=dict(all_registered_windows_complete=bool(all_full),
        four_families=all(set(families[m])=={'auction','combat','resources','gambling'} for m in families),
        no_expected_family_loss=all_full and all(f['mean_delta']['conditional_increment']>=-1e-12 for m in families.values() for f in m.values()),
        changed_expected_gain=all_full and any(f['mean_delta']['conditional_increment']>1e-12 for f in families['changed'].values()),
        prediction_recovery=rate is not None and rate>=.9)
    return dict(complete=all_full,registered_branches=len(expected),received_branches=len(rows),
        missing=missing,pairs=pairs,families=families,prediction_recovered=recovered,
        prediction_denominator=len(specs),prediction_rate=rate,gate_checks=gates,
        C3_development_passed=all(gates.values()),adoption=False,
        limitations='Actual-experience intervention on known cases; no predictor-only causality, population inference or unseen-game claim.')


def review(root, plan_path):
    plan_path=Path(plan_path);plan=json.loads(plan_path.read_bytes())
    registration=verify_registration(plan_path)
    source_manifest=json.loads((plan_path.parent/'registration.json').read_bytes())['sources']
    root=Path(root);rows=[]
    for spec in plan['specs']:
        origin_path=root/'origins'/(spec['id']+'.json')
        unavailable=root/'origins'/(spec['id']+'.unavailable.json')
        if unavailable.exists():
            if origin_path.exists():raise ValueError('origin availability is contradictory')
            for mode in ('changed','steady'):
                for experience in ('learning','frozen'):
                    rows.append(dict(id=spec['id'],world_mode=mode,experience=experience,
                        unavailable=True,window_complete=False,actual_steps=0))
            continue
        if not origin_path.exists():continue
        origin=load_origin(origin_path,spec,plan['boundary'])
        g=game_for(spec)
        assert origin['contract']['configuration']['registration']==registration
        for name,value in origin['contract']['sources'].items():
            assert source_manifest[name]==value
        for mode in ('changed','steady'):
            for experience in ('learning','frozen'):
                folder=root/'branches'/f"{spec['id']}--{mode}--{experience}"
                if not (folder/'result.json').exists():continue
                result=json.loads((folder/'result.json').read_bytes())
                checkpoint=json.loads((folder/'checkpoint.json').read_bytes());payload=checkpoint['payload']
                assert sha(encoded(payload))==checkpoint['sha256']
                cfg=result['configuration'];boundary=plan['boundary'];end=boundary+plan['window']
                assert result['complete'] is True
                assert payload['contract']['sources']==origin['contract']['sources']
                assert payload['contract']['runtime']==origin['contract']['runtime']==result['runtime']
                for name,value in cfg['sources'].items():assert source_manifest[name]==value
                assert cfg==payload['contract']['configuration']
                assert cfg['spec']==spec and cfg['registration']==registration
                assert cfg['world_mode']==mode and cfg['experience']==experience
                assert cfg['origin_sha256']==sha(origin_path.read_bytes())
                assert cfg['boundary']==boundary and cfg['window']==plan['window']
                trace=result['trace'];n=len(trace)
                assert boundary <= n <= end
                assert trace==payload['trace'] and trace[:boundary]==origin['trace']
                assert result['final']==payload['world']==trace[-1]['after']
                assert result['final_rng']==payload['rng']
                assert result['final_owner']==digest(payload['owner'])
                assert result['origin_owner']==digest(origin['owner'])
                assert result['origin_world']==origin['world'] and result['origin_rng']==origin['rng']
                assert result['checks']==payload['checks']==dict(fixed_axes=n,owner_parity=n,restored=n)
                assert result['actual_steps']==n-boundary and result['window_complete']==(n==end)
                assert result['conditional_increment']==sum(t['conditional_increment'] for t in trace[boundary:])
                assert result['score_increment']==g.goal(result['final']).value-g.goal(origin['world']).value
                assert result['world_terminal']==g.terminal(result['final'])
                assert n==end or result['world_terminal'], 'unresolved partial window presented as a result'
                audit_branch(g,spec,origin,trace,payload['owner'],payload['rng'],mode,experience,boundary)
                metrics=summarize_trace(trace,scheduled_turns=end,terminal=result['world_terminal'],
                    truth_schedule=[truth_after(spec,t,boundary,mode) for t in range(end)])
                changes=[r for r in metrics['regimes'] if r['start']==boundary]
                rows.append(dict(id=spec['id'],world_mode=mode,experience=experience,
                    window_complete=result['window_complete'],actual_steps=n-boundary,
                    origin_sha256=cfg['origin_sha256'],score_increment=result['score_increment'],
                    conditional_increment=result['conditional_increment'],
                    uniforms=[t['uniform'] for t in trace[boundary:]],
                    recovered_within_16=bool(changes and changes[0]['within_16'])))
    return aggregate(plan['specs'],rows,window=plan['window'])


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--plan',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    result=review(args.root,args.plan)
    Path(args.output).write_bytes(encoded(result))
    print(json.dumps({k:result[k] for k in ('complete','received_branches','registered_branches','C3_development_passed')}))
