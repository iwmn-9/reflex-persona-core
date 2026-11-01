"""Compare additive shared candidates to immutable parent controls, without rewriting provenance."""
from collections import defaultdict
import json
from pathlib import Path
import zipfile

from reflex.core import digest
from tools.owned_checkpoint import encoded, sha
from tools.run_adverse_recovery import replay
from tools.run_continuation_portfolio import game_for

ROOT=Path(__file__).resolve().parents[1]


def source(bundle):
    with zipfile.ZipFile(bundle) as z:
        raw=z.read('registered/plan.json');plan=json.loads(raw)
        manifest=json.loads(z.read('registered/registration.json'))
        assert sha(raw)==manifest['plan_sha256']
        for name,value in manifest['sources'].items():assert sha(z.read(name))==value
    return plan,manifest,dict(plan_sha256=manifest['plan_sha256'],source_sha256=sha(encoded(manifest['sources'])))


def execution(folder,spec,arm,scope,manifest,registration):
    folder=Path(folder)
    if not (folder/'result.json').exists():return None
    envelope=json.loads((folder/'checkpoint.json').read_bytes());payload=envelope['payload']
    assert sha(encoded(payload))==envelope['sha256']
    result=json.loads((folder/'result.json').read_bytes());cfg=result['configuration']
    assert result['complete'] is True and cfg==payload['contract']['configuration']
    assert cfg['spec']==spec and cfg['arm']==arm and cfg.get('continuation_scope','root')==scope
    assert cfg['registration']==registration and cfg['horizon']==32 and cfg['seeds']==[701,702,703]
    assert cfg['runner_sha256']==manifest['sources']['tools/run_adverse_recovery.py']
    for name,value in cfg['adapters'].items():
        assert manifest['sources']['tools/'+name]==value
        assert sha((ROOT/'tools'/name).read_bytes())==value
    for name,value in payload['contract']['sources'].items():
        assert manifest['sources'][name]==value
        # Replay uses the same core and physical/feedback definitions as execution.
        assert sha((ROOT/name).read_bytes())==value
    trace=result['trace'];n=len(trace);g=game_for(spec)
    assert trace==payload['trace'] and result['final']==payload['world']==trace[-1]['after']
    assert result['checks']==payload['checks']==dict(fixed_axes=n,owner_parity=n,restored=n)
    assert result['runtime']==payload['contract']['runtime']
    assert result['final_owner']==digest(payload['owner']) and result['final_rng']==payload['rng']
    assert g.terminal(result['final']) and result['score']==g.goal(result['final']).value
    assert result['replayed_actual_steps']==replay(g,spec,trace,payload['owner'],payload['rng'])==n
    for row in trace:
        actual=sum(p*(g.goal(g.transition(row['before'],row['action'],z)[0]).value-g.goal(row['before']).value)
                   for z,p in row['truth'].items())
        assert row['conditional_increment']==actual
    assert result['conditional_increment']==sum(t['conditional_increment'] for t in trace)
    return result


def compare(parent_root,candidate_root,parent_source,candidate_source,parity_receipt):
    parent,pm,pr=source(parent_source);plan,cm,cr=source(candidate_source)
    assert sha(Path(parent_source).read_bytes())==plan['parent_source_sha256']
    assert pm['plan_sha256']==plan['parent_plan_sha256']
    assert sha(encoded(pm))==plan['parent_registration_sha256']
    assert {s['id']:s for s in plan['specs']}=={s['id']:s for s in parent['specs']}
    assert plan['criteria']==parent['criteria'] and plan['independent_confirmation']==parent['independent_confirmation']
    for name,value in plan['common_sources'].items():
        assert pm['sources'][name]==cm['sources'][name]==value
    if json.loads(Path(parity_receipt).read_bytes()).get('returncode')!=0:
        raise ValueError('same-runtime old/new stable runner parity required')
    pairs=[];pending=[];root_results=[]
    for spec in plan['specs']:
        name=spec['id']
        stable=execution(Path(parent_root)/('adverse-'+name+'-stable'),spec,'stable','root',pm,pr)
        old=execution(Path(parent_root)/('adverse-'+name+'-expected'),spec,'expected','root',pm,pr)
        new=execution(Path(candidate_root)/'episodes'/name,spec,'expected','shared',cm,cr)
        if old is not None:root_results.append(name)
        if stable is None or new is None:
            pending.append(name);continue
        assert stable['runtime']==new['runtime']
        count=min(len(stable['trace']),len(new['trace']))
        assert [r['response'] for r in stable['trace'][:count]]==[r['response'] for r in new['trace'][:count]]
        pairs.append(dict(id=name,game=spec['game'],profile=spec['profile'],
            score_delta=new['score']-stable['score'],
            conditional_delta=new['conditional_increment']-stable['conditional_increment'],
            score=new['score'],stable_score=stable['score'],
            root_candidate_score=old['score'] if old is not None else None))
    by_family=defaultdict(list)
    for pair in pairs:by_family[pair['game']].append(pair)
    families={g:{k:sum(p[k] for p in members)/len(members) for k in ('score_delta','conditional_delta')}
              for g,members in by_family.items()}
    gates=dict(complete=len(pairs)==len(plan['specs']),parent_root_audit_complete=len(root_results)==len(plan['specs']),
        all_families=set(families)=={'auction','combat','resources','gambling'},
        no_family_loss=all(v>=-1e-12 for f in families.values() for v in f.values()),
        concordant_adverse_improvement=any(p['id'] in plan['concordant_adverse'] and p['score_delta']>1e-12 and p['conditional_delta']>1e-12 for p in pairs))
    return dict(complete=gates['complete'],pairs=pairs,pending=pending,families=families,
        gate_checks=gates,development_gate_passed=all(gates.values()),adoption=False,
        independent_confirmation='Requires C2 reference, C3 benefit and individual-loss review as well.',
        provenance=dict(parent_source=sha(Path(parent_source).read_bytes()),candidate_source=sha(Path(candidate_source).read_bytes()),
            parent_registration=pr,candidate_registration=cr))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent-root','candidate-root','parent-source','candidate-source','parity-receipt','output'):
        p.add_argument('--'+name,required=True)
    args=p.parse_args();values=vars(args);output=values.pop('output')
    result=compare(**values);Path(output).write_bytes(encoded(result))
    print(json.dumps({k:result[k] for k in ('complete','development_gate_passed','pending')}))
