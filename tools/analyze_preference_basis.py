"""Validate registered matched comparisons and retain every adverse case."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def analyze(root):
    reg=json.loads((root/'preregister.json').read_text(encoding='utf-8'))
    result=json.loads((root/'evaluation.json').read_text(encoding='utf-8'))
    rows=[json.loads(line) for line in (root/'trajectories.jsonl').read_text(encoding='utf-8').splitlines()]
    assert result['errors']==[] and len(rows)==len(reg['jobs'])==result['complete']
    assert len({r['index'] for r in rows})==len(rows)
    assert sorted(r['job'] for r in rows)==sorted(reg['jobs'])
    groups=defaultdict(list);examples=[];paths=0;diversity=defaultdict(list)
    for row in sorted(rows,key=lambda r:r['index']):
        for c in row['cases']:
            a=c['attribution'];v=a['values'];loss=a['loss'];arms=c['arms']
            assert abs(sum(loss.values())-a['total'])<1e-10
            assert abs(a['total']-(v['optimal']-v['legacy']))<1e-10
            assert loss['persona']>=-1e-10
            if 'persona_parts' in a:
                assert abs(sum(a['persona_parts'].values())-loss['persona'])<1e-10
                assert all(v>=-1e-10 for v in a['persona_parts'].values())
            if a.get('interaction') is not None:
                x=a['interaction'];assert abs(x['proxy_with_actor_law']-x['proxy_with_reference_law']-x['difference'])<1e-10
            assert all(x['reference_value']<=v['optimal']+1e-10 for x in arms.values())
            assert abs(arms['legacy']['reference_value']-v['legacy'])<1e-10
            assert abs(arms['repaired']['reference_value']-v['repaired'])<1e-10
            for arm in arms.values():
                audit=arm['root'];action=arm['action'];chosen=audit['actions'][action]
                assert audit['selected']==action and chosen['principle_eligible'] and chosen['corridor_eligible']
                assert chosen['score']>=max(x['score'] for x in audit['actions'].values() if x['corridor_eligible'])-1e-10
            assert len(c['execution_paths'])==3 and all(p[0]==arms['repaired']['action'] for p in c['execution_paths'])
            paths+=sum(len(p) for p in c['execution_paths'])
            key=row['game']+'/'+c['goal'];groups[key].append(c)
            diversity[key].append((row['index'],c['profile'],arms['legacy']['action'],arms['repaired']['action']))
            examples.append(dict(job=row['job'],public_start=row['public_start'],game=row['game'],goal=c['goal'],profile=c['profile'],
                proxy_gain=loss['proxy'],persona_loss=loss['persona'],prediction_loss=loss['prediction'],arms=arms,attribution=a))
    summary={}
    for key,cases in groups.items():
        def values(arm):return [c['arms'][arm]['reference_value'] for c in cases]
        def stats(d):return dict(mean=math.fsum(d)/len(d),min=min(d),max=max(d),improved=sum(x>1e-10 for x in d),same=sum(abs(x)<=1e-10 for x in d),worse=sum(x< -1e-10 for x in d))
        deltas={arm:stats([x-y for x,y in zip(values(arm),values('legacy'))]) for arm in ('deduplicated','endpoint_only','repaired')}
        interaction=[c['arms']['repaired']['reference_value']-c['arms']['deduplicated']['reference_value']-c['arms']['endpoint_only']['reference_value']+c['arms']['legacy']['reference_value'] for c in cases]
        by_position=defaultdict(list)
        for i,p,b,r in diversity[key]:by_position[i].append((b,r))
        summary[key]=dict(cases=len(cases),deltas=deltas,means={arm:math.fsum(values(arm))/len(cases) for arm in cases[0]['arms']},
            loss_means={k:math.fsum(c['attribution']['loss'][k] for c in cases)/len(cases) for k in ('persona','prediction','proxy')},
            factorial_interaction=stats(interaction),root_changes=sum(c['arms']['legacy']['action']!=c['arms']['repaired']['action'] for c in cases),
            distinct_persona_openings=dict(legacy=sum(len({b for b,r in rs})>1 for rs in by_position.values()),repaired=sum(len({r for b,r in rs})>1 for rs in by_position.values()),positions=len(by_position)))
        if all(c['attribution'].get('interaction') is not None for c in cases):
            summary[key]['prediction_proxy_interaction']={k:stats([c['attribution']['interaction'][k] for c in cases]) for k in ('proxy_with_reference_law','proxy_with_actor_law','difference')}
    out=dict(cohort=reg['cohort'],positions=len(rows),terminal_openings=sum(r.get('terminal_opening',False) for r in rows),
        cases=len(examples),actual_decisions_checked=paths,elapsed=result['elapsed'],groups=summary,
        negative_cases=[x for x in examples if x['proxy_gain']< -1e-10],
        largest_proxy_gains=sorted(examples,key=lambda x:x['proxy_gain'],reverse=True)[:6],
        largest_persona_costs=sorted(examples,key=lambda x:x['persona_loss'],reverse=True)[:6],
        inputs={n:sha(root/n) for n in ('preregister.json','evaluation.json','trajectories.jsonl','source_snapshot.json')})
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=analyze(Path(a.root));Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('negative_cases','largest_proxy_gains','largest_persona_costs')},ensure_ascii=False))


if __name__=='__main__':main()
