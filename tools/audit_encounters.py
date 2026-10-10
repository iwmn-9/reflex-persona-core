"""Recompute actual rules and pre-choice forecasts from saved public histories.

Does not call the world resolver or the policy. Full source-checked regeneration
is separate: python -m reflex.encounter_experiment --output ROOT --audit.
"""
from collections import Counter, deque
import argparse
import json
import math
from pathlib import Path
import numpy as np


def check(root):
    root=Path(root);result=json.loads((root/'evaluation.json').read_text(encoding='utf-8'))
    plan=result['plan'];actual=terminal=forecasts=0;runs=[];repairs=[]
    with (root/'trajectories.jsonl').open(encoding='utf-8') as f:
        for line in f:
            saved=json.loads(line);r=saved['run'];actors=[a['npc'] for a in r['actors']];n=len(actors)
            history={a:deque(maxlen=8) for a in actors}
            previous={a:dict(energy=.8,health=8.,stock=3.,score=0.) for a in actors}
            rules=plan['rules'][r['domain']];counts=Counter();harm=set();cooperative=Counter();examples=[]
            for tick,t in enumerate(saved['trace']):
                assert t['tick']==tick and t['fixed_personality']
                rows={x['npc']:x for x in t['rows']};assert list(rows)==actors
                rotated=actors.copy()
                for _ in range((tick//4)%(n-1)):rotated=[rotated[0],rotated[-1],*rotated[1:-1]]
                available={}
                for a,b in zip(rotated[:n//2],reversed(rotated[n//2:])):available[a]=b;available[b]=a
                luck=np.random.default_rng(np.random.SeedSequence([r['seed'],tick])).random((n,n))
                for own,row in rows.items():
                    assert row['before']==previous[own]
                    for target,p in t['forecasts'][own].items():
                        h=history[target]
                        if previous[target]['health']<=0 or previous[target]['energy']<.18 or (r['contact']=='public' and target!=available[own]):
                            expected=(0.,0.,1.)
                        elif r['contact']=='public':
                            pc=(sum(k=='cooperate' for k,_ in h)+1)/(len(h)+3)
                            pt=(sum(k=='claim' for k,_ in h)+1)/(len(h)+3);expected=(pc,pt,1-pc-pt)
                        else:
                            pc=(sum(k=='cooperate' and a==own for k,a in h)+1/(n-1))/(len(h)+2)
                            pt=(sum(k=='claim' and a==own for k,a in h)+1/(n-1))/(len(h)+2);expected=(pc,pt,1-pc-pt)
                        assert all(math.isclose(a,b,rel_tol=0,abs_tol=1e-12) for a,b in zip(p,expected));forecasts+=1
                    kind=row['kind'];target=row['target'];before=row['before']
                    if kind=='finished':
                        assert before['health']<=0 and row['after']==before and row['gain']==0
                        terminal+=1;counts['terminal_slots']+=1;continue
                    assert before['health']>0;actual+=1;counts['actual_choices']+=1
                    if kind=='rest':assert row['action']=='rest' and target is None
                    else:assert before['energy']>=.18
                    if kind in ('cooperate','claim'):
                        assert target in rows and target!=own and row['action']==f'{kind}:{target}'
                        assert previous[target]['health']>0
                        if r['contact']=='public':assert target==available[own]
                    else:assert kind in ('alone','rest') and target is None
                    matched=target is not None and rows[target]['target']==own
                    assert matched==row['matched']
                    other=rows[target]['kind'] if matched else None;assert other==row['other_kind']
                    i=actors.index(own);j=actors.index(target) if target else i
                    failure=bool(matched and kind==other=='cooperate' and luck[min(i,j),max(i,j)]<.05)
                    assert failure==row['luck_failure']
                    if kind in ('alone','rest'):gain,health=rules[kind]
                    elif not matched:gain,health=rules['unmatched'];counts['unmatched_attempts']+=1
                    elif failure:gain,health=-.1,-.15
                    else:gain,health=rules[('c' if kind=='cooperate' else 't')+('c' if other=='cooperate' else 't')]
                    energy=min(1,before['energy']+.4) if kind=='rest' else max(0,before['energy']-.18)
                    expected=dict(energy=max(0,energy-.025),health=max(0,min(8,before['health']+health)-rules['upkeep']),
                        stock=max(0,before['stock']+gain-.08),score=before['score']+gain)
                    assert row['after']==expected and row['gain']==gain
                    if matched:
                        key=(own,target)
                        if kind==other=='cooperate':
                            counts['cooperative_actor_turns']+=1;cooperative[key]+=1
                            if key in harm and not failure:
                                counts['cooperation_after_harm']+=1
                                if len(examples)<3:examples.append(dict(tick=tick,npc=own,target=target,profile=row['profile']))
                        if kind=='cooperate' and other=='claim':harm.add(key);counts['exploited']+=1
                        if kind==other=='claim':counts['mutual_claim_actor_turns']+=1
                        if failure:counts['joint_luck_failure_actor_turns']+=1
                previous={a:rows[a]['after'] for a in actors}
                for own,row in rows.items():history[own].append((row['kind'],row['target']))
            counts['cooperation_pairs']=counts['cooperative_actor_turns']//2
            counts['mutual_claim_pairs']=counts['mutual_claim_actor_turns']//2
            counts['repeated_cooperative_directed_relations']=sum(c>=3 for c in cooperative.values())
            # Counter omits zero keys while the experiment may materialize them.
            assert Counter(counts)==Counter(r['interaction']) and examples==r['repair_examples']
            for a in r['actors']:
                assert all(a[k]==previous[a['npc']][k] for k in ('energy','health','stock','score'))
            assert r['alive']==sum(a['health']>0 for a in r['actors'])
            assert math.isclose(r['mean_score'],sum(a['score'] for a in r['actors'])/n,rel_tol=0,abs_tol=1e-12)
            if examples:repairs.append(dict(seed=r['seed'],domain=r['domain'],layout=r['layout'],contact=r['contact'],setting=r['setting'],examples=examples))
            runs.append(r)
    assert runs==result['runs'] and actual==result['checks']['actual_choices']
    proof=json.loads((root/'audit.json').read_text(encoding='utf-8'))
    proof.update(independent_rule_transitions=actual,independent_terminal_slots=terminal,
        public_history_forecasts_checked=forecasts,forecast_and_mean_absolute_tolerance=1e-12,
        current_intent_not_used_in_forecast=True,cooperation_after_harm_worlds=len(repairs),
        illustrative_repair_worlds=repairs[:5])
    (root/'audit.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args()
    print(json.dumps(check(args.root),indent=2))
