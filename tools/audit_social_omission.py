"""Independent trace aggregates for the fixed omission comparison.

The experiment's --audit separately checks source hashes, all regenerated runs
and rule transitions. This checks choices, pair ownership and published totals.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path


def audit(root):
    root=Path(root);r=json.loads((root/'evaluation.json').read_text(encoding='utf-8'))
    rows={};levels={'continuous':set(),'coarse':set()};changes=retaliations=personality=0
    with (root/'trajectories.jsonl').open(encoding='utf-8') as f:
        for line in f:
            saved=json.loads(line);run=saved['run'];trace=saved['trace']
            key=tuple(run[k] for k in ('domain','schedule','seed','setting'))
            assert key not in rows;rows[key]=saved
            for t in trace:
                retaliations+=t['action']=='retaliate';personality+=not t['values_unchanged']
                if run['setting']!='prediction_only':
                    for row in t['appraisal'].values():
                        signal=row['stance'] if run['setting']=='coarse' else row['attitude']*row['confidence']
                        levels[run['setting']].add(round(signal,12))
            for actor in run['actors']:
                own=[t for t in trace if t['npc']==actor['npc']]
                # Python's compensated sum and NumPy's per-turn accumulation
                # can differ in the last bit; only this aggregate is tolerant.
                assert math.isclose(sum(t['actual_gain'] for t in own),actor['score'],rel_tol=0,abs_tol=1e-12)
                assert [sum(t['action']=='aid' for t in own if t['phase']==p) for p in range(3)]==actor['aid_by_phase']
                assert sum(t['action']=='aid' and not t['cooperate'] for t in own)==actor['aid_to_exploiter']
                assert (own[-1]['after']['health']>0)==actor['alive']
    pairs=Counter();deaths=0;switches=Counter();minimum_delta=0.
    for (domain,schedule,seed,setting),saved in rows.items():
        if setting!='coarse':continue
        old=rows[domain,schedule,seed,'continuous']
        for a,b in zip(saved['trace'],old['trace']):
            assert (a['tick'],a['npc'])==(b['tick'],b['npc'])
            changes+=a['action']!=b['action']
        for a,b in zip(saved['run']['actors'],old['run']['actors']):
            assert a['npc']==b['npc'];delta=a['score']-b['score']
            pairs['better' if delta>1e-9 else 'worse' if delta< -1e-9 else 'same']+=1
            minimum_delta=min(minimum_delta,delta);deaths+=not a['alive'] and b['alive']
            for setting,row in (('coarse',saved),('continuous',old)):
                own=[t['action'] for t in row['trace'] if t['npc']==a['npc']]
                switches[setting]+=sum(x!=y for x,y in zip(own,own[1:]))
    assert changes==r['checks']['changed_choices'] and retaliations==r['checks']['retaliations']==0
    assert personality==r['checks']['personality_changes']==0
    assert {s:len(v) for s,v in levels.items()}==r['relation_levels']
    assert levels['coarse']=={-.5,0.,.5} and deaths==0 and sum(pairs.values())==512
    record=dict(independent_aggregate_checks=True,score_sum_absolute_tolerance=1e-12,paired_actor_episodes=512,paired_score_comparison=dict(pairs),
        new_deaths_per_matched_actor=deaths,changed_choices=changes,action_switches=dict(switches),
        largest_single_episode_score_drop=minimum_delta,relation_levels=r['relation_levels'],
        human_validation=False,enjoyment_measured=False)
    proof=json.loads((root/'audit.json').read_text(encoding='utf-8'));proof.update(record)
    (root/'audit.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8');return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args()
    print(json.dumps(audit(args.root),indent=2))
