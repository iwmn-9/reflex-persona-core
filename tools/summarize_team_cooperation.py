"""Exploratory support causality, personality expression and partner coverage.

Same-turn interventions retain every other committed intent and world hit seed.
They are diagnostics, not independent matches or long-term welfare estimates.
"""
import argparse
from collections import defaultdict,Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.analyze_team_cooperation import check_combat,check_projects,seal


def summarize(root):
    root=Path(root);evaluation=json.loads((root/'evaluation.json').read_text())
    raw=root/'trajectories.jsonl'
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==evaluation['trajectories_sha256']
    snapshot=json.loads((root/'source_snapshot.json').read_text())
    for name,sha in snapshot.items():
        assert hashlib.sha256((root/'_source'/name).read_bytes()).hexdigest()==sha
    sys.path.insert(0,str(root/'_source'))
    from reflex import team_projects as tp
    from reflex.combat import resolve as combat_resolve
    from reflex.combat import battle_from_record
    counts=defaultdict(Counter);first=defaultdict(dict);world_histories=defaultdict(set);examples=[];rows=0
    for line in raw.open(encoding='utf-8'):
        row=json.loads(line);rows+=1;key=(row['genre'],row['participation'],row['mode'])
        group=counts[key];own={int(i) for i in row['profiles']}
        world_histories[key].add(hashlib.sha256(seal([row['team'],
            [(t['before'],t['choices'],t['after']) for t in row['trace']]]).encode()).hexdigest())
        t=row['trace'][0]
        first[(row['genre'],row['scenario'],row['participation'],row['mode'],row['seed'])][row['roster']]=tuple(t['choices'][str(i)] for i in sorted(own))
        for turn in row['trace']:
            before=turn['before'];actual=turn['after'];choices={int(i):v for i,v in turn['choices'].items()}
            learning=turn.get('partner_learning')
            if learning:
                for actor,models in learning['models'].items():
                    assert tuple(models)==('hold','independent')
                    weights=learning['before']['weights'][actor];action=turn['choices'][actor]
                    probability=sum(w*models[name].get(action,0.) for w,name in zip(weights,models))
                    group['observed_partner_actions']+=1
                    group['outside_all_declared_types']+=not any(m.get(action,0.)>0 for m in models.values())
                    group['current_action_probability_sum']+=probability
            for actor,action in choices.items():
                if actor not in own or not action.startswith(('heal:','aid:')):continue
                target=int(action.split(':')[1])
                if target==actor:continue
                alt=dict(choices);alt[actor]='guard' if row['genre']=='combat' else 'rest'
                if row['genre']=='combat':
                    w=battle_from_record(before)
                    # The original seed is the public study's disjoint world stream.
                    from reflex.core import digest
                    seed=int(digest(['team-cooperation-world',row['seed']])[:16],16)
                    after,audit=combat_resolve(w,alt,seed);counter=asdict(after)
                    check_combat(before,alt,counter,audit,seed)
                    benefit=actual['units'][target]['hp']-counter['units'][target]['hp']
                    own_cost=counter['units'][actor]['hp']-actual['units'][actor]['hp']
                    rescued=actual['units'][target]['hp']>0 and counter['units'][target]['hp']==0
                else:
                    w=tp.Workshop(**{**before,'people':tuple(tp.Worker(**p) for p in before['people']),
                        'stock':tuple(before['stock']),'points':tuple(before['points'])})
                    after,audit=tp.resolve(w,alt);counter=asdict(after)
                    check_projects(before,alt,counter,audit)
                    benefit=actual['people'][target]['energy']-counter['people'][target]['energy']
                    own_cost=counter['people'][actor]['energy']-actual['people'][actor]['energy'];rescued=False
                group['support_interventions']+=1;group['positive_immediate_recipient_effect']+=benefit>0
                group['zero_immediate_recipient_effect']+=benefit==0;group['recipient_effect_sum']+=benefit
                group['helper_health_or_energy_cost_sum']+=own_cost;group['same_turn_survival_rescues']+=rescued
                if (rescued or benefit==0) and len(examples)<12:
                    examples.append(dict(genre=row['genre'],scenario=row['scenario'],seed=row['seed'],
                        roster=row['roster'],mode=row['mode'],tick=before['tick'],actor=actor,target=target,
                        action=action,recipient_effect=benefit,helper_cost=own_cost,survival_rescue=rescued))
    diversity=defaultdict(list)
    for (genre,scenario,participation,mode,seed),rosters in first.items():
        diversity[(genre,participation,mode)].append(len(set(rosters.values())))
    result=dict(games=rows,exploratory=True,source_snapshot_sha256=evaluation['source_snapshot_sha256'],
        diagnostics=[dict(genre=k[0],participation=k[1],mode=k[2],counts=dict(v),
            distinct_actual_world_histories=len(world_histories[k]),
            first_turn_distinct_joint_choices_mean=sum(diversity[k])/len(diversity[k]),
            first_turn_roster_comparisons=len(diversity[k])) for k,v in sorted(counts.items())],examples=examples,
        limits=['support replacement holds other current intents fixed; no long-term causal claim',
                'first-turn diversity compares the declared roster arrangements, not all personality settings',
                'partner type coverage is not calibrated confidence or evidence of character motives',
                'counterfactuals are not additional games or empirical training data',
                'resource rules have no world randomness; repeated histories are not independent discoveries'])
    (root/'diagnostics.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(games=rows,diagnostics=result['diagnostics']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);summarize(p.parse_args().root)
