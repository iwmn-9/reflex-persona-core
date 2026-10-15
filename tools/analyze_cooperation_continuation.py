"""Observe agreement continuation without treating replanning as betrayal."""
from collections import Counter,defaultdict
from pathlib import Path
import argparse
import gzip
import json
from reflex.encounters import EncounterWorld
from tools.cooperation_experiment import configure


def analyze(root):
    root=Path(root);data=json.loads((root/'results.json').read_text());groups=defaultdict(Counter);examples=[]
    for r in data['runs']:
        if r['arm'] not in ('combined','continued'):continue
        with gzip.open(root/r['trajectory'],'rt',encoding='utf-8') as f:d=json.load(f)
        trace=d['trace'];w=EncounterWorld(r['seed'],r['domain'],r['layout'],r['contact'])
        group=groups[r['domain']+'|'+r['scenario']+'|'+r['contact']+'|'+r['arm']]
        for i,t in enumerate(trace[:-1]):
            following=trace[i+1];now={a['npc']:a for a in t['rows']};later={a['npc']:a for a in following['rows']}
            for ag in t['negotiation']['agreements']:
                members=ag['members'];group['one_step_followups']+=1
                repeated=any(a['offer']==ag['offer'] for a in following['negotiation']['agreements'])
                group['same_method_renewed' if repeated else 'method_not_renewed']+=1
                if all(now[m]['action'].startswith(('cooperate:','claim:')) for m in members):
                    # Expected next action of the conditional method, checked at
                    # the actually observed next state (not an imagined state).
                    contact=True
                    if r['contact']=='public':
                        w.tick=i+1;contact=w.available_partner(members[0])==members[1]
                    low=any(later[m]['before']['energy']<.18 for m in members)
                    expected={m:('finished' if later[m]['before']['health']<=0 else
                        ('rest' if low else now[m]['action']) if contact else
                        ('rest' if later[m]['before']['energy']<.18 else 'alone')) for m in members}
                    matched=all(later[m]['action']==expected[m] for m in members)
                    group['announced_method_continuations']+=1
                    group['next_roles_followed' if matched else 'next_roles_revised']+=1
                if r['scenario']=='change' and i==159:
                    group['changed_environment_renewed' if repeated else 'changed_environment_withdrawn']+=1
        if r['seed']==92100 and r['domain']=='watch' and r['scenario'] in ('standard','change'):
            examples.append(dict(world=r['trajectory'],profiles=r['actors'],
                turns=[dict(tick=t['tick'],actions=[dict(npc=a['npc'],profile=a['profile'],action=a['action'],outside=a['outside'],health=a['before']['health']) for a in t['rows']],
                    agreements=t['negotiation']['agreements']) for t in trace if t['tick'] in (0,1,66,67,68,159,160,161)]))
    result=dict(groups={k:dict(v) for k,v in groups.items()},examples=examples,
        interpretation='A fresh ballot governs every real action. Renewal/revision is observed behavior, not honesty, optimality, or a forecast calibration score. The model assumes continued consent; this diagnostic exposes that assumption.')
    (root/'continuation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result['groups'],indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();analyze(a.root)
