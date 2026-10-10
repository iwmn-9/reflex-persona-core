"""Matched public-state channel interventions on recorded learning divergence.

Post-hoc mechanism diagnosis, not a held-out strength test. Swap remembered
hypothesis weights, recent thresholds and conditional coefficients separately.
No future outcome or opponent private state is supplied to a controller.
"""
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.strong_search import PublicMemory, PERSONA
from reflex.strong_table import decide
from reflex.laboratory import PROFILES
from tools.intervene_goal_progress import position


def run(root,output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    diagnostic=json.loads((root/'owner_diagnostic.json').read_text(encoding='utf-8'))
    targets={(r['seed'],r['encounter'],r['tick']) for r in diagnostic['episode_divergences'] if r['before_last_card']}
    plan=dict(kind='post-hoc component intervention',targets=sorted(targets),
        channels=['trackers','recent','coefficients'],source_diagnostic_sha256=hashlib.sha256((root/'owner_diagnostic.json').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/'reflex/strong_search.py').read_bytes()).hexdigest())
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    saved={};results=[]
    for shard in range(4):
        for variant in ('progress','horizon_progress'):
            memories=None;series=None
            with (root/f'shard-{shard}'/variant/'trajectories.jsonl').open(encoding='utf-8') as f:
                for line in f:
                    t=json.loads(line);seed=t['seed'];key=(seed,t['encounter'],t['tick'])
                    if seed!=series:series=seed;memories=[PublicMemory('no_thanks',a) for a in range(4)]
                    s=position('no_thanks',t['before'])
                    if key in targets:saved[(variant,key)]=(t,copy.deepcopy(memories))
                    actor=s.turn
                    for observer in range(4):
                        if observer!=actor:memories[observer].observe(s,actor,t['moves'][str(actor)],f'encounter-{t["encounter"]}-tick-{t["tick"]}-actor-{actor}')
    channels=plan['channels']
    for key in sorted(targets):
        old,ma=saved[('progress',key)];new,mb=saved[('horizon_progress',key)]
        assert old['before']==new['before'];actor=old['before']['turn'];bench=old['benchmark'];assert actor!=bench
        others=[p for i,p in enumerate(PROFILES) if i!=(key[0]//4)%4]
        roster={a:p for a,p in zip([a for a in range(4) if a!=bench],others)};s=position('no_thanks',old['before'])
        state=old['decisions'][str(actor)]['context']['state'];rows=[]
        assert state==new['decisions'][str(actor)]['context']['state']
        for flags in itertools.product((False,True),repeat=3):
            memory=copy.deepcopy(ma[actor])
            for channel,on in zip(channels,flags):
                if on:setattr(memory,channel,copy.deepcopy(getattr(mb[actor],channel)))
            _,d,stats=decide('no_thanks',s,actor,roster[actor],*key,memory,state,'adaptive',PERSONA,variant='progress')
            rows.append(dict(changed_channels=[c for c,on in zip(channels,flags) if on],action=d['action_id'],
                predicted_actions=stats['actions'],public_prediction={str(a):memory.predict(s,a) for a in range(4) if a!=actor}))
        assert rows[0]['action']==old['moves'][str(actor)] and rows[-1]['action']==new['moves'][str(actor)]
        results.append(dict(seed=key[0],encounter=key[1],tick=key[2],actor=actor,public_before=old['before'],
            old_action=rows[0]['action'],new_action=rows[-1]['action'],interventions=rows))
        print(key,rows[0]['action'],rows[-1]['action'],flush=True)
    summary={c:sum(r['interventions'][1<< (2-i)]['action']!=r['old_action'] for r in results) for i,c in enumerate(channels)}
    payload=dict(plan=plan,cases=results,single_channel_changed_choices=summary,limitations=['post-hoc sampled search interventions, not additive attribution or causal win effects'])
    (output/'evaluation.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8');print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.root,a.output)
