"""Trace-level signal diagnostics and ordered-world matching, not new trials."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def diagnose(root):
    root=Path(root);registration=json.loads((root/'analysis_preregister.json').read_text(encoding='utf-8'))
    counts=defaultdict(Counter);forecasts=defaultdict(lambda:dict(n=0,predicted=0.,actual=0.));worlds={};cases=[];case_keys=set()
    for shard in registration['shards']:
        for variant in registration['variants']:
            folder=root/shard['folder']/variant;e=json.loads((folder/'evaluation.json').read_text(encoding='utf-8'))
            outcomes={(r['game'],r['seed'],r['encounter']):r for r in e['matches']}
            seen_actors=set();ordered={};last_cards={}
            with (folder/'trajectories.jsonl').open(encoding='utf-8') as f:
                for line in f:
                    t=json.loads(line);key=(t['game'],t['seed'],t['encounter']);game=key[0];b=t['before'];result=outcomes[key]
                    if game=='goofspiel':ordered[key]=b['prizes']
                    elif last_cards.get(key)!=b['card']:
                        ordered.setdefault(key,[]).append(b['card']);last_cards[key]=b['card']
                    for actor_text,d in t['decisions'].items():
                        actor=int(actor_text)
                        if actor==t['benchmark']:continue
                        stat=d['search'];guard=stat['guard'];chosen=d['action'];c=counts[game,variant]
                        c['npc_decisions']+=1
                        if len(stat['actions'])>1:
                            c['nonforced_decisions']+=1
                            c['nonforced_zero_goals']+=all(q['win_share']==0 for q in stat['actions'].values())
                        observation=(key,actor)
                        if observation not in seen_actors:
                            seen_actors.add(observation);q=forecasts[game,variant]
                            q['n']+=1;q['predicted']+=stat['actions'][chosen]['win_share'];q['actual']+=result['credits'][actor]
                        if guard.get('signal')!='goal_progress_on_constant_success':continue
                        c['progress_signals']+=1
                        if len(stat['actions'])<2:continue
                        c['progress_nonforced']+=1;old=guard['original_success_personality_action'];changed=old!=chosen
                        c['progress_changed_original_success_choice']+=changed
                        delta=stat['actions'][chosen]['goal_progress']-stat['actions'][old]['goal_progress']
                        c['progress_gain_sum']+=delta
                        c['progress_gain_positive']+=changed and delta>0
                        c['progress_gain_negative']+=changed and delta<0
                        pid=result['profiles'][actor_text];category=(variant,game,pid)
                        if changed and category not in case_keys:
                            case_keys.add(category)
                            cases.append(dict(game=game,variant=variant,seed=t['seed'],encounter=t['encounter'],tick=t['tick'],
                                actor=actor,profile=pid,before=b,selected=chosen,constant_goal_choice=old,
                                action_forecasts=stat['actions'],guard=guard,
                                actual_terminal_scores=result['scores'],actual_winner_credit=result['credits'][actor],
                                interpretation='same model and current state; actual ending is observed, alternative ending was not run'))
            for key,order in ordered.items():
                if key[0]=='no_thanks':assert len(order)==24 and len(set(order))==24
                if key in worlds:assert worlds[key]==order,'ordered worlds changed across variants'
                else:worlds[key]=order
    payload=dict(ordered_worlds_checked=len(worlds),variants=registration['variants'],
        ordered_world_digest=hashlib.sha256(json.dumps(sorted((str(k),v) for k,v in worlds.items())).encode()).hexdigest(),
        counts=[dict(game=g,variant=v,**dict(c)) for (g,v),c in counts.items()],
        first_choice_forecast=[dict(game=g,variant=v,**c,
            mean_forecast=c['predicted']/c['n'],actual_fraction=c['actual']/c['n']) for (g,v),c in forecasts.items()],
        cases=cases,limitations=['no causal attribution of an observed match win to one changed action',
            'first-choice model and actual future replanning have different continuation targets'])
    (root/'diagnostic.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(ordered_worlds_checked=len(worlds),counts=payload['counts'],first_choice_forecast=payload['first_choice_forecast']),indent=2))
    return payload


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();diagnose(a.root)
