"""Post-hoc first divergence: NPCs pursue own goals, not a three-player team.

Compare the first physical action divergence in each learning series. Later
encounters are excluded from this matched-history interpretation. The broad
variant may change more future controllers; the certified root must settle
without another nonforced choice and is separately identified.
"""
import argparse
import json
from pathlib import Path


def diagnose(root):
    root=Path(root);cases=[];episode_divergences=[]
    for shard in range(4):
        folder=root/f'shard-{shard}';plan=json.loads((folder/'analysis_preregister.json').read_text())
        before,after=plan['variants'];original={};finals={}
        for variant in (before,after):
            e=json.loads((folder/variant/'evaluation.json').read_text());finals[variant]={(r['seed'],r['encounter']):r for r in e['matches']}
        with (folder/before/'trajectories.jsonl').open(encoding='utf-8') as f:
            for line in f:
                t=json.loads(line);key=(t['seed'],t['encounter'],t['tick'])
                original[key]=dict(before=t['before'],moves=t['moves'],benchmark=t['benchmark'],
                    owner_states={a:d['context']['state'] for a,d in t['decisions'].items() if int(a)!=t['benchmark']})
        changed=set();changed_episodes=set()
        with (folder/after/'trajectories.jsonl').open(encoding='utf-8') as f:
            for line in f:
                t=json.loads(line);seed=t['seed']
                episode=(seed,t['encounter'])
                if episode in changed_episodes:continue
                key=(seed,t['encounter'],t['tick']);old=original[key]
                assert old['before']==t['before'],'public histories diverged before the first changed root'
                if old['moves']==t['moves']:continue
                changed_episodes.add(episode);actor=t['before']['turn'];a=str(actor)
                early=t['before']['remaining']>0
                episode_divergences.append(dict(seed=seed,encounter=t['encounter'],tick=t['tick'],remaining=t['before']['remaining'],
                    before_last_card=early,earlier_episode_action_history_already_changed=seed in changed,
                    interpretation='current interventions are inactive before last card; any early divergence is downstream of changed prior public learning/state' if early else 'last-card comparison'))
                if early:assert seed in changed
                if seed in changed:continue
                changed.add(seed)
                assert actor!=t['benchmark'];b=t['before'];ra=finals[before][key[:2]];rb=finals[after][key[:2]]
                known_terminal=(b['remaining']==0 and b['chips'][(actor+1)%4]==0)
                if after=='certified_expiry':assert known_terminal
                cases.append(dict(seed=seed,encounter=t['encounter'],tick=t['tick'],actor=actor,profile=ra['profiles'][a],
                    public_before=b,old_root=old['moves'][a],new_root=t['moves'][a],
                    owner_credit_before=ra['credits'][actor],owner_credit_after=rb['credits'][actor],
                    owner_score_before=ra['scores'][actor],owner_score_after=rb['scores'][actor],
                    npc_group_credit_before=1-ra['credits'][ra['benchmark']],npc_group_credit_after=1-rb['credits'][rb['benchmark']],
                    both_roots_rule_terminal=known_terminal,owner_state_before=old['owner_states'][a],
                    owner_state_after=t['decisions'][a]['context']['state']))
    summary=dict(series=16,first_divergences=len(cases),known_terminal_cases=sum(c['both_roots_rule_terminal'] for c in cases),
        owner_credit_improved=sum(c['owner_credit_after']>c['owner_credit_before'] for c in cases),
        owner_credit_worse=sum(c['owner_credit_after']<c['owner_credit_before'] for c in cases),
        owner_score_improved=sum(c['owner_score_after']<c['owner_score_before'] for c in cases),
        owner_score_worse=sum(c['owner_score_after']>c['owner_score_before'] for c in cases),
        group_credit_worse_without_owner_credit_loss=sum(c['npc_group_credit_after']<c['npc_group_credit_before'] and c['owner_credit_after']>=c['owner_credit_before'] for c in cases))
    summary['later_episode_pre_deadline_divergences']=sum(c['before_last_card'] for c in episode_divergences)
    result=dict(summary=summary,cases=cases,episode_divergences=episode_divergences,analysis='post-hoc mechanism diagnosis; aggregate NPC credit is not an individual NPC objective',
        limitations=['first divergence only, not all independent causal root effects',
                     'broad variant may also change subsequent controllers; certified terminal roots exclude that uncertainty'])
    (root/'owner_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();diagnose(a.root)
