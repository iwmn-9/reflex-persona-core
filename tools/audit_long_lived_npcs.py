"""Replay public transitions, owned social memory, and experience probes."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import copy
import gzip
import hashlib
import json
import tempfile

from reflex.core import digest
from reflex.encounters import EncounterWorld
from reflex.encounter_experiment import interaction_counts
from reflex.social import SocialBinding
from reflex.social_withdrawal import WithdrawalSocialMemory
from tools.long_lived_npcs import comparison_outcome, run_one, summarize


def audit_world(job):
    root,run=job;raw=gzip.decompress((Path(root)/run['trajectory']).read_bytes())
    assert hashlib.sha256(raw).hexdigest()==run['content_sha256']
    saved=json.loads(raw);trace=saved['trace'];r=saved['run']
    assert r=={k:v for k,v in run.items() if k not in ('trajectory','content_sha256')}
    w=EncounterWorld(r['seed'],r['domain'],r['layout'],r['contact']);initial,_=w.contexts()
    axes=digest([dict(personality=c['personality'],values=c['values']) for c in initial]);assert axes==r['axes_digest']
    memory={a:WithdrawalSocialMemory(c) for a,c in zip(w.actors,initial)}
    probes=defaultdict(list)
    for p in saved['probes']:probes[p['tick']].append(p)
    counts=Counter();harm={};examples=[];profile_actions=defaultdict(Counter);need_changes=Counter();last_need={}
    for t in trace:
        assert t['tick']==w.tick and t['fixed_personality']
        joint={row['npc']:row['action'] for row in t['rows']}
        for p in probes[w.tick]:
            actual=comparison_outcome(w,p['npc'],p['chosen'],joint)
            other=comparison_outcome(w,p['npc'],p['without_experience'],joint)
            assert abs(actual['score']-other['score']-p['gain_delta'])<1e-12
            assert abs(actual['health']-other['health']-p['health_delta'])<1e-12
            assert p['chosen']==joint[p['npc']]
            counts['same_state_experience_probes']+=1
        for row in t['rows']:
            own=row['npc'];target=row['target'];assert row['before']==w.state(own)
            if row['selected_appraisal'] is not None:
                b=SocialBinding(target,1 if row['kind']=='cooperate' else -1,max(0,1-w.health[target]/8))
                _,audit=memory[own].scores((row['action'],),(target,),{row['action']:b},w.tick)
                assert audit[row['action']]==row['selected_appraisal'];counts['social_appraisals']+=1
            else:assert r['arm'] not in ('social','net_effects') or target is None
            profile_actions[row['profile']][row['kind']]+=1
            if own in last_need and last_need[own]!=row['primary_need'] and row['kind']!='finished':need_changes[row['profile']]+=1
            last_need[own]=row['primary_need']
            if target is not None:
                pair=(own,target)
                if row['kind']=='cooperate' and row['other_kind']=='claim':harm[pair]=dict(tick=w.tick,row=row)
                if pair in harm and row['kind']==row['other_kind']=='cooperate' and not row['luck_failure']:
                    if len(examples)<4:
                        examples.append(dict(npc=own,profile=row['profile'],target=target,
                            harm_tick=harm[pair]['tick'],reconnection_tick=w.tick,
                            before_harm=harm[pair]['row']['before'],after_reconnection=row['after'],
                            interval_actions=[dict(tick=z['tick'],action=z['rows'][w.actors.index(own)]['action'])
                                for z in trace[max(0,harm[pair]['tick']-1):w.tick+1] if z['tick']<=harm[pair]['tick']+3 or z['tick']>=w.tick-3]))
                    del harm[pair]
        actual,events=w.resolve(joint)
        for a,b in zip(actual,t['rows']):
            assert a=={k:b[k] for k in a},(run['trajectory'],t['tick'],a['npc'])
        if r['arm']!='amnesic':
            for own,rows in zip(w.actors,events):
                for event in rows:memory[own].observe(event)
        counts['actor_ticks']+=len(actual);counts['actual_choices']+=sum(a['kind']!='finished' for a in actual)
    for a in r['actors']:
        assert w.state(a['npc'])=={k:a[k] for k in ('energy','stock','health','score')}
    interaction,_=interaction_counts(trace,r['contact']);assert interaction==r['interaction']
    assert len(trace)==r['turns']
    return dict(trajectory=run['trajectory'],counts=dict(counts),examples=examples,
        profile_actions={k:dict(v) for k,v in profile_actions.items()},need_changes=dict(need_changes))


def audit(root,workers=4,replay=True):
    root=Path(root);result=json.loads((root/'results.json').read_text(encoding='utf-8'))
    assert json.loads((root/'registration.json').read_text())==result['plan']
    repo=Path(__file__).resolve().parents[1]
    for name,expected in result['plan']['sources'].items():
        assert hashlib.sha256((root/'sources'/name).read_bytes()).hexdigest()==expected
        if replay:assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==expected
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows=[]
        for i,row in enumerate(pool.map(audit_world,[(str(root),r) for r in result['runs']])):
            rows.append(row)
            if (i+1)%48==0:print(f'replayed public rules and memories: {i+1}/{len(result["runs"])}',flush=True)
    assert summarize(result['runs'])==result['summary']
    decision_replays=0
    if replay:
        chosen=[r for r in result['runs'] if r['seed']==result['plan']['seeds'][0] and r['arm']=='social']
        with tempfile.TemporaryDirectory() as tmp:
            jobs=[(tmp,r['seed'],r['domain'],r['layout'],r['contact'],r['arm'],r['turns']) for r in chosen]
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for new,old in zip(pool.map(run_one,jobs),chosen):
                    assert new==old;decision_replays+=1
    totals=Counter()
    for row in rows:totals.update(row['counts'])
    proof=dict(worlds=len(rows),counts=dict(totals),whole_policy_world_replays=decision_replays,
        frozen_sources_verified=True,actual_transition_and_memory_replay=True,
        no_human_playtesting=True,runs=rows)
    (root/'audit.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in proof.items() if k!='runs'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--no-policy-replay',action='store_true');a=p.parse_args()
    audit(a.root,a.workers,not a.no_policy_replay)
