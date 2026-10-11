"""Independent real-rule ledgers plus complete frozen-source episode replay."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import digest
from reflex.plan_repair import edits
from reflex.laboratory import profiles
from tools.run_plan_repair import run_game,MODES
from tools.audit_observed_transfer import world,real_step


def identity(r):return digest([r['stage'],r['spec'],r['profile_index'],r['seed'],r['mode']])


def completed_receipts(root,rows,path):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    assert data['complete'] is True and Path(data['root']).resolve()==root
    assert data['source_snapshot_sha256']==hashlib.sha256((root/'source_snapshot.json').read_bytes()).hexdigest()
    assert data['trajectories_sha256']==hashlib.sha256((root/'trajectories.jsonl').read_bytes()).hexdigest()
    expected_source=hashlib.sha256((root/'_source/tools/analyze_plan_repair.py').read_bytes()).hexdigest()
    assert data['audit_source_sha256']==expected_source
    receipts={r['identity']:r for r in data['receipts']};actual={identity(r):r for r in rows}
    assert len(receipts)==len(data['receipts'])==len(rows) and set(receipts)==set(actual)
    output={}
    for key,r in actual.items():
        receipt=receipts[key];assert receipt['episode_sha256']==digest(r)
        c=receipt['checks'];assert c['actual_turns']==c['full_episode_replay_turns']==len(r['trace'])
        assert all(type(v) is int and v>=0 for v in c.values())
        output[key]=c
    return output


def audit(packet):
    r,job=packet
    assert identity(r)==digest(job)
    independent=0;target=None;previous_remaining=[];previous_tick=None
    changes=0;patches=0;fresh=0;held=0;proposals=0;fallbacks=0;root_changed=0
    for t in r['trace']:
        before=world(r['spec']['genre'],t['before'])
        after=real_step(before,t['root'],r['spec']['genre'],r['spec'],r['seed'])
        assert digest(asdict(after))==digest(t['after']);independent+=1
        metadata=t['deliberation']['metadata'];search=metadata['continuation_search']
        repair=search.get('repair');record=t.get('intention')
        assert bool(repair)==(r['mode']=='repair')
        if record:
            if target is None:target=record['target']
            assert record['target']==target and record['unit']=='public-turns'
            assert record['last_tick']==before.tick
            assert previous_tick is None or before.tick>previous_tick
            previous_tick=before.tick
            if previous_remaining:root_changed+=t['root']!=previous_remaining[0]
            if repair:assert repair['retained']==previous_remaining[:6]
            previous_remaining=record['remaining']
        if repair:
            rs=[p for k,p in search['proposals'].items() if k.startswith('proposal-repair-')]
            assert len(rs)==repair['proposed']<=repair['limit']==24;proposals+=len(rs)
            for p in rs:
                sequence=[p['root']]+p['schedule']
                assert edits(repair['retained'],sequence)==1
                for branch in p['branches']:
                    assert branch['actions'][0]==p['root']
                    # Every scripted step is identical across branches; only
                    # public unavailability permits the existing feedback fallback.
                    for pos,wanted in enumerate(sequence[1:],1):
                        if pos>=len(branch['actions']):break
                        if pos not in branch['schedule_fallbacks']:assert branch['actions'][pos]==wanted
                    fallbacks+=len(branch['schedule_fallbacks'])
        selected=t['deliberation'].get('selected_plan')
        if selected:
            proposal=metadata['plans'][selected]['proposal']
            patches+=proposal.startswith('proposal-repair-')
            held+=proposal=='proposal-retained'
            fresh+=not(proposal.startswith('proposal-repair-') or proposal=='proposal-retained')
    # Includes fresh proposal discovery and all owner/need/pressure updates,
    # not merely the selected path or the terminal outcome.
    again=run_game(job)
    assert digest(again)==digest(r),'full frozen episode must reproduce every record'
    return dict(actual_turns=independent,full_episode_replay_turns=len(again['trace']),
        patch_proposals=proposals,selected_patches=patches,selected_retained=held,selected_fresh=fresh,
        actual_head_deviations=root_changed,repair_branch_fallbacks=fallbacks)


def analyze(a):
    root=Path(a.root).resolve();snapshot=json.loads((root/'source_snapshot.json').read_text(encoding='utf-8'))
    for name,h in snapshot.items():assert hashlib.sha256((root/'_source'/name).read_bytes()).hexdigest()==h
    pre=json.loads((root/'preregister.json').read_text(encoding='utf-8'))
    rows=[json.loads(line) for line in (root/'trajectories.jsonl').read_text(encoding='utf-8').splitlines()]
    assert json.loads((root/'completed.json').read_text(encoding='utf-8'))==dict(games=len(rows),complete=True)
    expected={digest(j):j for j in pre['jobs']};actual={identity(r):r for r in rows}
    assert len(expected)==len(actual)==len(rows) and set(expected)==set(actual)
    receipts={}
    if a.replay_receipts:
        receipts=completed_receipts(root,rows,a.replay_receipts)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures={pool.submit(audit,(r,expected[k])):k for k,r in actual.items()}
            for count,f in enumerate(as_completed(futures),1):
                receipts[futures[f]]=f.result()
                if count%12==0:print(f'{count}/{len(rows)} full frozen episodes and independent worlds verified',flush=True)
    checks={k:sum(v[k] for v in receipts.values()) for k in next(iter(receipts.values()))}
    groups=[];pairs=[]
    for genre in sorted({r['spec']['genre'] for r in rows}):
        arms={}
        for mode in MODES:
            rr=[r for r in rows if r['spec']['genre']==genre and r['mode']==mode]
            audit_rows=[receipts[identity(r)] for r in rr]
            arms[mode]=dict(games=len(rr),mean_goal=sum(r['goal_value'] for r in rr)/len(rr),
                successes=sum(r['status']=='success' for r in rr),failures=sum(r['status']=='failure' for r in rr),
                surviving=sum(r['surviving'] is True for r in rr),shortages=sum(r['shortages'] or 0 for r in rr),
                reachable_route_lost=sum(r['solvable_route_lost'] or 0 for r in rr),
                actions=sum(r['steps'] for r in rr),**{k:sum(x[k] for x in audit_rows) for k in checks if k not in ('actual_turns','full_episode_replay_turns')})
        groups.append(dict(genre=genre,arms=arms))
    for r in rows:
        if r['mode']!='repair':continue
        for reference in ('fresh','retained'):
            job=(r['stage'],r['spec'],r['profile_index'],r['seed'],reference);b=actual[digest(job)]
            pairs.append(dict(spec=r['spec'],profile=r['profile'],seed=r['seed'],reference=reference,
                delta=r['goal_value']-b['goal_value'],baseline_status=b['status'],candidate_status=r['status'],
                survival_regression=b['surviving'] is True and r['surviving'] is False,
                shortages_delta=(r['shortages'] or 0)-(b['shortages'] or 0),
                reachable_route_lost_delta=(r['solvable_route_lost'] or 0)-(b['solvable_route_lost'] or 0)))
    result=dict(format='plan-repair-v1',games=len(rows),groups=groups,pairs=pairs,checks=checks,
        trajectories_sha256=hashlib.sha256((root/'trajectories.jsonl').read_bytes()).hexdigest(),
        source_snapshot_sha256=hashlib.sha256((root/'source_snapshot.json').read_bytes()).hexdigest(),
        preregister_digest=digest(pre),full_replay=True,default_changed=False,
        limits=pre['limits']+['fixed authored conditions; no confidence or human-level claim','more repairs do not imply better plans'])
    (root/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(games=len(rows),checks=checks,groups=groups),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--workers',type=int,default=2);p.add_argument('--frozen',action='store_true');p.add_argument('--replay-receipts');a=p.parse_args()
    if not 1<=a.workers<=8:p.error('bounded workers required')
    marker=Path(a.root).resolve()/'_live_replay_requested.json'
    if not a.frozen and not a.replay_receipts and marker.exists():
        requested=json.loads(marker.read_text(encoding='utf-8'));study=Path(a.root).resolve()
        assert Path(requested['root']).resolve()==study
        assert requested['source_snapshot_sha256']==hashlib.sha256((study/'source_snapshot.json').read_bytes()).hexdigest()
        target=Path(requested['receipt']).resolve();assert target.parent==study and target.name=='_live_replay_receipts.json'
        while not target.exists():time.sleep(.5)
        a.replay_receipts=str(target)
    if a.frozen or a.replay_receipts:analyze(a)
    else:
        subprocess.run([sys.executable,str(Path(a.root).resolve()/'_source/tools/analyze_plan_repair.py'),
            '--root',str(Path(a.root).resolve()),'--workers',str(a.workers),'--frozen'],check=True)
