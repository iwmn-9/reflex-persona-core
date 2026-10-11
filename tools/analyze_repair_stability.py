"""Whole frozen-episode reexecution plus independent actual transition ledger."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import hashlib,json,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import digest
from reflex.observed_transfer import make_probe
from reflex.laboratory import profiles
from reflex.plan_repair import edits
from tools.run_repair_stability import run_game,MODES
from tools.audit_observed_transfer import world,real_step


def identity(r):return digest([r['stage'],r['spec'],r['profile_index'],r['seed'],r['mode']])


def completed_receipts(root,rows,identify=identity,audit_filename='tools/analyze_repair_stability.py'):
    path=root/'_live_replay_receipts.json'
    while not path.exists():time.sleep(2)
    data=json.loads(path.read_text(encoding='utf-8'))
    assert data['complete'] and Path(data['root']).resolve()==root
    assert data['source_snapshot_sha256']==hashlib.sha256((root/'source_snapshot.json').read_bytes()).hexdigest()
    assert data['trajectories_sha256']==hashlib.sha256((root/'trajectories.jsonl').read_bytes()).hexdigest()
    assert data['audit_source_sha256']==hashlib.sha256((root/'_source'/audit_filename).read_bytes()).hexdigest()
    marker=json.loads((root/'_live_replay_requested.json').read_text())
    assert marker['helper_sha256']==data['helper_sha256'] and marker['audit_source_sha256']==data['audit_source_sha256']
    receipts={r['identity']:r for r in data['receipts']};actual={identify(r):r for r in rows}
    assert len(receipts)==len(data['receipts'])==len(actual)==len(rows) and set(receipts)==set(actual)
    for key,r in actual.items():
        assert receipts[key]['episode_sha256']==digest(r)
        c=receipts[key]['checks'];assert c['actual_turns']==c['full_replay_turns']==len(r['trace'])
        assert all(type(v) is int and v>=0 for v in c.values())
    return {k:r['checks'] for k,r in receipts.items()}


def audit(packet):
    r,job=packet;assert identity(r)==digest(job)
    p=make_probe(r['spec'],profiles()[r['profile_index']]);owner=None;previous=[];last=None
    counts=dict(actual_turns=0,full_replay_turns=0,guards=0,rejections=0,unprotected=0,selected_repairs=0,
        feedback_checks=0,feedback_rejections=0,feedback_unprotected=0,completion_calls=0,completion_settled=0)
    for t in r['trace']:
        before=world(r['spec']['genre'],t['before'])
        assert digest(asdict(real_step(before,t['root'],r['spec']['genre'],r['spec'],r['seed'])))==digest(t['after'])
        counts['actual_turns']+=1
        intention=t['intention'];scope=(intention['owner'],intention['preferences'],intention['target'],intention['unit'])
        if owner is None:owner=scope
        assert scope==owner;tick=p.observe(before)['tick'];assert intention['last_tick']==tick and (last is None or tick>last);last=tick
        a=t['deliberation'];m=a['metadata'];s=m['continuation_search'];guard=m.get('repair_stability')
        completed=m.get('completion_search')
        if completed:
            counts['completion_calls']+=1;counts['completion_settled']+=completed['all_model_endpoints_settled']
            assert r['mode'].startswith('complete-') and completed['max_horizon']==16
            assert completed['all_model_endpoints_settled']==(completed['levels'][-1]['running']==0)
        if guard:
            assert r['mode'] in ('guard','feedback','complete-feedback');counts['guards']+=1
            counts['rejections']+=not guard['accepted'];counts['unprotected']+=not guard['protected']
            if guard['protected']:assert guard['accepted']==all(v>=-1e-12 for v in guard['branch_deltas'])
            if not guard['accepted'] and not m.get('replan_check',{}).get('checked'):
                assert m['plans'][a['selected_plan']]['proposal']==guard['reference_proposal']
        check=m.get('replan_check')
        if check:
            assert r['mode'] in ('feedback','complete-feedback')
            counts['feedback_checks']+=check['checked'];counts['feedback_rejections']+=not check['accepted']
            counts['feedback_unprotected']+=check['checked'] and not check['protected']
            if check['checked']:
                assert check['seeds']==([960,961,962,963] if r['spec']['genre']=='combat' else [1])
                for branch in sum(check['branches'].values(),[]):
                    assert len(branch['feedback_replans'])<=1
                    for event in branch['feedback_replans']:assert event['audit']['target']==owner[2]
            if not check['accepted']:assert m['plans'][a['selected_plan']]['proposal']==check['reference_proposal']
        for name,row in s['proposals'].items():
            if name.startswith('proposal-repair-'):
                assert edits(previous[:6],[row['root']]+row['schedule'])==1
                for branch in row['branches']:
                    for pos,wanted in enumerate(row['schedule'],1):
                        if pos>=len(branch['actions']):break
                        if pos not in branch['schedule_fallbacks']:assert branch['actions'][pos]==wanted
        selected=a.get('selected_plan')
        if selected:counts['selected_repairs']+=m['plans'][selected]['proposal'].startswith('proposal-repair-')
        previous=intention['remaining']
    again=run_game(job);assert digest(again)==digest(r),'complete frozen numerical replay mismatch'
    counts['full_replay_turns']=len(again['trace']);return counts


def analyze(a):
    root=Path(a.root).resolve();snapshot=json.loads((root/'source_snapshot.json').read_text(encoding='utf-8'))
    for name,h in snapshot.items():assert hashlib.sha256((root/'_source'/name).read_bytes()).hexdigest()==h
    pre=json.loads((root/'preregister.json').read_text(encoding='utf-8'));jobs={digest(j):j for j in pre['jobs']}
    rows=[json.loads(x) for x in (root/'trajectories.jsonl').read_text(encoding='utf-8').splitlines()]
    actual={identity(r):r for r in rows};assert len(actual)==len(rows)==len(jobs) and set(actual)==set(jobs)
    assert json.loads((root/'completed.json').read_text(encoding='utf-8'))==dict(games=len(rows),complete=True)
    checks={}
    if (root/'_live_replay_requested.json').exists():checks=completed_receipts(root,rows)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            fs={pool.submit(audit,(r,jobs[k])):k for k,r in actual.items()}
            for count,f in enumerate(as_completed(fs),1):
                checks[fs[f]]=f.result()
                if count%12==0:print(f'{count}/{len(rows)} complete episodes independently verified',flush=True)
    groups=[];pairs=[]
    for genre in sorted({r['spec']['genre'] for r in rows}):
        arms={}
        for mode in pre.get('modes',MODES):
            rr=[r for r in rows if r['spec']['genre']==genre and r['mode']==mode]
            arms[mode]=dict(games=len(rr),mean_goal=sum(r['goal_value'] for r in rr)/len(rr),
                wins=sum(r['status']=='success' for r in rr),losses=sum(r['status']=='failure' for r in rr),
                **{k:sum(checks[identity(r)][k] for r in rr) for k in next(iter(checks.values()))})
        groups.append(dict(genre=genre,arms=arms))
    for r in rows:
        if r['mode']=='retained':continue
        b=actual[digest([r['stage'],r['spec'],r['profile_index'],r['seed'],'retained'])]
        pairs.append(dict(spec=r['spec'],profile=r['profile'],seed=r['seed'],mode=r['mode'],delta=r['goal_value']-b['goal_value'],
            survival_regression=b['surviving'] is True and r['surviving'] is False,
            shortages_delta=(r['shortages'] or 0)-(b['shortages'] or 0),route_loss_delta=(r['solvable_route_lost'] or 0)-(b['solvable_route_lost'] or 0),
            actions_equal=r['actions']==b['actions']))
    result=dict(format=pre['format'],games=len(rows),groups=groups,pairs=pairs,
        checks={k:sum(c[k] for c in checks.values()) for k in next(iter(checks.values()))},full_replay=True,
        source_snapshot_sha256=hashlib.sha256((root/'source_snapshot.json').read_bytes()).hexdigest(),
        trajectories_sha256=hashlib.sha256((root/'trajectories.jsonl').read_bytes()).hexdigest(),preregister_digest=digest(pre),default_changed=False)
    (root/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result['groups']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--frozen',action='store_true');a=p.parse_args()
    if a.frozen:analyze(a)
    else:
        import subprocess
        if (Path(a.root)/'_live_replay_requested.json').exists():analyze(a)
        else:subprocess.run([sys.executable,str(Path(a.root)/'_source/tools'/Path(__file__).name),'--root',a.root,'--workers',str(a.workers),'--frozen'],check=True)
