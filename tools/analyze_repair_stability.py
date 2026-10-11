"""Whole frozen-episode reexecution plus independent actual transition ledger."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import hashlib,json
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


def audit(packet):
    r,job=packet;assert identity(r)==digest(job)
    p=make_probe(r['spec'],profiles()[r['profile_index']]);owner=None;previous=[];last=None
    counts=dict(actual_turns=0,full_replay_turns=0,guards=0,rejections=0,unprotected=0,selected_repairs=0)
    for t in r['trace']:
        before=world(r['spec']['genre'],t['before'])
        assert digest(asdict(real_step(before,t['root'],r['spec']['genre'],r['spec'],r['seed'])))==digest(t['after'])
        counts['actual_turns']+=1
        intention=t['intention'];scope=(intention['owner'],intention['preferences'],intention['target'],intention['unit'])
        if owner is None:owner=scope
        assert scope==owner;tick=p.observe(before)['tick'];assert intention['last_tick']==tick and (last is None or tick>last);last=tick
        a=t['deliberation'];m=a['metadata'];s=m['continuation_search'];guard=m.get('repair_stability')
        if guard:
            assert r['mode']=='guard';counts['guards']+=1
            counts['rejections']+=not guard['accepted'];counts['unprotected']+=not guard['protected']
            if guard['protected']:assert guard['accepted']==all(v>=-1e-12 for v in guard['branch_deltas'])
            if not guard['accepted']:
                assert m['plans'][a['selected_plan']]['proposal']==guard['reference_proposal']
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
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs={pool.submit(audit,(r,jobs[k])):k for k,r in actual.items()}
        for count,f in enumerate(as_completed(fs),1):
            checks[fs[f]]=f.result()
            if count%12==0:print(f'{count}/{len(rows)} complete episodes independently verified',flush=True)
    groups=[];pairs=[]
    for genre in sorted({r['spec']['genre'] for r in rows}):
        arms={}
        for mode in MODES:
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
        subprocess.run([sys.executable,str(Path(a.root)/'_source/tools'/Path(__file__).name),'--root',a.root,'--workers',str(a.workers),'--frozen'],check=True)
