"""Freeze source, audit 30 connection starts, run whole genre episodes."""
import argparse,hashlib,json,shutil,subprocess,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import Policy,character_key as owner_key,digest
from reflex.laboratory import profiles
from reflex.contingent_games import make_bridge
from tools.run_contingent_coverage import cases

def policy_for(c,r):return Policy(principle_priority='personal',commitments={owner_key(c):r})
def hashes():
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for directory in ('reflex','tools','tests') for p in sorted((ROOT/directory).glob('*.py'))}

def smoke(mahjong):
    rows=[]
    for case in cases('confirmation'):
        for p in profiles():
            for rigidity in (0.,.5,1.):
                if case['game']=='mahjong_closed':
                    if not mahjong:continue
                    from reflex.contingent_mahjong import engine,candidates,HandState,HandModel
                    env=engine().RiichiEnv(game_mode='4p-red-single',seed=1300);obs=env.reset()
                    seat=next(iter(obs));v,cs=candidates(obs[seat]);w=HandState(v,tuple(cs),0);b=HandModel(p,'conditional-mahjong');c=b.context(w)
                    d=policy_for(c,rigidity).choose(c,False);key=d['action_id']
                    assert key in {a['id'] for a in cs}
                elif case['game'].startswith('observed_'):
                    from reflex.observed_transfer import make_probe
                    b=make_probe(case['spec'],p);w=b.start();c=b.observe(w);d=policy_for(c,rigidity).choose(c,False);key=d['action_id']
                    w,_=b.advance(w,key,1300)
                else:
                    b=make_bridge(p,case);w=b.start();c=b.observe(b.project(w));d=policy_for(c,rigidity).choose(c,False);key=d['action_id']
                    w,_=b.actual(w,key,1300)
                assert next(a for a in c['actions'] if a['id']==key)['legal']
                rows.append(dict(case=case,profile=p['id'],rigidity=rigidity,root=key,legal_roots=len(c['actions']),owner=owner_key(c)))
    return rows

def worker(task):
    case,p,seed,rigidity=task;start=time.perf_counter()
    if case['game'].startswith('observed_'):
        from reflex.observed_transfer import run
        r=run(case['spec'],p,'observed',seed,horizon=2,rigidity=rigidity)
        r['case']=case;r['goal']=dict(target='adapter-purpose',status=r['status'],value=r['goal_value'])
    elif case['game']=='mahjong_closed':
        from reflex.contingent_mahjong import play
        r=play(p,seed,'goal-reflex',rigidity=rigidity)
    else:
        from reflex.contingent_trials import play
        b=make_bridge(p,case);c=b.observe(b.project(b.start()))
        r=play(case,p,seed,'goal-reflex',horizon=2,policy=policy_for(c,rigidity))
        w=b.start()
        for t in r['trace']:
            assert b.record(w)==t['before'];w,flow=b.actual(w,t['root'],seed)
            assert b.record(w)==t['after'] and flow==t['flow']
        assert b.terminal(w);r['action_replay_verified']=True
    r.update(rigidity=rigidity,elapsed_seconds=time.perf_counter()-start)
    return r

def main():
    a=argparse.ArgumentParser();a.add_argument('--output',required=True);a.add_argument('--frozen',action='store_true')
    a.add_argument('--workers',type=int,default=4);a.add_argument('--mahjong',action='store_true');args=a.parse_args()
    out=Path(args.output).resolve()
    if not args.frozen:
        out.mkdir(parents=True);source=out/'source';source.mkdir()
        for name in ('reflex','tools','tests'):shutil.copytree(ROOT/name,source/name,ignore=shutil.ignore_patterns('__pycache__'))
        cmd=[sys.executable,str(source/'tools/run_personal_commitment.py'),'--output',str(out),'--frozen','--workers',str(args.workers)]
        if args.mahjong:cmd.append('--mahjong')
        return subprocess.call(cmd,cwd=source)
    assert ROOT==out/'source' and 1<=args.workers<=8
    selected={'auction_contest','empire_race','duel','gambling','team_projects','team_combat','hagetaka',
        'observed_resources','observed_delivery','observed_auction','observed_combat'}
    if args.mahjong:selected.add('mahjong_closed')
    tasks=[(c,p,s,r) for c in cases('confirmation') if c['game'] in selected for p in profiles()
        for s in (1300,1301) for r in (0.,.5,1.)]
    fixed=hashes();pre=dict(format='personal-commitment-study-v1',jobs=tasks,source_hashes=fixed,
        baseline='rigidity0 = finite priority; .5 and1 fixed owner commitment',horizon=2,
        limitations=['30 starts are compatibility checks, not full-match strength','whole matches cover selected12 connections',
            'subjective-belief experiment is a separate authored rule set; not connected to every existing opponent reader',
            'not human-level, equal win rates, or psychological measurement'])
    (out/'preregister.json').write_text(json.dumps(pre,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    starts=smoke(args.mahjong);(out/'starts.json').write_text(json.dumps(starts,indent=2)+'\n',encoding='utf-8')
    print('connection starts',len(starts),'protocols',len({r['case']['game'] for r in starts}),flush=True)
    from reflex.commitment_experiment import pledge_episode
    with (out/'pledge.jsonl').open('w',encoding='utf-8') as f:
        for changed in (False,True):
            for seed in range(64):
                for rigidity in (0.,.5,1.):
                    for inertia in (0.,1.):f.write(json.dumps(pledge_episode(seed,rigidity,inertia,changed=changed))+'\n')
    runs=[];errors=[]
    with (out/'trajectories.jsonl').open('w',encoding='utf-8') as f,ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(worker,t):i for i,t in enumerate(tasks)}
        for future in as_completed(jobs):
            index=jobs[future]
            try:r=future.result()
            except Exception as exc:errors.append(dict(index=index,error=repr(exc)));print('ERROR',index,repr(exc),flush=True);continue
            r['job_index']=index;f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
            runs.append({k:v for k,v in r.items() if k!='trace'})
            print(len(runs),len(tasks),r['case']['game'],r['profile'],r['rigidity'],r.get('goal',r.get('point_delta')),flush=True)
    assert hashes()==fixed
    result=dict(format=pre['format'],planned=len(tasks),episodes=len(runs),errors=errors,runs=sorted(runs,key=lambda r:r['job_index']),
        preregister_digest=digest(pre),source_hashes=fixed,starts=len(starts),protocols=len({r['case']['game'] for r in starts}),
        pledge_episodes=768,default_changed=False,cloud_runtime_used=False)
    (out/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return 1 if errors else 0
if __name__=='__main__':raise SystemExit(main())
