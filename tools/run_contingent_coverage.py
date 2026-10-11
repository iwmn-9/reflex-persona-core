"""Freeze then run every existing mechanic family; preserve losses and gaps."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reflex.core import digest
from reflex.laboratory import profiles

def source_hashes():
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for directory in ('reflex','tools','tests') for p in sorted((ROOT/directory).glob('*.py'))}


def cases(stage):
    changed=stage=='confirmation'
    return [
        dict(game='navigation',variant=int(changed)),dict(game='duel',variant=int(changed)),
        dict(game='project',variant=int(changed)),dict(game='connect_four'),
        dict(game='reversi',opening=17 if changed else 0),
        dict(game='quoridor',players=2 if changed else 4),
        dict(game='no_thanks',players=5 if changed else 4,deal=17 if changed else 0),
        dict(game='goofspiel',players=5 if changed else 4,cards=7,order='descending' if changed else 'ascending'),
        dict(game='hagetaka',players=5 if changed else 4,deal=17 if changed else 0),
        dict(game='auction_contest',players=3 if changed else 4,deal=17 if changed else 0),
        dict(game='gambling',scenario='change' if changed else 'opportunity'),
        dict(game='team_projects',scenario='exhausted' if changed else 'balanced',members=2 if changed else 3),
        dict(game='team_combat',map='choke' if changed else 'cover',goal='both' if changed else 'either'),
        dict(game='laboratory',scenario='workshop' if changed else 'ordinary'),
        dict(game='observed_resources',spec=dict(genre='resources',route='mixed',limit=13 if changed else 12,famine=changed)),
        dict(game='observed_delivery',spec=dict(genre='delivery',limit=9 if changed else 8,capacity=3 if changed else 2,supply=2 if changed else 3)),
        dict(game='observed_auction',spec=dict(genre='auction',prizes=[11,5,14] if changed else [6,13,9],budget=10 if changed else 9,rival_shift=True)),
        dict(game='observed_combat',spec=dict(genre='combat',hp=4 if changed else 5,limit=11 if changed else 10,distance=4)),
        dict(game='mahjong_closed'),
        dict(game='encounter_exchange',layout='balanced8' if changed else 'balanced4',contact='public'),
        dict(game='encounter_watch',layout='balanced8' if changed else 'balanced4',contact='public'),
        dict(game='social_allocation',schedule='bad_luck' if changed else 'change_and_repair'),
        dict(game='social_patrol',schedule='bad_luck' if changed else 'change_and_repair'),
        dict(game='empire_race',players=3 if changed else 4,change=changed),
        dict(game='cyclic_arena',phase_offset=10 if changed else 0),
        dict(game='shared_commons',phase_offset=10 if changed else 0),
        dict(game='branching_delivery',phase_offset=10 if changed else 0),
        dict(game='tiny_resources'),dict(game='tiny_arena'),
        dict(game='public_graph',graph=12102 if changed else 12100,length=12 if changed else 5)]


def worker(task):
    case,p,seed,method,horizon,revisions,budget,validate=task;start=time.perf_counter()
    if case['game'].startswith('observed_'):
        from reflex.observed_transfer import run
        r=run(case['spec'],p,'observed',seed,horizon=horizon,search=method=='conditional',
            width=1,depth=revisions,continuity=method=='conditional',contingent=method=='conditional',
            patch_budget=budget,settlement_weight='absolute')
        r.update(case=case,method=method,goal=dict(target='adapter-purpose',status=r['status'],value=r['goal_value']))
    elif case['game']=='mahjong_closed':
        from reflex.contingent_mahjong import play
        r=play(p,seed,method)
    else:
        from reflex.contingent_trials import play
        r=play(case,p,seed,method,horizon,revisions,budget,(501,502,503,504) if validate else None)
    r['elapsed_seconds']=time.perf_counter()-start
    # Frozen action-stream replay verifies every driver ledger. Independent
    # coordinate/conservation referees are additionally inside applicable games.
    if not case['game'].startswith('observed_') and case['game']!='mahjong_closed':
        from reflex.contingent_games import make_bridge
        b=make_bridge(p,case);w=b.start()
        for row in r['trace']:
            assert b.record(w)==row['before'];w,flow=b.actual(w,row['root'],seed)
            assert b.record(w)==row['after'] and flow==row['flow'] and b.goal(w).record()==row['goal']
        assert b.terminal(w)
        r['action_replay_verified']=True
    return r


def summarize(runs):
    rows=[]
    for game in sorted({r['case']['game'] for r in runs}):
        for profile in sorted({r['profile'] for r in runs}):
            paired=[]
            for seed in sorted({r['seed'] for r in runs if r['case']['game']==game}):
                group={r['method']:r for r in runs if r['case']['game']==game and r['profile']==profile and r['seed']==seed}
                if len(group)!=2:continue
                a=group['goal-reflex'];b=group['conditional']
                av=a.get('goal',{}).get('value',a.get('point_delta'));bv=b.get('goal',{}).get('value',b.get('point_delta'))
                paired.append(dict(seed=seed,baseline=av,conditional=bv,delta=bv-av,
                    baseline_steps=a['steps'],conditional_steps=b['steps'],actions_changed=a.get('actions')!=b.get('actions') if 'actions' in a else None))
            rows.append(dict(game=game,profile=profile,pairs=len(paired),better=sum(p['delta']>1e-12 for p in paired),
                worse=sum(p['delta']<-1e-12 for p in paired),same=sum(abs(p['delta'])<=1e-12 for p in paired),paired=paired))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--stage',choices=('development','confirmation'),default='development')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--seeds',type=int,default=1);p.add_argument('--horizon',type=int,default=3)
    p.add_argument('--revisions',type=int,default=2);p.add_argument('--patch-budget',type=int,default=4);p.add_argument('--frozen',action='store_true')
    p.add_argument('--validate',action='store_true')
    p.add_argument('--games',nargs='*');args=p.parse_args();out=Path(args.output).resolve()
    if not 1<=args.workers<=8 or not 1<=args.seeds<=8:raise ValueError('bounded independent CPU workers/seeds required')
    if not args.frozen:
        if out.exists():raise FileExistsError('new frozen evidence directory required')
        out.mkdir(parents=True);snapshot=out/'source';snapshot.mkdir()
        for name in ('reflex','tools','tests'):shutil.copytree(ROOT/name,snapshot/name,ignore=shutil.ignore_patterns('__pycache__'))
        arguments=list(sys.argv[1:])
        for i,arg in enumerate(arguments):
            if arg=='--output':arguments[i+1]=str(out);break
            if arg.startswith('--output='):arguments[i]='--output='+str(out);break
        command=[sys.executable,str(snapshot/'tools'/Path(__file__).name)]+arguments+['--frozen']
        return subprocess.call(command,cwd=snapshot)
    if ROOT!=out/'source':raise ValueError('execute the frozen source, not the live checkout')
    selected=[c for c in cases(args.stage) if not args.games or c['game'] in args.games]
    if args.games and set(args.games)-{c['game'] for c in selected}:raise ValueError('unknown game coverage request')
    if args.validate and any(c['game'].startswith('observed_') or c['game']=='mahjong_closed' for c in selected):
        raise ValueError('independent bank comparison currently requires generic full-game bridges')
    tasks=[(c,profile,seed,method,args.horizon,args.revisions,args.patch_budget,args.validate)
        for c in selected for profile in profiles() for seed in range(1200 if args.stage=='development' else 1250,(1200 if args.stage=='development' else 1250)+args.seeds)
        for method in ('goal-reflex','conditional')]
    fixed=source_hashes();pre=dict(format='conditional-coverage-v1',stage=args.stage,jobs=tasks,source_hashes=fixed,
        horizon=args.horizon,width=1,revisions=args.revisions,patch_budget=args.patch_budget,validation_seeds=[501,502,503,504] if args.validate else None,
        baseline='same game-side model and endpoint/persona evaluator; future unchanged persona reflex',
        condition='bounded sparse observation-history patches; execute one root; replan on actual perception',
        status='existing authored games/adapters; exploratory integration and endpoint comparison, not human-level or unseen-game proof',
        limitations=['finite model bank; no calibrated confidence','one owner; no new joint conditional agreement',
            'mahjong conditional continuation is own hand only','game-side proxies/models are explicit, not learned universal scores'],
        hypothesis='observation-conditioned programs can improve choice without latent-world access; regressions retained; defaults remain unchanged')
    with (out/'preregister.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(pre,f,ensure_ascii=False,indent=2)
    runs=[];errors=[];start=time.perf_counter()
    with (out/'trajectories.jsonl').open('x',encoding='utf-8',newline='\n') as stream,ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(worker,t):i for i,t in enumerate(tasks)}
        for future in as_completed(futures):
            i=futures[future]
            try:r=future.result()
            except Exception as exc:
                errors.append(dict(index=i,task=tasks[i],error=repr(exc)));print('ERROR',i,tasks[i][0],repr(exc),flush=True);continue
            r['job_index']=i;stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush()
            runs.append({k:v for k,v in r.items() if k!='trace'})
            print(f"{len(runs)+len(errors)}/{len(tasks)} {r['case']['game']} {r['profile']} {r['method']} {r.get('goal',r.get('point_delta'))} {r['elapsed_seconds']:.1f}s",flush=True)
    assert source_hashes()==fixed
    result=dict(format=pre['format'],episodes=len(runs),planned=len(tasks),errors=errors,summary=summarize(runs),runs=sorted(runs,key=lambda r:r['job_index']),
        elapsed_seconds=time.perf_counter()-start,source_hashes=fixed,preregister_digest=digest(pre),default_policy_changed=False,cloud_runtime_used=False)
    with (out/'evaluation.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    return 1 if errors else 0


if __name__=='__main__':raise SystemExit(main())
