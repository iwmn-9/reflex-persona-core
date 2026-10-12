"""Compare a neutral peer model with publicly validated finite hypotheses."""
import argparse,copy,hashlib,json,shutil,subprocess,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import digest,VALUES
from reflex.integrated_trials import life
from reflex.public_responses import PublicResponses
from reflex.laboratory import profiles


def mixed(world):
    library=profiles()
    for i,peer in enumerate(world.actors):
        a,b=world.profiles[peer],library[(i+world.seed)%len(library)];weight=.35
        world.profiles[peer]=dict(id='mixed-'+a['id'],traits=tuple((1-weight)*x+weight*y for x,y in zip(a['traits'],b['traits'])),
            values={k:(1-weight)*a['values'].get(k,0.)+weight*b['values'].get(k,0.) for k in VALUES})


def worker(job):
    domain,layout,seed,variant,arm=job;start=time.perf_counter()
    factory=None if arm=='neutral' else lambda c:PublicResponses(c['scope'],profiles())
    r=life(seed,domain,layout,'owned',turns=36,horizon=2,peer_model_factory=factory,world_setup=mixed if variant=='mixed' else None)
    r.update(peer_arm=arm,variant=variant,elapsed_seconds=time.perf_counter()-start);return r


def hashes():
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for d in ('reflex','tools','tests') for p in sorted((ROOT/d).glob('*.py'))}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--seeds',type=int,default=2);ap.add_argument('--layouts',nargs='*',default=['balanced4','balanced8'])
    ap.add_argument('--variants',nargs='*',default=['known','mixed']);ap.add_argument('--frozen',action='store_true');a=ap.parse_args();out=Path(a.output).resolve()
    if not 1<=a.workers<=8 or not 1<=a.seeds<=8 or set(a.variants)-{'known','mixed'} or set(a.layouts)-{'balanced4','balanced8'}:raise ValueError('bounded social cohort required')
    if not a.frozen:
        out.mkdir(parents=True,exist_ok=False);source=out/'source';source.mkdir()
        for d in ('reflex','tools','tests'):shutil.copytree(ROOT/d,source/d,ignore=shutil.ignore_patterns('__pycache__'))
        return subprocess.call([sys.executable,str(source/'tools'/Path(__file__).name),*sys.argv[1:],'--frozen'],cwd=source)
    assert ROOT==out/'source';fixed=hashes();jobs=[(d,l,s,v,arm) for d in ('exchange','watch') for l in a.layouts for v in a.variants for s in range(4400,4400+a.seeds) for arm in ('neutral','public')]
    pre=dict(format='public-peer-reliability-v1',source_hashes=fixed,jobs=jobs,
        hypotheses=['publicly learned response distributions improve prediction and useful social decisions beyond fixed neutral peers',
            'benefit must survive actual peers whose traits/values mix the declared hypothesis profiles'],
        controls='same real personas, own social lifecycle, 36ticks/H2/two model seeds; only projected peer response model changes',
        limitations=['shared world NPCs are not independent experiments','profile hypotheses cannot see true peer traits or private memories',
            'prequential prediction gate does not prove decision utility; real score/health/stalling compared separately',
            'modeled peer belief remains a frozen copy between real observations; no secret or hypothetical learning updates real belief'],cloud_runtime_used=False)
    (out/'preregister.json').write_text(json.dumps(pre,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');runs=[];errors=[];start=time.perf_counter()
    with (out/'trajectories.jsonl').open('x',encoding='utf-8') as stream,ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs={pool.submit(worker,j):i for i,j in enumerate(jobs)}
        for f in as_completed(fs):
            i=fs[f]
            try:r=f.result()
            except Exception as e:errors.append(dict(index=i,job=jobs[i],error=repr(e)));print('ERROR',i,repr(e),flush=True);continue
            r['index']=i;stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush();slim={k:v for k,v in r.items() if k not in ('trace','peer_models')}
            updates=[u for t in r['trace'] for u in t.get('peer_updates',[]) if u['score']['scored']]
            slim.update(prediction_trials=len(updates),prediction_gain=sum(u['score']['gain'] for u in updates)/len(updates) if updates else None,
                predictive_adoptions=sum(u['active'] for u in updates),rest_choices=sum(k=='rest' for t in r['trace'] for k in t['chosen'].values()))
            runs.append(slim);print(len(runs)+len(errors),'/',len(jobs),jobs[i],round(r['elapsed_seconds'],1),flush=True)
    assert hashes()==fixed;groups={}
    for r in runs:groups.setdefault(digest([r['domain'],r['layout'],r['seed'],r['variant']]),{})[r['peer_arm']]=r
    paired=[]
    for g in groups.values():
        if set(g)!={'neutral','public'}:continue
        a,b=g['neutral'],g['public'];by={r['npc']:r for r in b['actors']}
        for own in a['actors']:
            other=by[own['npc']];paired.append(dict(domain=a['domain'],layout=a['layout'],seed=a['seed'],variant=a['variant'],npc=own['npc'],profile=own['profile'],
                baseline=own['goal']['value'],candidate=other['goal']['value'],delta=other['goal']['value']-own['goal']['value'],
                baseline_health=own['health'],candidate_health=other['health']))
    result=dict(format=pre['format'],planned=len(jobs),worlds=len(runs),errors=errors,runs=runs,paired=paired,source_hashes=fixed,
        preregister_sha256=digest(pre),elapsed_seconds=time.perf_counter()-start)
    (out/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');return int(bool(errors))


if __name__=='__main__':raise SystemExit(main())
