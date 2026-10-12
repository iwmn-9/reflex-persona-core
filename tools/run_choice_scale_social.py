"""Factorially isolate factual gain scoring and public peer prediction."""
import argparse,hashlib,json,shutil,subprocess,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.integrated_trials import life
from reflex.public_responses import PublicResponses
from reflex.laboratory import profiles
from reflex.core import digest
from tools.run_public_peer_reliability import mixed
ARMS={'neutral_level':(False,False),'neutral_gain':(False,True),'public_level':(True,False),'public_gain':(True,True)}
def hashes():return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for d in ('reflex','tools','tests') for p in sorted((ROOT/d).glob('*.py'))}
def worker(job):
    domain,layout,seed,variant,arm=job;public,gain=ARMS[arm];start=time.perf_counter()
    factory=(lambda c:PublicResponses(c['scope'],profiles())) if public else None
    r=life(seed,domain,layout,'owned',turns=36,horizon=2,peer_model_factory=factory,
        world_setup=mixed if variant=='mixed' else None,objective_gain=gain)
    r.update(peer_arm=arm,variant=variant,elapsed_seconds=time.perf_counter()-start);return r
def paired_worker(job):
    domain,layout,seed,variant,public=job;cache={};runs=[]
    factory=(lambda c:PublicResponses(c['scope'],profiles())) if public else None
    for gain in (False,True):
        start=time.perf_counter();arm=('public_' if public else 'neutral_')+('gain' if gain else 'level')
        try:r=life(seed,domain,layout,'owned',turns=36,horizon=2,peer_model_factory=factory,
            world_setup=mixed if variant=='mixed' else None,objective_gain=gain,forecast_cache=cache)
        except Exception as e:
            runs.append(dict(error=repr(e),job=(domain,layout,seed,variant,arm)));continue
        r.update(peer_arm=arm,variant=variant,elapsed_seconds=time.perf_counter()-start);runs.append(r)
    return runs
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--seeds',type=int,default=2)
    ap.add_argument('--workers',type=int,default=2);ap.add_argument('--paired-cache',action='store_true');ap.add_argument('--frozen',action='store_true');a=ap.parse_args();out=Path(a.output).resolve()
    if not 1<=a.seeds<=4 or not 1<=a.workers<=4:raise ValueError('bounded declared social comparison required')
    if not a.frozen:
        out.mkdir(parents=True,exist_ok=False);source=out/'source';source.mkdir()
        for d in ('reflex','tools','tests'):shutil.copytree(ROOT/d,source/d,ignore=shutil.ignore_patterns('__pycache__'))
        command=[sys.executable,str(source/'tools'/Path(__file__).name),'--output',str(out),'--seeds',str(a.seeds),'--workers',str(a.workers),'--frozen']
        if a.paired_cache:command.append('--paired-cache')
        return subprocess.call(command,cwd=source)
    assert ROOT==out/'source';fixed=hashes();jobs=[(d,l,s,v,arm) for d in ('exchange','watch') for l in ('balanced4','balanced8') for s in range(7200,7200+a.seeds) for v in ('known','mixed') for arm in ARMS]
    pre=dict(format='choice-scale-social-v1',source_hashes=fixed,jobs=jobs,arms=ARMS,
        hypothesis='gain-origin scoring converts improved public peer knowledge into useful resource/value choices better than absolute stock evaluation',
        controls='same true profile/world seed/36 ticks/H2/two model seeds; current own score is known, peer private traits/memories/actions remain unavailable',
        paired_rollout_cache=a.paired_cache,
        limitations=['shared-world NPCs correlated','only two actual seeds by default','resource and survival goal is not win rate or whole personality utility'])
    (out/'preregister.json').write_text(json.dumps(pre,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');runs=[];errors=[];start=time.perf_counter()
    with (out/'trajectories.jsonl').open('x',encoding='utf-8') as stream,ProcessPoolExecutor(max_workers=a.workers) as pool:
        tasks=[(d,l,s,v,p) for d in ('exchange','watch') for l in ('balanced4','balanced8') for s in range(7200,7200+a.seeds) for v in ('known','mixed') for p in (False,True)] if a.paired_cache else jobs
        fs={pool.submit(paired_worker if a.paired_cache else worker,j):i for i,j in enumerate(tasks)}
        for f in as_completed(fs):
            i=fs[f]
            try:result=f.result()
            except Exception as e:
                failed=[i]
                # Pair task order is domain/layout/seed/variant/public. Lookup
                # actual declared world indices instead of arithmetic offsets.
                if a.paired_cache:
                    d,l,s,v,p=tasks[i];failed=[jobs.index((d,l,s,v,('public_' if p else 'neutral_')+suffix)) for suffix in ('level','gain')]
                errors.extend(dict(index=k,job=jobs[k],error=repr(e)) for k in failed);print('ERROR',failed,repr(e),flush=True);continue
            for r in (result if a.paired_cache else [result]):
                if 'error' in r:
                    k=jobs.index(tuple(r['job']));errors.append(dict(index=k,job=jobs[k],error=r['error']));print('ERROR',k,r['error'],flush=True);continue
                r['index']=jobs.index((r['domain'],r['layout'],r['seed'],r['variant'],r['peer_arm']))
                stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush()
                runs.append({k:v for k,v in r.items() if k!='trace'});print(len(runs)+len(errors),'/',len(jobs),r['domain'],r['layout'],r['variant'],r['seed'],r['peer_arm'],round(r['elapsed_seconds'],1),flush=True)
    assert hashes()==fixed
    (out/'evaluation.json').write_text(json.dumps(dict(format=pre['format'],planned=len(jobs),worlds=len(runs),errors=errors,runs=runs,
        source_hashes=fixed,preregister_sha256=digest(pre),elapsed_seconds=time.perf_counter()-start),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return int(bool(errors))
if __name__=='__main__':raise SystemExit(main())
