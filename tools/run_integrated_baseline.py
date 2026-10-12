"""Frozen old/return/owned comparisons; retain every error and paired loss."""
import argparse,hashlib,io,json,shutil,subprocess,sys,time,zipfile
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reflex.core import digest
from reflex.laboratory import profiles
from tools.run_contingent_coverage import cases

OLD='c244958990a5b5c7b2105a538bd7a12b72107afb'
GAMES=('auction_contest','hagetaka','empire_race','team_combat','gambling','cyclic_arena','shared_commons','branching_delivery')

def hashes(root):
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('reflex','tools','tests') for p in sorted((root/folder).glob('*.py'))}

def save(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2)

def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    current=out/'current';old=out/'legacy';current.mkdir();old.mkdir()
    for folder in ('reflex','tools','tests'):
        shutil.copytree(ROOT/folder,current/folder,ignore=shutil.ignore_patterns('__pycache__'))
    data=subprocess.run(['git','archive','--format=zip',OLD,'reflex','tools','tests'],cwd=ROOT,check=True,capture_output=True).stdout
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for member in z.infolist():
            target=(old/member.filename).resolve()
            if not target.is_relative_to(old.resolve()):raise ValueError('archive outside frozen source')
        z.extractall(old)
    for relative in ('reflex/integrated_trials.py','tools/run_integrated_baseline.py'):
        shutil.copyfile(ROOT/relative,old/relative)
    jobs={'legacy':[],'current':[],'life':[]}
    for stage in ('development','confirmation'):
        for case in cases(stage):
            if case['game'] not in GAMES:continue
            for profile in profiles():
                for seed in range(1700,1704):
                    for arm in ('legacy','returns','owned'):
                        job=dict(kind='game',case=case,profile=profile,seed=seed,arm=arm,stage=stage,horizon=2)
                        jobs['legacy' if arm=='legacy' else 'current'].append(job)
    for domain in ('exchange','watch'):
        for layout in ('balanced4','balanced8'):
            for seed in range(1700,1704):
                for arm in ('reflex','returns','owned'):
                    jobs['life'].append(dict(kind='life',domain=domain,layout=layout,seed=seed,arm=arm,turns=36,horizon=2))
    pre=dict(format='integrated-baseline-v1',old_revision=OLD,jobs=jobs,source_hashes={'legacy':hashes(old),'current':hashes(current)},
        old_overlays=['same new game-side harness; old core and rules remain archived'],
        acceptance=['all specified jobs and losses retained','selected actual action streams replay exactly','modeled experience never changes real actor',
            'return units independent of empty terminal tail','hypothetical immediate choices use same owned policy and memories'],
        hypotheses=['return-unit correction changes some choices without altering default immediate behavior',
            'owned executor improves prediction parity; strength gains must be measured, not assumed'],
        limitations=['existing authored adapters and explicit endpoint proxies, not rule-only universal intelligence',
            'horizon two, two declared model seeds; no calibrated win probability or recursive outer planner',
            'empirical learning enabled only in three declared immediate-feedback games; owned package includes actual and modeled learning',
            'life peers use declared neutral hypotheses, not secret true personality; life score is not a win rate',
            'schema/window monitors assessed separately; no blanket speed or human-likeness claim'],cloud_runtime_used=False)
    save(out/'preregister.json',pre)
    print(json.dumps({k:len(v) for k,v in jobs.items()}),flush=True)

def worker(job):
    from reflex.integrated_trials import play,life
    start=time.perf_counter()
    if job['kind']=='game':r=play(job['case'],job['profile'],job['seed'],job['arm'],horizon=job['horizon'])
    else:r=life(job['seed'],job['domain'],job['layout'],job['arm'],turns=job['turns'],horizon=job['horizon'])
    r['elapsed_seconds']=time.perf_counter()-start;r['job']=job
    return r

def run(out,cohort,workers):
    pre=json.loads((out/'preregister.json').read_text(encoding='utf-8'));source='legacy' if cohort=='legacy' else 'current'
    if ROOT!=out/source:raise ValueError('execute the declared frozen source')
    fixed=pre['source_hashes'][source]
    if hashes(ROOT)!=fixed:raise ValueError('frozen sources changed before execution')
    jobs=pre['jobs'][cohort];runs=[];errors=[];started=time.perf_counter()
    with (out/(cohort+'.jsonl')).open('x',encoding='utf-8',newline='\n') as stream,ProcessPoolExecutor(max_workers=workers) as pool:
        fs={pool.submit(worker,j):i for i,j in enumerate(jobs)}
        for f in as_completed(fs):
            i=fs[f]
            try:r=f.result()
            except Exception as e:
                errors.append(dict(index=i,job=jobs[i],error=repr(e)));print('ERROR',cohort,i,repr(e),flush=True);continue
            r['index']=i;stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush()
            runs.append({k:v for k,v in r.items() if k!='trace'})
            print(cohort,len(runs)+len(errors),'/',len(jobs),jobs[i].get('case',jobs[i].get('domain')),jobs[i]['arm'],round(r['elapsed_seconds'],2),flush=True)
    assert hashes(ROOT)==fixed
    save(out/(cohort+'-result.json'),dict(planned=len(jobs),episodes=len(runs),runs=sorted(runs,key=lambda r:r['index']),errors=errors,
        elapsed_seconds=time.perf_counter()-started,source_hashes=fixed,preregister_sha256=digest(pre)))
    return 1 if errors else 0

def summarize(out):
    pre=json.loads((out/'preregister.json').read_text(encoding='utf-8'));results={k:json.loads((out/(k+'-result.json')).read_text(encoding='utf-8')) for k in pre['jobs']}
    runs=results['legacy']['runs']+results['current']['runs'];groups={}
    for r in runs:
        key=digest([r['case'],r['profile'],r['seed'],r['job']['stage']]);groups.setdefault(key,{})[r['arm']]=r
    paired=[]
    for group in groups.values():
        if set(group)!=set(('legacy','returns','owned')):continue
        a=group['legacy'];b=group['returns'];c=group['owned']
        paired.append(dict(game=a['case']['game'],stage=a['job']['stage'],profile=a['profile'],seed=a['seed'],case=a['case'],
            legacy=a['goal']['value'],returns=b['goal']['value'],owned=c['goal']['value'],
            return_delta=b['goal']['value']-a['goal']['value'],owned_delta=c['goal']['value']-b['goal']['value'],
            returns_actions_changed=a['actions']!=b['actions'],owned_actions_changed=b['actions']!=c['actions']))
    rows=[]
    for game in GAMES:
        ps=[p for p in paired if p['game']==game]
        row=dict(game=game,pairs=len(ps))
        for name in ('return','owned'):
            xs=[p[name+'_delta'] for p in ps];row[name]=dict(better=sum(x>1e-12 for x in xs),worse=sum(x<-1e-12 for x in xs),same=sum(abs(x)<=1e-12 for x in xs),mean_delta=sum(xs)/len(xs) if xs else None)
        rows.append(row)
    result=dict(format=pre['format'],game_episodes=sum(results[k]['episodes'] for k in ('legacy','current')),
        life_worlds=results['life']['episodes'],life_actor_episodes=sum(len(r['actors']) for r in results['life']['runs']),
        errors={k:v['errors'] for k,v in results.items()},paired=paired,summary=rows,preregister_sha256=digest(pre))
    save(out/'evaluation.json',result);print(json.dumps({k:v for k,v in result.items() if k!='paired'},ensure_ascii=False,indent=2))
    return int(any(result['errors'].values()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--summarize',action='store_true')
    p.add_argument('--cohort',choices=('legacy','current','life'));p.add_argument('--workers',type=int,default=4);a=p.parse_args();out=Path(a.output).resolve()
    if a.prepare:prepare(out);return 0
    if a.summarize:return summarize(out)
    if not a.cohort or not 1<=a.workers<=8:raise ValueError('declared cohort and bounded independent workers required')
    return run(out,a.cohort,a.workers)

if __name__=='__main__':raise SystemExit(main())
