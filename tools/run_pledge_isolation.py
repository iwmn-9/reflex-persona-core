"""Replace the entire affected diagnostic cohort; leave valid game trials intact."""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.commitment_experiment import pledge_episode
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--frozen',action='store_true');a=p.parse_args();out=Path(a.output).resolve()
    if not a.frozen:
        out.mkdir(parents=True);source=out/'source';source.mkdir()
        for name in ('reflex','tools','tests'):shutil.copytree(ROOT/name,source/name,ignore=shutil.ignore_patterns('__pycache__'))
        return subprocess.call([sys.executable,str(source/'tools/run_pledge_isolation.py'),'--output',str(out),'--frozen'],cwd=source)
    assert ROOT==out/'source'
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for directory in ('reflex','tools','tests') for p in sorted((ROOT/directory).glob('*.py'))}
    jobs=[(changed,seed,r,i) for changed in (False,True) for seed in range(64) for r in (0.,.5,1.) for i in (0.,1.)]
    pre=dict(format='pledge-isolation-v1',jobs=jobs,source_hashes=hashes,
        replacement='all768 pledge diagnostics from v2 replaced; valid288 game episodes retained',
        actor_random_seed=42,world_random_seeds=list(range(64)),initial_state='principle mode, belonging deficit .1',
        limitation='authored stochastic rules; no empirical human-personality validation')
    (out/'preregister.json').write_text(json.dumps(pre,indent=2)+'\n',encoding='utf-8')
    with (out/'pledge.jsonl').open('w',encoding='utf-8') as f:
        for changed,seed,r,i in jobs:f.write(json.dumps(pledge_episode(seed,r,i,changed=changed))+'\n')
    print('isolated pledge episodes',len(jobs))
if __name__=='__main__':raise SystemExit(main())
