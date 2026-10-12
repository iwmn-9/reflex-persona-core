"""Compare complete immediate records against a caller-supplied old checkout."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--old-source');p.add_argument('--source');p.add_argument('--contexts');p.add_argument('--destination');a=p.parse_args()
    if a.source:
        sys.path.insert(0,a.source)
        from reflex.core import Policy,character_key
        contexts=json.loads(Path(a.contexts).read_text(encoding='utf-8'));rows=[]
        for c in contexts:
            for mode,rigidity in (('lexicographic',None),('finite',None),('personal',0.),('personal',.5),('personal',1.)):
                policy=Policy(principle_priority=mode,commitments=None if rigidity is None else {character_key(c):rigidity})
                for stochastic in (False,True):rows.append(policy.choose(c,stochastic))
        Path(a.destination).write_text(json.dumps(rows,ensure_ascii=False)+'\n',encoding='utf-8');return 0
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(ROOT))
    from tools.verify_integrated_contracts import contexts
    cs=[c for case,profile,c in contexts()];file=out/'contexts.json';file.write_text(json.dumps(cs,ensure_ascii=False)+'\n',encoding='utf-8')
    for name,source in (('old',Path(a.old_source).resolve()),('current',ROOT)):
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--source',str(source),'--contexts',str(file),'--destination',str(out/(name+'.json'))],check=True)
    old=json.loads((out/'old.json').read_text(encoding='utf-8'));current=json.loads((out/'current.json').read_text(encoding='utf-8'))
    mismatches=[i for i,(x,y) in enumerate(zip(old,current)) if x!=y];assert len(old)==len(current)==len(cs)*10
    result=dict(format='integrated-immediate-parity-v1',contexts=len(cs),records=len(old),mismatches=mismatches,
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.json')},
        configurations=['lexicographic','finite','personal-0','personal-.5','personal-1'],stochastic=[False,True],
        limitation='immediate choices only; new forecast units and social appraisal are intentionally opt-in')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
    return int(bool(mismatches))

if __name__=='__main__':raise SystemExit(main())
