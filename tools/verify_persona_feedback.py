"""Whole ordinary lifecycle parity and final audit-only source bridge."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--old-source');ap.add_argument('--output',required=True)
    ap.add_argument('--source');ap.add_argument('--confirmation');ap.add_argument('--replay');ap.add_argument('--reuse-defaults',action='store_true');a=ap.parse_args();out=Path(a.output).resolve()
    if a.source:
        sys.path.insert(0,str(Path(a.source).resolve()))
        from reflex.loop_experiment import group,FACTORIES
        rows=[]
        for game in FACTORIES:
            for r in group(game,(9100,9101),'integrated_read'):
                r.pop('p50_ms');rows.append(r)
        out.write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8');return
    if a.reuse_defaults:
        receipt=json.loads((out/'defaults.json').read_text(encoding='utf-8'))
        assert receipt['source_hashes']=={p.name:sha(p) for p in sorted((ROOT/'reflex').glob('*.py'))}
        for name in ('old','current'):assert receipt[name]==sha(out/(name+'.json'))
    else:
        out.mkdir(parents=True,exist_ok=False)
        for name,source in (('old',Path(a.old_source).resolve()),('current',ROOT)):
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--source',str(source),'--output',str(out/(name+'.json'))],check=True)
    old=json.loads((out/'old.json').read_text(encoding='utf-8'));new=json.loads((out/'current.json').read_text(encoding='utf-8'))
    assert old==new
    (out/'defaults.json').write_text(json.dumps(dict(old=sha(out/'old.json'),current=sha(out/'current.json'),
        source_hashes={p.name:sha(p) for p in sorted((ROOT/'reflex').glob('*.py'))}),indent=2),encoding='utf-8')
    def rows(folder):
        return {(r['game'],r['regime'],r['profile'],r['seed'],r['metric']):r for r in
            (json.loads(line) for line in (Path(folder)/'trajectories.jsonl').read_text(encoding='utf-8').splitlines())}
    for folder in (a.confirmation,a.replay):
        e=json.loads((Path(folder)/'evaluation.json').read_text(encoding='utf-8'))
        if e['episodes']!=e['planned'] or e['errors']:raise ValueError('complete cohort required before replay comparison')
    original=rows(a.confirmation);replay=rows(a.replay);assert set(replay)<=set(original)
    for key,r in replay.items():
        earlier=original[key]
        for field in ('actions','score','values','appraisal','comparisons','model_uses','revoked'):assert r[field]==earlier[field],(key,field)
        for x,y in zip(r['trace'],earlier['trace']):
            for field in ('action','response','flow','score','energy','appraisal','reading','predictive','owner'):assert x[field]==y[field],(key,field)
            if y['comparison'] is None:assert x['comparison'] is None
            else:
                for field,value in y['comparison'].items():assert x['comparison'][field]==value,(key,field)
    record=dict(ordinary_episodes=len(old),ordinary_decisions=sum(len(r['trace']) for r in old),ordinary_whole_records_equal=True,
        final_replays=len(replay),whole_actual_and_learning_equal=True,
        default_sha256=sha(out/'old.json'),current_sha256=sha(out/'current.json'),
        source_hashes={p.relative_to(ROOT).as_posix():sha(p) for d in ('reflex','tests') for p in sorted((ROOT/d).glob('*.py'))})
    (out/'result.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in record.items() if k!='source_hashes'}))


if __name__=='__main__':main()
