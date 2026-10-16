"""Export verified continuous-world comparisons and immutable numeric evidence."""
import argparse,contextlib,io,json,gzip,hashlib,sys
from collections import Counter,defaultdict
from pathlib import Path
from zipfile import ZipFile,ZipInfo,ZIP_DEFLATED
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.audit_owned_value import audit


def comparisons(rows):
    grouped=defaultdict(dict)
    for r in rows:
        key=tuple(r[k] for k in ('game','profile','seed','regime'))
        assert r['arm'] not in grouped[key],'duplicate study/arm'
        grouped[key][r['arm']]=r
    pairs=[]
    for key,arms in sorted(grouped.items()):
        if 'endpoint' not in arms:continue
        for base,r in arms.items():
            if base=='endpoint':continue
            delta=arms['endpoint']['score']-r['score']
            pairs.append(dict(zip(('game','profile','seed','regime'),key),baseline=base,delta=delta,
                better=int(delta>1e-12),same=int(abs(delta)<=1e-12),worse=int(delta< -1e-12)))
    summary=[]
    for base in sorted({p['baseline'] for p in pairs}):
        for game in (*sorted({p['game'] for p in pairs}),'all'):
            group=[p for p in pairs if p['baseline']==base and (game=='all' or p['game']==game)]
            if not group:continue
            summary.append(dict(baseline=base,game=game,pairs=len(group),
                **{k:sum(p[k] for p in group) for k in ('better','same','worse')},
                mean_delta=float(np.mean([p['delta'] for p in group])),worst=min(p['delta'] for p in group)))
    return dict(pairs=pairs,summary=summary)


def export(output,development,holdouts,library,extras):
    output.mkdir(parents=True,exist_ok=True);evidence={};checks=Counter();all_rows=[]
    studies=[development,*holdouts,library]
    with ZipFile(output/'source_snapshots.zip','w',compression=ZIP_DEFLATED) as archive:
        for root in [*studies,*extras]:
            snapshot=json.loads((root/'source_snapshot.json').read_text())
            for name,sha in snapshot.items():
                data=(root/'_source'/name).read_bytes()
                assert hashlib.sha256(data).hexdigest()==sha
                info=ZipInfo(root.name+'/'+name,date_time=(2026,10,17,0,0,0));info.compress_type=ZIP_DEFLATED
                archive.writestr(info,data)
            for name in ('registration.json','source_snapshot.json','summary.json','evaluation.json','predictions.jsonl'):
                if (root/name).is_file():
                    data=(root/name).read_bytes()
                    (output/(root.name+'-'+name)).write_bytes(data)
            if root not in studies:continue
            with contextlib.redirect_stdout(io.StringIO()):result=audit(root)
            evidence[root.name]=result;checks.update(result['checks']);all_rows+=result['runs']
            (output/(root.name+'-audit.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
            data=(root/'trajectories.jsonl').read_bytes()
            (output/(root.name+'-trajectories.jsonl.gz')).write_bytes(gzip.compress(data,compresslevel=9,mtime=0))
    holdout_rows=[r for root in holdouts for r in evidence[root.name]['runs']]
    librows=evidence[library.name]['runs']
    result=dict(development=comparisons(evidence[development.name]['runs']),
        holdout=comparisons(holdout_rows),library=comparisons([*holdout_rows,*librows]),checks=dict(checks),
        runs=len(all_rows),executed=sum(r['executed'] for r in all_rows),
        semantics='Matched conditions, not independent NPC samples or win rates. All scores maximize declared game purpose.')
    # Keep the library comparison table distinct from the original three arms.
    for key in ('pairs','summary'):
        result['library'][key]=[p for p in result['library'][key] if p['baseline']=='endpoint_library']
    (output/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:result[k] for k in ('runs','executed','checks')},indent=2))
    print(json.dumps(result['holdout']['summary'],indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--development',required=True)
    p.add_argument('--holdouts',nargs='+',required=True);p.add_argument('--library',required=True)
    p.add_argument('--extras',nargs='*',default=[]);a=p.parse_args()
    export(Path(a.output),Path(a.development),list(map(Path,a.holdouts)),Path(a.library),list(map(Path,a.extras)))
