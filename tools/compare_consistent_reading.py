"""Audit a complete 2x2 predictor/reader-selection long-play ablation."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.owned_checkpoint import encoded,sha


def load(audit):
    audit=Path(audit)
    proof=json.loads((audit/'verification.json').read_bytes())
    assert proof['complete']
    for name,checksum in proof['artifacts'].items():
        assert sha((audit/name).read_bytes())==checksum
    rows=[json.loads(line) for line in gzip.decompress((audit/'runs.jsonl.gz').read_bytes()).splitlines()]
    assert len(rows)==256
    return {r['spec']['id']:r for r in rows}


def compare(reference,current,plan_path,output):
    reference,current,plan_path,output=map(Path,(reference,current,plan_path,output))
    plan=json.loads(plan_path.read_bytes());old=load(reference);new=load(current)
    assert sha((reference/'verification.json').read_bytes())==plan['comparison']['reference_verification_sha256']
    assert sha((reference/'runs.jsonl.gz').read_bytes())==plan['comparison']['reference_runs_sha256']
    oldplan=json.loads((reference/'plan.json').read_bytes())
    assert oldplan['specs']==plan['specs'] and old.keys()==new.keys()
    assert json.loads((current/'plan.json').read_bytes())==plan
    changed=[k for k,v in oldplan['sources'].items() if plan['sources'].get(k)!=v]
    assert changed==['reflex/decision_loop.py'],changed
    # Evidence is tied to its frozen worker, not to a later checkout that may
    # legitimately contain post-study test corrections.
    for folder,contract in ((reference,oldplan),(current,plan)):
        with zipfile.ZipFile(folder/'source.zip') as archive:
            for name,checksum in contract['sources'].items():
                assert sha(archive.read(name))==checksum
    rows=[]
    for key in sorted(old):
        assert old[key]['spec']==new[key]['spec']
        for predictor in ('stable','transfer'):
            a,b=old[key]['arms'][predictor],new[key]['arms'][predictor]
            assert [(t['response'],t['truth']) for t in a['trace']]==[(t['response'],t['truth']) for t in b['trace']]
            # Event histories are action-independent in these registered
            # worlds. The integration change must not change prediction.
            assert [t['response_prediction'] for t in a['trace']]==[t['response_prediction'] for t in b['trace']]
            switches=lambda trace:sum(x['action']!=y['action'] for x,y in zip(trace,trace[1:]))
            rows.append(dict(spec=old[key]['spec'],predictor=predictor,
                delta={m:b[m]-a[m] for m in ('score','conditional')},
                switches=[switches(a['trace']),switches(b['trace'])],
                changed_choices=sum(x['action']!=y['action'] for x,y in zip(a['trace'],b['trace'])),
                scores=[a['score'],b['score']],conditionals=[a['conditional'],b['conditional']]))
    def group(rs):
        return dict(pairs=len(rs),mean_delta={k:sum(r['delta'][k] for r in rs)/len(rs) for k in ('score','conditional')},
                    better=sum(r['delta']['score']>1e-12 for r in rs),same=sum(abs(r['delta']['score'])<=1e-12 for r in rs),
                    worse=sum(r['delta']['score']< -1e-12 for r in rs),
                    mean_switches=[sum(r['switches'][i] for r in rs)/len(rs) for i in range(2)])
    summary={}
    for predictor in ('stable','transfer'):
        part=[r for r in rows if r['predictor']==predictor]
        families={g:group([r for r in part if r['spec']['game']==g]) for g in sorted({r['spec']['game'] for r in part})}
        summary[predictor]=dict(families=families,
            groups=[dict(game=g,regime=m,**group([r for r in part if r['spec']['game']==g and r['spec']['regime']==m]))
                    for g in families for m in sorted({r['spec']['regime'] for r in part})],
            personas={g:{str(p):group([r for r in part if r['spec']['game']==g and r['spec']['profile']==p]) for p in range(4)} for g in families},
            all_family_means_nonnegative=all(v>=-1e-12 for x in families.values() for v in x['mean_delta'].values()),
            adverse=[r for r in part if any(v< -1e-12 for v in r['delta'].values())])
    if output.exists():raise FileExistsError('retain prior comparison')
    output.mkdir(parents=True)
    (output/'comparison.json').write_bytes(encoded(dict(complete=True,comparisons=len(rows),summary=summary,
        changed_common_sources=changed,adoption=False,
        limitation='Development configurations; 16 response streams reused across families/personas. Neither repeated seeds nor 512 comparisons constitute independent H32 confirmation. All predictive laws are unchanged by this integration-only intervention.')))
    (output/'rows.json.gz').write_bytes(gzip.compress(encoded(rows),mtime=0))
    (output/'verification.json').write_bytes(encoded(dict(complete=True,reference_verification_sha256=sha((reference/'verification.json').read_bytes()),
        current_verification_sha256=sha((current/'verification.json').read_bytes()),plan_sha256=sha(plan_path.read_bytes()),
        analyzer_sha256=sha(Path(__file__).read_bytes()),artifacts={p.name:sha(p.read_bytes()) for p in output.iterdir()})))
    print(json.dumps({p:{k:summary[p][k] for k in ('families','all_family_means_nonnegative')} for p in summary},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','current','plan','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();compare(a.reference,a.current,a.plan,a.output)
