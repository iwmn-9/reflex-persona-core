"""Verify complete frozen cohorts and retain every paired regression."""
import argparse,hashlib,json,sys
from pathlib import Path
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import digest

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def tally(deltas):return dict(better=sum(d>1e-10 for d in deltas),worse=sum(d< -1e-10 for d in deltas),same=sum(abs(d)<=1e-10 for d in deltas))

def analyze(root):
    root=Path(root);pre=read(root/'preregister.json');e=read(root/'evaluation.json')
    assert e['planned']==e['complete']==len(pre['jobs']) and not e['errors']
    assert e['preregister_sha256']==digest(pre)
    assert e['trajectories_sha256']==sha(root/'trajectories.jsonl')
    assert pre['source_snapshot_sha256']==sha(root/'source_snapshot.json')
    assert all(sha(root/'_source'/name)==h for name,h in read(root/'source_snapshot.json').items())
    rows=[json.loads(x) for x in (root/'trajectories.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==len(pre['jobs']) and {r['index'] for r in rows}==set(range(len(rows)))
    summaries={r['index']:r for r in e['runs']}
    for r in rows:
        assert r['job']==pre['jobs'][r['index']]
        assert {k:v for k,v in r.items() if k!='trace'}==summaries[r['index']]
        if 'replay' in r:
            assert r['replay'] and r['actions']==[t['root'] for t in r['trace']]
        elif 'success_probability' in r:assert 0<=r['success_probability']<=r['optimal_probability']+1e-12<=1+1e-12
        else:assert -1<=r['goal']<=r['optimal']+1e-12<=1+1e-12
    graph='success_probability' in rows[0];gambling='regime' in rows[0]
    kind='graph-policy' if graph else 'gambling-policy' if gambling else 'game-episode'
    group=defaultdict(dict)
    for r in rows:
        key=[r['regime'],r['cash'],r['left'],r['profile']] if gambling else [r['case'],r['profile'],r.get('seed')]
        group[digest(key)][r['arm']]=r
    comparisons=(('baseline_window','sampled_tree'),('baseline_window','exact_tree'),('exact_reflex','exact_tree'),
        ('sampled_tree','exact_tree'),('baseline_rolling','exact_full'),('baseline_roll2','baseline_window'),
        ('baseline','sampled_tree'),('baseline','exact_tree'))
    value=lambda r:r['success_probability'] if graph else r['goal'] if gambling else r['goal']['value']
    pairs=[]
    for g in group.values():
        for old,new in comparisons:
            if old not in g or new not in g:continue
            a,b=g[old],g[new];label=a['regime'] if gambling else str(a['case']['length']) if graph else a['case']['game']
            pairs.append(dict(condition=[a['regime'],a['cash'],a['left']] if gambling else a['case'],seed=a.get('seed'),profile=a['profile'],
                group=label,comparison=new+'-'+old,before=value(a),after=value(b),delta=value(b)-value(a)))
    summary=[]
    for label in sorted({p['group'] for p in pairs}):
        for comparison in sorted({p['comparison'] for p in pairs}):
            pp=[p for p in pairs if p['group']==label and p['comparison']==comparison]
            if pp:summary.append(dict(group=label,comparison=comparison,pairs=len(pp),**tally([p['delta'] for p in pp]),
                mean_delta=sum(p['delta'] for p in pp)/len(pp),by_persona={n:tally([p['delta'] for p in pp if p['profile']==n]) for n in sorted({p['profile'] for p in pp})}))
    arms=[]
    for label in sorted({r['regime'] if gambling else str(r['case']['length']) if graph else r['case']['game'] for r in rows}):
        for arm in sorted({r['arm'] for r in rows}):
            rr=[r for r in rows if (r['regime'] if gambling else str(r['case']['length']) if graph else r['case']['game'])==label and r['arm']==arm]
            ts=[t for r in rr for t in r['trace']];trees=[t['tree'] for t in ts if t.get('tree')]
            arms.append(dict(group=label,arm=arm,runs=len(rr),decisions=len(ts),mean_value=sum(value(r) for r in rr)/len(rr),
                mean_optimal=None if not (graph or gambling) else sum(r['optimal_probability'] if graph else r['optimal'] for r in rr)/len(rr),
                tree_calls=len(trees),declined=sum(not t['complete'] for t in trees),logical_nodes=sum(t['nodes'] for t in trees),
                status=None if graph or gambling else dict(Counter(r['goal']['status'] for r in rr))))
    return dict(kind=kind,stage=pre.get('stage','gambling_transfer'),runs=len(rows),decisions=sum(len(r['trace']) for r in rows),
        pairs=pairs,comparisons=summary,arms=arms,errors=[],
        controls=dict(all_registered_jobs_verified=True,all_source_hashes_verified=True,preregister_sha256=digest(pre),
            trajectories_sha256=sha(root/'trajectories.jsonl'),source_snapshot_sha256=sha(root/'source_snapshot.json')))

def main():
    p=argparse.ArgumentParser();p.add_argument('--cohorts',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError('fresh result required')
    result=[analyze(root) for root in a.cohorts]
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{k:r[k] for k in ('kind','stage','runs','decisions','comparisons','arms')} for r in result],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
