"""Match fixed-horizon worlds without treating personas as independent seeds."""
import argparse,hashlib,json,sys
from collections import Counter,defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.laboratory import PROFILES


def summarize(repo,root,plan,partial=False):
    rows={};origins={};missing=[];incomplete=[]
    def add(row,origin):
        key=tuple(row[k] for k in ('game','profile','seed','regime'))+(row['tail_steps']+1,)
        assert key not in rows,('duplicate condition',key)
        rows[key]=row;origins[key]=str(origin)
    for old in plan['retained_baselines']:
        source=repo/old['path']/'trajectories.jsonl'
        assert hashlib.sha256(source.read_bytes()).hexdigest()==old['sha256']
        selected=[r for r in map(json.loads,source.read_text().splitlines()) if r['arm']=='receding_recovery']
        assert sorted([r['game'],r['profile'],r['regime']] for r in selected)==sorted(old['conditions'])
        for row in selected:add(row,source)
    for job in plan['new_jobs']:
        path=root/('horizon-'+job['key']);source=path/'trajectories.jsonl'
        if not source.exists():
            (incomplete if path.exists() else missing).append(job['key']);continue
        if not (path/'audit.json').exists():incomplete.append(job['key']);continue
        audit=json.loads((path/'audit.json').read_text())
        assert audit['complete'] and audit['source_freeze_verified'] and audit['completed_runs']==1
        assert audit['trajectory_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
        row=json.loads(source.read_text());g,p,s,r,turns,interval,tail,offset=job['job']
        assert [row['game'],row['profile'],row['seed'],row['regime'],row['turns'],row['interval'],row['tail_steps'],row['offset']]==[g,PROFILES[p]['id'],s,r,turns,interval,tail,offset]
        add(row,source)
    if not partial:assert not missing and not incomplete,(missing,incomplete)
    comparisons=[]
    for key,right in sorted(rows.items()):
        if key[-1]==16 or key[:-1]+(16,) not in rows:continue
        left=rows[key[:-1]+(16,)];lt=left['trace'];rt=right['trace']
        for a,b in zip(lt,rt):assert a['response']==b['response'] and a['truth']==b['truth'],'unmatched world events'
        def score_at(trace,tick):return trace[min(tick,len(trace))-1]['goal']
        comparisons.append(dict(game=key[0],profile=key[1],seed=key[2],regime=key[3],horizon=key[4],
            delta=right['score']-left['score'],baseline_score=left['score'],candidate_score=right['score'],
            tick112_delta=score_at(rt,112)-score_at(lt,112),
            action_changes=sum(a['action']!=b['action'] for a,b in zip(lt,rt)),
            baseline_actions=dict(Counter(t['action'] for t in lt)),candidate_actions=dict(Counter(t['action'] for t in rt)),
            baseline_source=origins[key[:-1]+(16,)],candidate_source=origins[key]))
    groups=[]
    for level in ('game','persona','condition'):
        bins=defaultdict(list)
        for row in comparisons:
            key=(row['game'],row['horizon'])
            if level in ('persona','condition'):key+=(row['profile'],)
            if level=='condition':key+=(row['regime'],)
            bins[key].append(row)
        for key,items in sorted(bins.items()):
            ds=[x['delta'] for x in items]
            groups.append(dict(level=level,key=key,conditions=len(items),world_seeds=sorted({x['seed'] for x in items}),
                mean_delta=sum(ds)/len(ds),mean_tick112_delta=sum(x['tick112_delta'] for x in items)/len(items),
                better=sum(d>1e-12 for d in ds),same=sum(abs(d)<=1e-12 for d in ds),worse=sum(d< -1e-12 for d in ds),worst=min(ds)))
    return dict(complete=not missing and not incomplete,registered_runs=plan['total_development_runs'],
        collected_runs=len(rows),missing=missing,incomplete=incomplete,comparisons=comparisons,groups=groups,
        limitation='Development seed only; no cross-game strength mean or significance claim. Partial summaries are not adoption evidence.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/long_horizon_development')
    p.add_argument('--plan',default='evidence/long_horizon/development_plan.json');p.add_argument('--partial',action='store_true')
    p.add_argument('--output',default='results/long_horizon_development/summary.json');a=p.parse_args()
    result=summarize(Path('.'),Path(a.root),json.loads(Path(a.plan).read_text()),a.partial)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('missing','incomplete','comparisons','groups')}))
    print(json.dumps([x for x in result['groups'] if x['level']=='game'],indent=2))
