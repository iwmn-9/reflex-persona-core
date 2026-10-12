"""Matched endpoints and enacted preferences; never label resource score a win."""
import argparse,collections,hashlib,json
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def contrast(rows):
    pairs=collections.defaultdict(dict)
    for r in rows:pairs[(r['game'],r['regime'],r['profile'],r['seed'])][r['metric']]=r
    out=[]
    for key,p in sorted(pairs.items()):
        if set(p)!={'purpose','persona'}:continue
        a,b=p['purpose'],p['persona'];d=b['score']-a['score']
        out.append(dict(game=key[0],regime=key[1],profile=key[2],seed=key[3],purpose_delta=d,
            appraisal_delta=b['appraisal']-a['appraisal'],action_changed=a['actions']!=b['actions'],
            value_deltas={k:b['values'][k]-a['values'][k] for k in a['values']},
            prior_model_uses=a['model_uses'],persona_model_uses=b['model_uses']))
    def aggregate(xs):return dict(pairs=len(xs),better=sum(r['purpose_delta']>1e-9 for r in xs),worse=sum(r['purpose_delta']< -1e-9 for r in xs),
        same=sum(abs(r['purpose_delta'])<=1e-9 for r in xs),action_changes=sum(r['action_changed'] for r in xs),
        mean_purpose_delta=sum(r['purpose_delta'] for r in xs)/max(1,len(xs)),
        prior_model_uses=sum(r['prior_model_uses'] for r in xs),persona_model_uses=sum(r['persona_model_uses'] for r in xs))
    return dict(total=aggregate(out),by_game={g:aggregate([r for r in out if r['game']==g]) for g in sorted({r['game'] for r in out})},
        by_profile={g:aggregate([r for r in out if r['profile']==g]) for g in sorted({r['profile'] for r in out})},pairs=out)


def analyze(folder,partial=False):
    folder=Path(folder);pre=json.loads((folder/'preregister.json').read_text(encoding='utf-8'));raw=folder/'trajectories.jsonl'
    rows=[];comparisons=[]
    for line in raw.read_text(encoding='utf-8').splitlines():
        try:r=json.loads(line)
        except json.JSONDecodeError:
            if partial:continue
            raise
        comparisons.extend(dict(game=r['game'],regime=r['regime'],profile=r['profile'],seed=r['seed'],**t['comparison']) for t in r['trace'] if t['comparison'] and r['metric']=='persona')
        rows.append({k:v for k,v in r.items() if k!='trace'})
    errors=[];complete=False
    if (folder/'evaluation.json').exists():
        e=json.loads((folder/'evaluation.json').read_text(encoding='utf-8'));errors=e['errors'];complete=len(rows)==e['planned'] and not errors
        assert len(rows)==e['episodes']
    if not partial and not complete:raise ValueError('complete cohort required')
    assert len({r['index'] for r in rows})==len(rows)
    for path,h in pre['source_hashes'].items():assert sha(folder/'source'/path)==h
    return dict(cohort=folder.name,complete=complete,episodes=len(rows),errors=errors,comparison=contrast(rows),
        persona_trials=dict(comparisons=len(comparisons),positive=sum(r['persona_delta']>0 for r in comparisons),negative=sum(r['persona_delta']<0 for r in comparisons),
            purpose_lower_persona_higher=sum(r['purpose_delta']<0 and r['persona_delta']>0 for r in comparisons),
            revoked=sum(r['revoked'] for r in comparisons),
            max_decomposition_error=max((abs(r['purpose_delta']-r['planned_purpose_delta']-r['purpose_surprise_delta']) for r in comparisons),default=0)),
        raw_sha256=sha(raw),preregister_sha256=sha(folder/'preregister.json'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cohorts',nargs='+',required=True);ap.add_argument('--output',required=True);ap.add_argument('--partial',action='store_true');a=ap.parse_args()
    s=dict(format='persona-feedback-analysis-v1',partial=a.partial,cohorts=[analyze(p,a.partial) for p in a.cohorts])
    p=Path(a.output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([dict(cohort=c['cohort'],episodes=c['episodes'],**c['comparison']['total'],trials=c['persona_trials']) for c in s['cohorts']]))


if __name__=='__main__':main()
