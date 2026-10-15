"""Physical replay of the historical-protocol recovery comparison."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.laboratory import PROFILES
from tools.run_lifecycle_soak import Adapter,distribution,sample


def audit(root,previous=None):
    fingerprints=json.loads((root/'source_snapshot.json').read_text())
    for name,sha in fingerprints.items():assert hashlib.sha256((root/'_source'/name).read_bytes()).hexdigest()==sha
    reg=json.loads((root/'registration.json').read_text());rows=[json.loads(x) for x in (root/'trajectories.jsonl').read_text().splitlines()]
    expected={(g,PROFILES[p]['id'],s,r,arm) for g,p,s,r,n,v,pe in reg['jobs'] for arm in ('choice','recovery')}
    actual={(r['game'],r['profile'],r['seed'],r['schedule'],r['arm']) for r in rows}
    assert len(rows)==len(expected) and expected==actual
    profiles={p['id']:p for p in PROFILES};decisions=0
    for r in rows:
        a=Adapter(r['game'],profiles[r['profile']],r['seed']%2);s=a.start();scores=[]
        uniforms=np.random.default_rng(r['seed']).random(r['turns'])
        for tick,t in enumerate(r['trace']):
            if a.terminal(s):scores.append(a.goal(s));s=a.start(s['round']+1,tick)
            assert t['tick']==tick and t['action'] in a.keys(s)
            p=distribution(a.symbols(),tick,r['turns'],r['schedule'])
            s,flow,symbol=a.transition(s,t['action'],sample(p,uniforms[tick]))
            assert symbol==t['response'] and a.goal(s)==t['goal']
            assert t['predictor_observations']==tick+1
            decisions+=1
        if a.terminal(s):scores.append(a.goal(s))
        assert scores==r['scores'] and float(np.mean(scores))==r['score']
    result=json.loads((root/'summary.json').read_text());by_game=[]
    for g in ('all','auction','resources','combat','gambling'):
        delta=[p['delta'] for p in result['pairs'] if g=='all' or p['game']==g]
        by_game.append(dict(game=g,pairs=len(delta),better=sum(d>1e-12 for d in delta),equal=sum(abs(d)<=1e-12 for d in delta),
            worse=sum(d< -1e-12 for d in delta),mean_delta=float(np.mean(delta))))
    matched=0
    if previous:
        old=[json.loads(x) for x in (previous/'trajectories.jsonl').read_text().splitlines()]
        by={(r['game'],r['profile'],r['schedule']):r for r in old if r['arm']=='persona_choice'}
        for r in rows:
            if r['arm']=='choice':
                prior=by[(r['game'],r['profile'],r['schedule'])]
                assert r['trace']==prior['trace'] and r['score']==prior['score'];matched+=1
    result.update(source_hashes_verified=True,physical_replay_decisions=decisions,comparisons=by_game,historical_traces_matched=matched)
    (root/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(comparisons=by_game,historical_traces_matched=matched),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--previous');a=p.parse_args()
    audit(Path(a.root),None if a.previous is None else Path(a.previous))
