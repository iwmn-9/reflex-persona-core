"""Independent physical replay and paired summaries; never chooses an action."""
import argparse, hashlib, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.laboratory import PROFILES
from tools.run_prediction_recovery import RecoveryAdapter, truth, sample, summary


def audit(root):
    hashes=json.loads((root/'source_snapshot.json').read_text())
    for name,sha in hashes.items():assert hashlib.sha256((root/'_source'/name).read_bytes()).hexdigest()==sha,name
    reg=json.loads((root/'registration.json').read_text())
    rows=[json.loads(line) for line in (root/'trajectories.jsonl').read_text().splitlines()]
    expected={(g,PROFILES[p]['id'],s,r,arm) for g,p,s,r,n,pe in reg['jobs'] for arm in reg['arms']}
    actual={(r['game'],r['profile'],r['seed'],r['regime'],r['arm']) for r in rows}
    assert len(rows)==len(expected) and expected==actual
    by_profile={p['id']:p for p in PROFILES};count=0;phases=defaultdict(list)
    for r in rows:
        a=RecoveryAdapter(r['game'],by_profile[r['profile']],r['seed']%2);s=a.start();scores=[]
        uniforms=np.random.default_rng(r['seed']).random(r['turns'])
        assert len(r['trace'])==r['turns']
        for tick,t in enumerate(r['trace']):
            if a.terminal(s):scores.append(a.goal(s));s=a.start(s['round']+1,tick)
            assert t['tick']==tick and t['round']==s['round'] and t['action'] in a.keys(s)
            p=truth(a.symbols(),tick,r['turns'],r['regime'])
            s,flow,symbol=a.transition(s,t['action'],sample(p,uniforms[tick]))
            assert symbol==t['response'] and a.goal(s)==t['goal']
            assert abs(sum(t['law'].values())-1)<1e-12
            phases[(r['game'],r['regime'],r['arm'],min(3,tick*4//r['turns']))].append(t['expected_brier'])
            count+=1
        if a.terminal(s):scores.append(a.goal(s))
        assert scores==r['scores'] and float(np.mean(scores))==r['score']
    result=summary(rows);assert result==json.loads((root/'summary.json').read_text())
    comparisons=[]
    for game in ('all','auction','resources','combat','gambling'):
        pairs=[p for p in result['pairs'] if game=='all' or p['game']==game]
        if not pairs:continue
        for base in reg['arms']:
            if base=='recovery':continue
            for metric in ('score','expected_brier','immediate_persona_regret'):
                delta=[p[metric]['recovery']-p[metric][base] for p in pairs]
                sign=1 if metric=='score' else -1
                comparisons.append(dict(game=game,baseline=base,metric=metric,pairs=len(delta),
                    better=sum(sign*d>1e-12 for d in delta),equal=sum(abs(d)<=1e-12 for d in delta),
                    worse=sum(sign*d< -1e-12 for d in delta),mean_delta=float(np.mean(delta)),
                    worst_delta=min(delta) if sign==1 else max(delta)))
    result.update(physical_replay_decisions=count,source_hashes_verified=True,comparisons=comparisons,
        phase_error=[dict(game=k[0],regime=k[1],arm=k[2],phase=k[3],expected_brier=float(np.mean(v))) for k,v in sorted(phases.items())])
    (root/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(decisions=count,comparisons=[x for x in comparisons if x['game']=='all']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();audit(Path(a.root))
