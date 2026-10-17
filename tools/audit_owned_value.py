"""Recompute physical trajectories and compare whole persistent worlds."""
import argparse,json,sys,hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import digest
from reflex.laboratory import PROFILES
from reflex.decision_loop import DecisionLoop
from reflex.model_observer import resolve_feedback
from tools.continuing_games import ContinuingGame
from tools.run_owned_value import law,sample


def audit(root,*,allow_abandoned=False,allow_superseded=False):
    rows=[json.loads(s) for s in (root/'trajectories.jsonl').read_text().splitlines()]
    reg=json.loads((root/'registration.json').read_text())
    planned=len(reg['jobs'])*len(reg['arms']);complete=len(rows)==planned
    status=None
    if not complete:
        assert allow_abandoned or allow_superseded,'incomplete registered study'
        status=json.loads((root/'run_status.json').read_text(encoding='utf-8-sig'))
        if allow_superseded:
            assert status['status']=='superseded_by_matched_platform_study' and status['replacement_run']
        else:
            assert reg.get('stage')=='development' and status['status']=='abandoned_after_development_regression'
        assert status['completed_rows']==len(rows) and status['planned_rows']==planned and status['reason']
    expected={(g,PROFILES[p]['id'],s,r,a) for g,p,s,r,*_ in reg['jobs'] for a in reg['arms']}
    actual=[tuple(row[k] for k in ('game','profile','seed','regime','arm')) for row in rows]
    assert len(set(actual))==len(actual) and (set(actual)==expected if complete else set(actual)<expected),'duplicate or unregistered conditions'
    snapshot=json.loads((root/'source_snapshot.json').read_text())
    assert all(hashlib.sha256((root/'_source'/name).read_bytes()).hexdigest()==sha for name,sha in snapshot.items())
    checks=Counter();brief=[]
    for r in rows:
        profile=next(p for p in PROFILES if p['id']==r['profile'])
        g=ContinuingGame(r['game'],profile,r['turns'],r['seed']%2);w=g.start();rng=np.random.default_rng(r['seed'])
        assert len(r['trace'])==r['executed']
        for t in r['trace']:
            assert t['before']==w and t['tick']==w['tick']
            truth=law(t['tick'],r['turns'],r['regime'])
            assert t['truth']==truth and t['response']==sample(truth,rng.random())
            checks['world_rng_replay']+=1
            after,flow=g.transition(w,t['action'],t['response'])
            assert after==t['after'] and g.goal(after).value==t['goal']
            if t.get('planned') and t['forecast']:
                assert t['predicted']==t['forecast'][t['action']]
            if 'checkpoint' in t:
                c=g.context(w);template=g.actor(c,learned=r['arm']!='endpoint_library',renewal=r['arm'] in ('endpoint_renewal','endpoint_both','endpoint_recovery'),
                    learning='segmented' if r['arm'] in ('endpoint_segmented','endpoint_posterior','endpoint_persistent','reflex_segmented') else 'windows')
                own=DecisionLoop.from_record(c,t['checkpoint'],predictor=template.predictor)
                assert digest(own.record())==t['before_owner']
                req=g.request(c);result=DecisionLoop.decide_batch([(own,req)],False,_counterfactual_roots=(t['action'],))[0]
                resolve_feedback(own,result['ticket'],g.feedback(w,after,t['action'],c,flow))
                assert digest(own.record())==t['owner'],'stored actual action/feedback replay drift'
                checks['restored_action_feedback']+=1
            w=after;checks['physical_replay']+=1
        assert g.terminal(w) and w==r['final'] and g.goal(w).value==r['score']
        assert r['checks']['owner_parity']==r['executed']==r['checks']['fixed_axes']
        for k,v in r['checks'].items():checks[k]+=v
        planned=[t for t in r['trace'] if t.get('forecast_error') is not None]
        brief.append({**{k:v for k,v in r.items() if k not in ('trace','seconds')},
            'actions':dict(Counter(t['action'] for t in r['trace'])),
            'forecast_mae':float(np.mean([abs(t['forecast_error']) for t in planned])) if planned else None,
            'forecast_bias':float(np.mean([t['forecast_error'] for t in planned])) if planned else None,
            'blocks':[dict(start=i,goal_gain=r['trace'][min(i+32,len(r['trace']))-1]['goal']-
                (r['trace'][i-1]['goal'] if i else g.goal(g.start()).value),
                mean_brier=float(np.mean([t['brier'] for t in r['trace'][i:i+32]]))) for i in range(0,len(r['trace']),32)]})
    groups=defaultdict(dict)
    for r in brief:groups[(r['game'],r['profile'],r['seed'],r['regime'])][r['arm']]=r
    raw_groups=defaultdict(dict)
    for r in rows:raw_groups[(r['game'],r['profile'],r['seed'],r['regime'])][r['arm']]=r
    for arms in raw_groups.values():
        if 'reflex' not in arms:continue
        for arm in ('flow','endpoint'):
            if arm not in arms:continue
            for left,right in zip(arms['reflex']['trace'],arms[arm]['trace']):
                assert left['response']==right['response'] and left['law']==right['law']
                checks['matched_observation_prediction']+=1
    comparisons=[]
    for key,arms in groups.items():
        if 'endpoint' not in arms:continue
        for baseline in ('reflex','flow','endpoint_library','short_endpoint'):
            if baseline not in arms:continue
            delta=arms['endpoint']['score']-arms[baseline]['score']
            comparisons.append(dict(game=key[0],profile=key[1],seed=key[2],regime=key[3],baseline=baseline,
                delta=delta,better=int(delta>1e-12),same=int(abs(delta)<=1e-12),worse=int(delta< -1e-12)))
    summaries=[]
    for base in sorted({p['baseline'] for p in comparisons}):
        for game in (*sorted({p['game'] for p in comparisons}),'all'):
            ps=[p for p in comparisons if p['baseline']==base and (game=='all' or p['game']==game)]
            summaries.append(dict(baseline=base,game=game,pairs=len(ps),
                **{x:sum(p[x] for p in ps) for x in ('better','same','worse')},
                mean_delta=float(np.mean([p['delta'] for p in ps])),worst=min(p['delta'] for p in ps)))
    result=dict(source=root.name,source_freeze_verified=True,checks=dict(checks),runs=brief,comparisons=comparisons,summaries=summaries,
        trajectory_sha256=hashlib.sha256((root/'trajectories.jsonl').read_bytes()).hexdigest(),
        complete=complete,planned_runs=planned,completed_runs=len(rows),early_stop=status)
    (root/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(checks=result['checks'],summaries=summaries),indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--allow-abandoned',action='store_true')
    p.add_argument('--allow-superseded',action='store_true')
    a=p.parse_args();audit(Path(a.root),allow_abandoned=a.allow_abandoned,allow_superseded=a.allow_superseded)
