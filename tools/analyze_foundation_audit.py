"""Audit every preregistered job; separate score tradeoffs from missing foresight."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import digest


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tally(xs):
    return dict(better=sum(x>1e-12 for x in xs),worse=sum(x<-1e-12 for x in xs),
                same=sum(abs(x)<=1e-12 for x in xs))


def cohort(root):
    root=Path(root);pre=read(root/'preregister.json');evaluation=read(root/'evaluation.json')
    assert not evaluation['errors'] and evaluation['planned']==evaluation['episodes']==len(pre['jobs'])
    assert digest(pre)==evaluation['preregister_sha256']
    assert sha(root/'trajectories.jsonl')==evaluation['trajectories_sha256']
    assert sha(root/'source_snapshot.json')==pre['source_snapshot_sha256']
    hashes=read(root/'source_snapshot.json')
    assert all(sha(root/'_source'/p)==h for p,h in hashes.items())
    rows=[json.loads(t) for t in (root/'trajectories.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==len(pre['jobs']) and {r['index'] for r in rows}==set(range(len(rows)))
    summaries={r['index']:r for r in evaluation['runs']}
    for r in rows:
        assert r['job']==pre['jobs'][r['index']] and r['replay']
        assert all(summaries[r['index']][k]==v for k,v in r.items() if k!='trace')
        assert r['actions']==[t['root'] for t in r['trace']]
        for t in r['trace']:
            f=t['forecast']
            if f:
                assert t['root']==(f['purpose'] if r['arm']=='purpose4' else f['persona'])
                assert abs(f['planned_purpose_cost']-(max(f['predicted'].values())-f['predicted'][t['root']]))<1e-12
            if t['oracle_q']:
                assert abs(t['oracle_regret']-(max(t['oracle_q'].values())-t['oracle_q'][t['root']]))<1e-12
    groups=defaultdict(dict)
    for r in rows:groups[digest([r['case'],r['profile'],r['seed']])][r['arm']]=r
    pairs=[]
    for group in groups.values():
        assert len({digest(r['initial']) for r in group.values()})==1
        assert len({digest(r['prefix']) for r in group.values()})==1
        for old,new in (('reflex','persona4'),('persona4','purpose4'),('persona4','persona8'),('persona4','oracle')):
            if old not in group or new not in group:continue
            a,b=group[old],group[new]
            pairs.append(dict(case=a['case'],profile=a['profile'],seed=a['seed'],comparison=new+'-'+old,
                before=a['goal']['value'],after=b['goal']['value'],delta=b['goal']['value']-a['goal']['value'],
                actions_changed=a['actions']!=b['actions']))
    summary=[]
    for game in sorted({r['case']['game'] for r in rows}):
        for comparison in sorted({p['comparison'] for p in pairs}):
            ps=[p for p in pairs if p['case']['game']==game and p['comparison']==comparison]
            if not ps:continue
            summary.append(dict(game=game,comparison=comparison,pairs=len(ps),
                **tally([p['delta'] for p in ps]),actions_changed=sum(p['actions_changed'] for p in ps),
                by_persona={name:tally([p['delta'] for p in ps if p['profile']==name]) for name in sorted({p['profile'] for p in ps})}))
    arms=[]
    for game in sorted({r['case']['game'] for r in rows}):
        for arm in sorted({r['arm'] for r in rows}):
            rr=[r for r in rows if r['case']['game']==game and r['arm']==arm]
            ts=[t for r in rr for t in r['trace']];fs=[t['forecast'] for t in ts if t['forecast']]
            arms.append(dict(game=game,arm=arm,episodes=len(rr),steps=len(ts),
                status=dict(Counter(r['goal']['status'] for r in rr)),pre_settled=sum(r['pre_settled'] for r in rr),
                unique_initial_positions=len({digest(r['initial']) for r in rr}),
                unique_physical_trajectories=len({digest([[t['root'],t['after']] for t in r['trace']]) for r in rr}),
                persona_purpose_disagreements=sum(f['persona']!=f['purpose'] for f in fs),
                planned_purpose_cost=sum(f['planned_purpose_cost'] for f in fs),
                settled_model_fraction=None if not fs else sum(f['settled_branches'] for f in fs)/sum(f['branches'] for f in fs)))
    graph=[]
    if any(r['case']['game']=='public_graph' for r in rows):
        for length in sorted({r['case']['length'] for r in rows}):
            for arm in sorted({r['arm'] for r in rows}):
                rr=[r for r in rows if r['case']['length']==length and r['arm']==arm];ts=[t for r in rr for t in r['trace']]
                blank=wrong=0;opportunities=0
                for t in ts:
                    q=t['oracle_q'];best=max(q.values());spread=best-min(q.values())
                    opportunities+=spread>1e-12;f=t['forecast']
                    if not f:continue
                    v=f['predicted'];m=max(v.values());heads={k for k,x in v.items() if abs(x-m)<1e-12}
                    truth={k for k,x in q.items() if abs(x-best)<1e-12}
                    blank+=max(v.values())-min(v.values())<1e-12 and spread>1e-12
                    wrong+=not heads&truth
                graph.append(dict(length=length,arm=arm,episodes=len(rr),steps=len(ts),
                    wins=sum(r['goal']['status']=='success' for r in rr),
                    expected_optimal_wins=sum(r['oracle_initial'] for r in rr),
                    opportunity_decisions=opportunities,positive_regret_decisions=sum(t['oracle_regret']>1e-12 for t in ts),
                    mean_path_regret=sum(t['oracle_regret'] for t in ts)/len(rr),
                    flat_forecast_at_real_opportunity=blank,wrong_forecast_top=wrong))
    return dict(stage=pre['stage'],episodes=len(rows),steps=sum(len(r['trace']) for r in rows),errors=[],
        controls=dict(all_sources_verified=True,all_registered_jobs_present=True,all_actual_replays=True,
            source_snapshot_sha256=pre['source_snapshot_sha256'],trajectories_sha256=sha(root/'trajectories.jsonl'),
            preregister_sha256=digest(pre)),comparisons=summary,arms=arms,graphs=graph,pairs=pairs),rows


def exact_cohort(root):
    root=Path(root);pre=read(root/'preregister.json');result=read(root/'evaluation.json')
    assert not result['errors'] and result['complete']==result['planned']==len(pre['jobs'])
    assert result['preregister_sha256']==digest(pre) and result['policies_sha256']==sha(root/'policies.jsonl')
    assert pre['source_snapshot_sha256']==sha(root/'source_snapshot.json')
    assert all(sha(root/'_source'/p)==h for p,h in read(root/'source_snapshot.json').items())
    rows=[json.loads(t) for t in (root/'policies.jsonl').read_text(encoding='utf-8').splitlines()]
    assert {r['index'] for r in rows}==set(range(len(pre['jobs']))) and len(rows)==len(pre['jobs'])
    recorded={r['index']:r for r in result['runs']}
    for r in rows:
        assert r['job']==pre['jobs'][r['index']]
        assert {k:v for k,v in r.items() if k!='trace'}==recorded[r['index']]
        assert 0<=r['success_probability']<=r['optimal_probability']<=1
        assert abs(r['optimal_probability']-r['success_probability']-r['expected_opportunity_loss'])<1e-12
    arms=[]
    for length in sorted({r['length'] for r in rows}):
        for arm in sorted({r['arm'] for r in rows}):
            rr=[r for r in rows if r['length']==length and r['arm']==arm]
            arms.append(dict(length=length,arm=arm,policies=len(rr),
                mean_success_probability=sum(r['success_probability'] for r in rr)/len(rr),
                mean_optimal_probability=sum(r['optimal_probability'] for r in rr)/len(rr),
                mean_opportunity_loss=sum(r['expected_opportunity_loss'] for r in rr)/len(rr),
                suboptimal=sum(r['expected_opportunity_loss']>1e-12 for r in rr),
                by_persona={name:dict(probability=sum(r['success_probability'] for r in rr if r['profile']==name)/(len(rr)/4),
                    loss=sum(r['expected_opportunity_loss'] for r in rr if r['profile']==name)/(len(rr)/4))
                    for name in sorted({r['profile'] for r in rr})}))
    groups=defaultdict(dict)
    for r in rows:groups[r['graph'],r['length'],r['profile']][r['arm']]=r
    pairs=[]
    for key,g in groups.items():
        assert set(g)=={'reflex','persona4','purpose4'}
        a,b=g['persona4'],g['purpose4']
        aa={t['key']:t['action'] for t in a['trace']};bb={t['key']:t['action'] for t in b['trace']}
        pairs.append(dict(graph=key[0],length=key[1],profile=key[2],
            delta=b['success_probability']-a['success_probability'],all_reachable_choices_equal=aa==bb))
    return dict(policies=len(rows),owned_states=sum(r['unique_owned_states'] for r in rows),arms=arms,pairs=pairs,
        controls=dict(source_snapshot_sha256=pre['source_snapshot_sha256'],policies_sha256=result['policies_sha256'],
            preregister_sha256=digest(pre),all_sources_verified=True,exact_expectation_identity_verified=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--cohorts',nargs='*',default=[]);p.add_argument('--exact');p.add_argument('--output',required=True);a=p.parse_args()
    summaries=[]
    for path in a.cohorts:
        summary,_=cohort(path);summaries.append(summary)
    out=Path(a.output)
    if out.exists():raise FileExistsError('fresh analysis required')
    result=dict(format='foundation-audit-summary-v1',cohorts=summaries)
    if a.exact:result['exact']=exact_cohort(a.exact)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{k:r[k] for k in ('stage','episodes','steps','comparisons','graphs')} for r in summaries],ensure_ascii=False,indent=2))
    if a.exact:print(json.dumps(result['exact']['arms'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
