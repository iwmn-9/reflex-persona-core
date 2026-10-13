"""Count registered episodes once, separating pre-settled setup from AI play."""
import argparse,json,hashlib
from pathlib import Path
from collections import Counter,defaultdict


def summarize(rows,stage):
    active=[r for r in rows if r['trace']];setup=[r for r in rows if not r['trace']]
    result=dict(stage=stage,registered_executions=len(rows),owner_active_episodes=len(active),setup_only_episodes=len(setup),
        owner_decisions=sum(len(r['trace']) for r in rows),setup_only_by_game=dict(Counter(r['case']['game'] for r in setup)),
        counting='setup may already settle; those rows are preserved but excluded from personality/strength comparisons; repeated personas are not independent worlds')
    if stage=='train':
        result.update(training_rows=sum(len(r['features']) for r in rows),unique_observed_episodes=len({r['episode'] for r in active}),groups=[])
        for game in sorted({r['case']['game'] for r in rows}):
            rs=[r for r in active if r['case']['game']==game]
            result['groups'].append(dict(game=game,episodes=len(rs),unique_episodes=len({r['episode'] for r in rs}),states=sum(len(r['features']) for r in rs),statuses=dict(Counter(r['goal']['status'] for r in rs))))
        return result
    key=lambda r:(json.dumps(r['case'],sort_keys=True),r['profile'],r['seed'])
    grouped=defaultdict(dict)
    for r in rows:
        if r['arm'] in grouped[key(r)]:raise AssertionError('duplicate scheduled arm')
        grouped[key(r)][r['arm']]=r
    assert all(set(g)=={'proxy','memory','tail'} for g in grouped.values())
    result['comparison']=[]
    for game in sorted({r['case']['game'] for r in rows}):
        for arm in ('memory','tail'):
            pairs=[];excluded=0
            for g in grouped.values():
                r=g[arm];old=g['proxy']
                if r['case']['game']!=game:continue
                assert r['prefix']==old['prefix']
                if not old['trace'] or not r['trace']:
                    assert not old['trace'] and not r['trace'];excluded+=1;continue
                delta=r['goal']['value']-old['goal']['value'];a=r['mean_terminal_crps'];b=old['mean_terminal_crps']
                pairs.append(dict(case=r['case'],profile=r['profile'],seed=r['seed'],before=old['goal']['value'],after=r['goal']['value'],delta=delta,
                    actions_changed=r['actions']!=old['actions'],visited_policy_crps_delta=None if a is None or b is None else a-b))
            result['comparison'].append(dict(game=game,arm=arm,pairs=len(pairs),setup_pairs_excluded=excluded,
                better=sum(p['delta']>1e-12 for p in pairs),worse=sum(p['delta']< -1e-12 for p in pairs),same=sum(abs(p['delta'])<=1e-12 for p in pairs),paired=pairs))
    result['statuses']=[dict(game=game,arm=arm,counts=dict(Counter(r['goal']['status'] for r in active if r['case']['game']==game and r['arm']==arm)))
        for game in sorted({r['case']['game'] for r in rows}) for arm in ('proxy','memory','tail')]
    result['declines']=sum(not t['tree']['complete'] for r in active for t in r['trace'])
    result['frontiers']={arm:dict(evaluated=sum(t['frontier_count'] for r in active if r['arm']==arm for t in r['trace']),
        supported=sum(t['supported'] for r in active if r['arm']==arm for t in r['trace']),settled_fraction_sum=sum(t['settled_fraction_sum'] for r in active if r['arm']==arm for t in r['trace'])) for arm in ('memory','tail')}
    behavior=defaultdict(dict)
    for r in active:behavior[(json.dumps(r['case'],sort_keys=True),r['seed'],r['arm'])][r['profile']]=r['actions'][0]
    result['persona_roots']=[dict(arm=arm,worlds=sum(k[2]==arm and len(v)==4 for k,v in behavior.items()),
        differentiated=sum(k[2]==arm and len(v)==4 and len(set(v.values()))>1 for k,v in behavior.items())) for arm in ('proxy','memory','tail')]
    return result


def analyze(root):
    root=Path(root);pre=json.loads((root/'preregister.json').read_text(encoding='utf-8'));rows=[json.loads(s) for s in (root/'trajectories.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==len(pre['jobs']) and {r['index'] for r in rows}==set(range(len(pre['jobs']))),'complete unique registration coverage required'
    for r in rows:
        j=pre['jobs'][r['index']]
        assert r['case']==j[0] and r['profile']==j[1]['id'] and r['seed']==j[2] and r['replay']
        if pre['stage']!='train':assert r['arm']==j[3]
    result=summarize(rows,pre['stage']);result['trajectories_sha256']=hashlib.sha256((root/'trajectories.jsonl').read_bytes()).hexdigest()
    result['analysis_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (root/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();r=analyze(a.root);print(json.dumps({k:v for k,v in r.items() if k not in ('comparison','statuses')}))
