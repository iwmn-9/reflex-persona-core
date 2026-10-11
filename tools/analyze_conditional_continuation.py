"""Independent world ledgers, full frozen replay and matched outcome comparisons."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def seal(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(x):return hashlib.sha256(seal(x).encode()).hexdigest()
def stable(row):return {k:v for k,v in row.items() if k not in ('game_seconds','planner_seconds')}


def replay(job):
    root,identity,expected=job;sys.path.insert(0,str(Path(root)/'_source'))
    from tools.run_conditional_continuation import run_game
    row=run_game(identity)
    assert digest(stable(row))==expected,'full frozen episode/decision/world receipt mismatch'
    return len(row['trace'])


def main(a):
    root=Path(a.root);assert (root/'completed.json').exists(),'incomplete study'
    sys.path.insert(0,str(root/'_source'))
    from tools.analyze_team_cooperation import check_projects,check_combat
    from reflex.core import Policy,TRAITS,VALUES
    from reflex.laboratory import PROFILES
    from tools.run_team_cooperation import ROSTERS
    from reflex.combat import battle_from_record,victory
    from reflex.combat_planning import mission
    from reflex.team_projects import Workshop,Worker,purpose,winners
    hashes=json.loads((root/'source_snapshot.json').read_text(encoding='utf-8'))
    for name,value in hashes.items():assert hashlib.sha256((root/'_source'/name).read_bytes()).hexdigest()==value
    protocol=json.loads((root/'preregister.json').read_text(encoding='utf-8'));checks=Counter();rows=[];jobs=[]
    for line in (root/'trajectories.jsonl').open(encoding='utf-8'):
        row=json.loads(line);identity=tuple(row[k] for k in ('genre','scenario','roster','seed','arena','mode'))
        jobs.append((str(root),identity,digest(stable(row))))
        prior=row['initial'];memories={};own=[i for i,p in enumerate(prior['units'] if row['genre']=='combat' else prior['people']) if p['team']==row['team']]
        first=None;errors=[]
        for index,turn in enumerate(row['trace']):
            assert seal(turn['before'])==seal(prior);choices={int(k):v for k,v in turn['choices'].items()}
            if row['genre']=='combat':check_combat(prior,choices,turn['after'],turn['world_audit'],int(digest(['team-cooperation-world',row['seed']])[:16],16))
            else:check_projects(prior,choices,turn['after'],turn['world_audit'])
            checks['independent_world_turns']+=1;checks['actual_intents']+=len(choices)
            for actor,c,d in zip(turn['signed'],turn['contexts'],turn['decisions']):
                profile=PROFILES[ROSTERS[row['roster']][own.index(actor)%3] if len(own)==3 else (own.index(actor)+row['roster'])%4]
                assert c['personality']==dict(zip(TRAITS,profile['traits']))
                assert c['values']=={k:float(profile['values'].get(k,0)) for k in VALUES}
                assert c['tick']==prior['tick'] and c['scope']['npc']==('unit-' if row['genre']=='combat' else 'worker-')+str(actor)
                assert d['context_hash']==digest(c) and d['action_id']==choices[actor]
                assert d['next_state']['intent_action']==d['action_id']
                if actor in memories:assert c['state']==memories[actor]
                memories[actor]=d['next_state'];checks['owner_receipts']+=1
            assert [Policy(principle_priority='finite').choose(c,False) for c in turn['contexts']]==turn['independent']
            n=turn['negotiation']
            if n and n['adopted']:
                for owner,regret in n['member_regrets'].items():assert regret<=n['concessions'][owner]+1e-12
                checks['bounded_consent_receipts']+=1
                actual=row['trace'][min(index+5,len(row['trace'])-1)]['after']
                value=mission(battle_from_record(actual),row['team']) if row['genre']=='combat' else purpose(Workshop(**{**actual,
                    'people':tuple(Worker(**p) for p in actual['people']),'stock':tuple(actual['stock']),'points':tuple(actual['points'])}),row['team'])
                errors.append(n['selected_purpose']-value)
            if index==0:first=[choices[i] for i in own]
            prior=turn['after']
        assert seal(prior)==seal(row['final'])
        ws=victory(battle_from_record(prior)) if row['genre']=='combat' else winners(Workshop(**{**prior,
            'people':tuple(Worker(**p) for p in prior['people']),'stock':tuple(prior['stock']),'points':tuple(prior['points'])}))
        credit=1/len(ws) if row['team'] in ws else 0. if ws else .5
        assert credit==row['win_credit'];assert len(row['trace'])==row['ticks']
        rows.append(dict(**{k:v for k,v in row.items() if k!='trace'},first=first,
            selected_forecast_errors=errors,distinct_world_history=digest([t['after'] for t in row['trace']])))
    assert len(rows)==len(protocol['jobs'])*len(protocol['modes'])
    if a.replay:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            tasks=[pool.submit(replay,j) for j in jobs]
            for i,task in enumerate(as_completed(tasks),1):
                checks['frozen_full_episode_turns']+=task.result()
                if i%24==0 or i==len(tasks):print(f'full replay {i}/{len(tasks)}',flush=True)
    groups=[];by={}
    for r in rows:by.setdefault(tuple(r[k] for k in ('genre','scenario','roster','seed','arena')), {})[r['mode']]=r
    for genre in sorted({r['genre'] for r in rows}):
        for arena in sorted({r['arena'] for r in rows}):
            selected=[r for r in rows if r['genre']==genre and r['arena']==arena]
            if not selected:continue
            arms={}
            for mode in protocol['modes']:
                rs=[r for r in selected if r['mode']==mode];errors=[e for r in rs for e in r['selected_forecast_errors']]
                work=Counter();[work.update(r['work']) for r in rs]
                per_profile={str(p):float(np.mean([r['win_credit'] for r in rs if r['roster']==p])) for p in sorted({r['roster'] for r in rs})}
                arms[mode]=dict(games=len(rs),mean_credit=float(np.mean([r['win_credit'] for r in rs])),
                    won=sum(r['won'] for r in rs),lost=sum(r['lost'] for r in rs),mean_health=float(np.mean([r['health'] for r in rs])),
                    distinct_world_histories=len({r['distinct_world_history'] for r in rs}),by_roster=per_profile,work=dict(work),
                    selected_horizon_mean_error=float(np.mean(errors)) if errors else None,
                    selected_horizon_mae=float(np.mean(np.abs(errors))) if errors else None)
                first_by=defaultdict(set)
                for r in rs:first_by[(r['scenario'],r['seed'])].add(tuple(r['first']))
                arms[mode]['first_roster_choices_mean']=float(np.mean([len(v) for v in first_by.values()]))
                if genre=='projects':
                    for field in ('energy','skill','prestige'):
                        arms[mode]['mean_personal_'+field]=float(np.mean([p[field] for r in rs for p in r['final']['people'] if p['team']==r['team']]))
            comparisons=[]
            contrasts=[(m,'bargain_uncertain') for m in protocol['modes'] if m!='bargain_uncertain']
            if 'bargain_exchange' in protocol['modes'] and 'bargain_personal' in protocol['modes']:contrasts.append(('bargain_exchange','bargain_personal'))
            for candidate,reference in contrasts:
                paired=[(identity[3],arms[candidate]['win_credit']-arms[reference]['win_credit']) for identity,arms in by.items() if identity[0]==genre and identity[4]==arena]
                seeds=sorted({s for s,d in paired});means=np.array([np.mean([d for ss,d in paired if ss==s]) for s in seeds])
                rng=np.random.default_rng(98001011);boot=means[rng.integers(len(means),size=(10000,len(means)))].mean(1)
                comparisons.append(dict(candidate=candidate,reference=reference,paired_cases=len(paired),seed_clusters=len(seeds),
                    delta=float(means.mean()),bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
                    better=sum(d>0 for s,d in paired),same=sum(d==0 for s,d in paired),worse=sum(d<0 for s,d in paired)))
            groups.append(dict(genre=genre,arena=arena,arms=arms,comparisons=comparisons))
    result=dict(games=len(rows),checks=dict(checks),groups=groups,
        source_snapshot_sha256=hashlib.sha256((root/'source_snapshot.json').read_bytes()).hexdigest(),
        trajectories_sha256=hashlib.sha256((root/'trajectories.jsonl').read_bytes()).hexdigest(),
        limits=['seed intervals conditional on the authored matrix; deterministic histories and repeated rosters are not independent worlds',
            'selected forecast error conditions on different chosen roots/trajectories; not a matched causal prediction benchmark',
            'solo mission health proxy still uses original normalization; no solo-specific value retuning',
            'full policy replay and independent world ledgers verify execution, not human-level intelligence'])
    (root/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--replay',action='store_true');p.add_argument('--workers',type=int,default=4)
    main(p.parse_args())
