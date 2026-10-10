"""Paired learning-series analysis; individual goals, not an NPC team reward."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def load(p):return json.loads(p.read_text(encoding='utf-8'))


def analyze(root):
    root=Path(root);matches=[];runs=[];plans=[]
    for folder in sorted(root.glob('shard-*')):
        plan=load(folder/'analysis_preregister.json');plans.append(plan)
        for variant in plan['variants']:
            replay=load(folder/'source_replay_registration.json') if (folder/'source_replay_registration.json').exists() else None
            condition_folder=replay['primary_condition_folder'] if replay and replay['condition']==variant else variant
            file=folder/condition_folder/'evaluation.json';e=load(file);p=e['plan']
            assert e['plan']==load(file.parent/'preregister.json')
            if replay and replay['condition']==variant:
                proof=load(folder/'source_replay.json');assert proof['complete_trajectory_equal'] and proof['matches']==len(e['matches'])
                assert proof['trajectory_sha256']==e['trajectory_sha256']
            assert p['seeds']==plan['seeds'] and p['encounters']==plan['encounters']
            assert p['strong']==plan['strong'] and p['persona']==plan['persona'] and p['sources']==plan['sources']
            assert p['variant']==plan.get('controller_variants',{}).get(variant,plan.get('controller_variant','progress'))
            assert p['memory_kind']==('global' if variant=='global' else 'validated' if variant=='validated' else 'supported')
            runs.append(dict(folder=str(file.parent.relative_to(root)).replace('\\','/'),variant=variant,matches=len(e['matches']),
                evaluation_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),trajectory_sha256=e['trajectory_sha256']))
            matches.extend(dict(**{k:v for k,v in r.items() if k!='beliefs'},condition=variant) for r in e['matches'])
    assert len(plans)==4
    for plan in plans[1:]:
        for key in ('variants','encounters','games','sources','strong','persona'):assert plan[key]==plans[0][key]
    variants=plans[0]['variants'];games=plans[0]['games'];seeds=[s for p in plans for s in p['seeds']];encounters=plans[0]['encounters']
    assert len(seeds)==len(set(seeds))==16
    index={(r['condition'],r['game'],r['seed'],r['encounter']):r for r in matches}
    assert len(index)==len(matches)==len(variants)*len(games)*len(seeds)*encounters
    negative=0
    for game in games:
        for seed in seeds:
            for enc in range(encounters):
                a=index[variants[0],game,seed,enc];b=index[variants[1],game,seed,enc]
                assert a['benchmark']==b['benchmark'] and a['profiles']==b['profiles']
                assert a['final']['prizes' if game=='goofspiel' else 'seen']==b['final']['prizes' if game=='goofspiel' else 'seen']
                if game=='goofspiel':
                    for k in ('scores','credits','final','failures','reading_changes','guards'):assert a[k]==b[k]
                    negative+=1
    rng=np.random.default_rng(72190);summary=[];contrasts=[]
    for game in games:
        sign=1 if game=='goofspiel' else -1
        for variant in variants:
            rows=[r for r in matches if r['game']==game and r['condition']==variant];profiles={}
            for profile in ('growth','steady','care','ego'):
                observations=[(r,int(a)) for r in rows for a,p in r['profiles'].items() if p==profile]
                profiles[profile]=dict(appearances=len(observations),credit=sum(r['credits'][a] for r,a in observations),
                    credit_rate=float(np.mean([r['credits'][a] for r,a in observations])),mean_score=float(np.mean([r['scores'][a] for r,a in observations])),
                    pairwise_cpu_score_rate=float(np.mean([float(sign*r['scores'][a]>sign*r['scores'][r['benchmark']])+.5*float(r['scores'][a]==r['scores'][r['benchmark']]) for r,a in observations])))
            summary.append(dict(game=game,condition=variant,matches=len(rows),profiles=profiles,
                cpu_credit=sum(r['credits'][r['benchmark']] for r in rows),npc_sum_credit=sum(1-r['credits'][r['benchmark']] for r in rows),
                npc_failures=sum(sum(r['failures'])-r['failures'][r['benchmark']] for r in rows)))
        for profile in ('each_npc_mean','growth','steady','care','ego'):
            series=[];better=same=worse=0;score_series=[]
            for seed in seeds:
                diffs=[];scores=[]
                for enc in range(encounters):
                    a=index[variants[0],game,seed,enc];b=index[variants[1],game,seed,enc]
                    actors=[int(actor) for actor,p in a['profiles'].items() if p!='benchmark' and (profile=='each_npc_mean' or p==profile)]
                    if not actors:continue
                    delta=np.mean([b['credits'][actor]-a['credits'][actor] for actor in actors]);diffs.append(delta)
                    scores.append(float(np.mean([sign*(b['scores'][actor]-a['scores'][actor]) for actor in actors])))
                    better+=delta>0;same+=delta==0;worse+=delta<0
                if diffs:series.append(float(np.mean(diffs)));score_series.append(float(np.mean(scores)))
            values=np.array(series);sample=values[rng.integers(len(values),size=(10000,len(values)))].mean(1)
            contrasts.append(dict(game=game,profile=profile,independent_series=len(series),mean_credit_difference=float(values.mean()),
                paired_series_bootstrap_95=list(map(float,np.quantile(sample,[.025,.975]))),series_differences=series,
                mean_signed_score_difference=float(np.mean(score_series)),better=better,same=same,worse=worse))
    result=dict(shard_plans=plans,runs=runs,matches=matches,summary=summary,contrasts=contrasts,goof_negative_control_exact_pairs=negative,
        limitations=['independent units are learning series, not turns or all games','bootstrap intervals have no multiplicity correction',
            'fixed high-budget CPU, not proved optimal or human-calibrated','NPC summed credit descriptive; each NPC optimizes own purposes'])
    (root/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(summary=summary,contrasts=contrasts),indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();analyze(a.root)
