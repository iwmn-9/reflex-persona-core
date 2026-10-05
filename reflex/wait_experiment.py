"""Meaningful stops versus unsupported repetition; prospective fixed candidate."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from pathlib import Path
import json
import time
from .core import digest
from .progress import ProgressWatch,ProgressConfig
from .progress_adapters import combat_request,combat_feedback,economy_request,economy_feedback
from .laboratory import profiles
from .validation_experiment import combat,resources,auction,gambling,replay,replay_progress
from .combat_planning import TacticalControl
from .resource_planning import EconomyControl
from .purpose_experiment import source_hashes
from .cross_games import core_hashes


def diagnose(r,p):
    """Same declared purpose meanings for both variants, independent of scoring."""
    watches={};runs={};unresolved_runs={};totals=Counter();longest=0;unresolved_longest=0;previous={}
    for tick,row in enumerate(r['trace']):
        if r['genre']=='combat':
            from .combat import battle_from_record,alive,make_context
            before=battle_from_record(row['before']);after=battle_from_record(row['after'])
            actors=alive(before,r['seed']%2)
            records=[]
            for i in actors:
                c=make_context(before,i,p,r['seed'],'eliminate',survival_security=True)
                key=row['choices'].get(str(i),row['choices'].get(i))
                records.append((i,c,key,combat_request(before,i,c),combat_feedback(before,after,i,key)))
        elif r['genre']=='resources':
            from .resource_world import world_from_record
            from .route_experiment import routed_context
            before=world_from_record(row['before']);after=world_from_record(row['after'])
            if before.turn!=r['seed']%4:continue
            i=before.turn;c,_=routed_context(before,p,r['seed'],tick,None,row['selected_route'])
            key=row['action'];records=[(i,c,key,economy_request(before,c,previous.get(i)),economy_feedback(before,after,i,key))]
            previous[i]=after
        else:return {}
        for i,c,key,request,feedback in records:
            evidence=next((e for e in row.get('purpose',()) if e['scope']==c['scope']),{})
            config=ProgressConfig(**evidence['before']['config']) if evidence else ProgressConfig()
            w=watches.setdefault(i,ProgressWatch(c['scope'],config))
            _,raw=w.mask(c,request)
            _,audit=w.mask(c,request,evidence.get('recovery'))
            unsupported=audit['applied'] and key in audit['blocked']
            totals['unsupported_root_selections']+=int(unsupported)
            totals['no_supported_alternative']+=int(audit['unresolved'])
            a=request.activities[key]
            if a.kind in ('wait','idle','maintain'):
                totals['expired_stop_selections']+=int(key in raw['blocked'])
                totals['unsupported_stop_selections']+=int(unsupported)
                totals['supported_stop_selections']+=int(not unsupported and not audit['unresolved'])
                totals['future_gain_unproven_stops']+=int(key in raw['blocked'] and audit.get('recovery',{}).get('approved') is False)
            unresolved_stop=audit['unresolved'] and key in audit['blocked'] and a.kind in ('wait','idle','maintain')
            totals['unresolved_stop_selections']+=int(unresolved_stop)
            unresolved_runs[i]=unresolved_runs.get(i,0)+1 if unresolved_stop else 0
            unresolved_longest=max(unresolved_longest,unresolved_runs[i])
            runs[i]=runs.get(i,0)+1 if unsupported else 0;longest=max(longest,runs[i])
            w.observe(c['tick'],key,request,feedback)
    return dict(totals,longest_unsupported_run=longest,longest_unresolved_stop_run=unresolved_longest)


def run_job(job):
    genre,scene,persona,seed,variant,rival=job;p=next(x for x in profiles() if x['id']==persona);fixed=json.dumps(p,sort_keys=True)
    guarded=variant=='purposeful'
    if genre=='combat':
        terrain,goal=scene.split('/')
        r=combat(p,seed,terrain,goal,'baseline',learn=True,survival_security=True,rival=rival,
            planner=TacticalControl(),progress_watch=guarded)
    elif genre=='resources':
        r=resources(p,seed,scene,'baseline',planner=EconomyControl(),progress=guarded)
    elif genre in ('auction','hagetaka'):r=auction(p,seed,scene,'baseline',game=genre)
    else:r=gambling(p,seed,scene,'baseline')
    assert fixed==json.dumps(p,sort_keys=True)
    r.update(variant=variant,rival=rival)
    r.update(diagnose(r,p))
    return r,replay(r),replay_progress(r)


def jobs(seeds):
    from .combat import GOALS
    from .resource_experiment import SCENARIOS as RESOURCE
    from .contests_experiment import SCENARIOS as CONTEST
    from .gambling import SCENARIOS as GAMBLING
    work=[]
    for terrain in ('open','choke'):
        for goal in GOALS:
            for p in profiles():
                for seed in seeds:
                    for rival in ('reference','raider','switch'):
                        for variant in ('deliberation','purposeful'):
                            work.append(('combat',terrain+'/'+goal,p['id'],seed,variant,rival))
    for scene in RESOURCE:
        for p in profiles():
            for seed in seeds:
                for v in ('deliberation','purposeful'):work.append(('resources',scene,p['id'],seed,v,'game-owned'))
    for genre,scenes in (('auction',CONTEST),('hagetaka',CONTEST),('gambling',GAMBLING)):
        for scene in scenes:
            for p in profiles():
                for seed in seeds:work.append((genre,scene,p['id'],seed,'control','game-owned'))
    return work


def experiment(output,seeds=(150,151),workers=4,progress=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);frozen=source_hashes();core=core_hashes()
    work=jobs(seeds);results={};transitions=0;checks=0;started=time.perf_counter()
    with (output/'trajectories.jsonl').open('w',encoding='utf-8') as stream,ProcessPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(run_job,j):i for i,j in enumerate(work)}
        for future in as_completed(pending):
            r,count,verified=future.result();transitions+=count;checks+=verified
            stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush();results[pending[future]]={k:v for k,v in r.items() if k!='trace'}
            if len(results)%16==0 and progress:progress(f'{len(results)}/{len(work)} games; {transitions} rule transitions, {checks} purpose updates replayed',flush=True)
    assert frozen==source_hashes() and core==core_hashes()
    return finalize(output,[results[i] for i in range(len(work))],transitions,checks,frozen,core,seeds,time.perf_counter()-started)


def finalize(output,runs,transitions,checks,frozen,core,seeds,elapsed):
    summary=[];pairs=[]
    for genre in ('combat','resources','auction','hagetaka','gambling'):
        for variant in ('deliberation','purposeful','control'):
            for p in profiles():
                group=[r for r in runs if (r['genre'],r['variant'],r['profile'])==(genre,variant,p['id'])]
                if not group:continue
                keys=('won','lost','unsupported_root_selections','unsupported_stop_selections','expired_stop_selections','future_gain_unproven_stops','supported_stop_selections','unresolved_stop_selections','no_supported_alternative','own_proposal_collisions','focal_failed_moves','shortages','negative_ev_bets','ruined')
                summary.append(dict(genre=genre,variant=variant,profile=p['id'],games=len(group),
                    longest_unsupported_run=max(r.get('longest_unsupported_run',0) for r in group),
                    longest_unresolved_stop_run=max(r.get('longest_unresolved_stop_run',0) for r in group),
                    **{k:sum(r.get(k,0) for r in group) for k in keys}))
        old={(r['scenario'],r['profile'],r['seed'],r['rival']):r for r in runs if (r['genre'],r['variant'])==(genre,'deliberation')}
        for rival in sorted({r['rival'] for r in runs if r['genre']==genre}):
            group=[r for r in runs if (r['genre'],r['variant'],r['rival'])==(genre,'purposeful',rival)]
            if not group:continue
            combined=[(old[(b['scenario'],b['profile'],b['seed'],rival)],b) for b in group]
            pairs.append(dict(genre=genre,rival=rival,pairs=len(combined),
                better=sum(b['win_credit']>a['win_credit'] for a,b in combined),
                worse=sum(b['win_credit']<a['win_credit'] for a,b in combined),
                same=sum(b['win_credit']==a['win_credit'] for a,b in combined)))
    gates={}
    for genre in ('combat','resources'):
        old={r['profile']:r for r in summary if (r['genre'],r['variant'])==(genre,'deliberation')}
        new={r['profile']:r for r in summary if (r['genre'],r['variant'])==(genre,'purposeful')}
        no_drop=all(new[k]['won']>=a['won'] for k,a in old.items())
        no_family=all(p['better']>=p['worse'] for p in pairs if p['genre']==genre)
        no_unsupported=all(b['unsupported_root_selections']==0 for b in new.values())
        gates[genre]=dict(no_persona_win_drop=no_drop,no_opponent_family_regression=no_family,
            no_unsupported_selected_roots=no_unsupported,passed=no_drop and no_family and no_unsupported)
    result=dict(format='purposeful-wait-v1',games=len(runs),seeds=list(seeds),seed_status='92 development; 100/101 first confirmation failed then development; 110/111 interrupted for preparation counterexample; 120/121 interrupted for coupled-margin logic error; 130/131 interrupted for inter-turn preparation counterexample; 140-143 four-seat integration; 150/151 reserved before final fixed-candidate confirmation',
        source_hashes=frozen,core_hashes=core,elapsed_seconds=elapsed,rule_transitions_replayed=transitions,purpose_updates_replayed=checks,
        summary=summary,comparisons=pairs,quality_gate=gates,runs=runs,performance_gate=False,defaults_changed=False,
        acceptance_note='Win-retention checks diagnose competence regressions; persona/environment compatibility may yield legitimate win differences. Wins alone do not establish rationality, personality fidelity or acceptance.',
        limitations=['authored goal/readiness/activity meanings, not an automatic universal purpose detector',
            'bounded wait proof is game-provided expectation, not guaranteed future progress',
            'constraints narrow available means, never change trait/value numbers; unsupported alternative fallback may remain unresolved',
            'actual purpose progress is separate from immediate outcome-vector learning',
            'uncertain useful attempts are not condemned for stochastic failures',
            'two seeds and small authored games, not proof of general intelligence or human-like consistency'])
    (output/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    lines=['# 目的のある停止と無意味な停滞','',f'{len(runs)}局、seed {list(seeds)}、実遷移{transitions}と目的更新{checks}を再生。','',
        '|ゲーム|方式|人格|勝利/局数|静的診断で期限到達した停止|診断上有効な停止|支持のある代案なし|未解決の停止|最長の未解決停止|',
        '|---|---|---|---|---|---|---|---|---|']
    for r in summary:lines.append(f"|{r['genre']}|{r['variant']}|{r['profile']}|{r['won']}/{r['games']}|{r['expired_stop_selections']}|{r['supported_stop_selections']}|{r['no_supported_alternative']}|{r['unresolved_stop_selections']}|{r['longest_unresolved_stop_run']}|")
    lines+=['',f'保守的な回帰診断: {gates}。勝率の維持は採否の唯一の条件ではない。人格と環境/相手の相性による差を均等化せず、判断の不具合と目的/人格に沿った負けを区別する必要がある。','',
        '得たもの: 目的/準備の実観測、有限の待機猶予、解除条件、過去最大を超えない往復を進展と扱わない検査。共同案を辞退しても単独反射と未来選択に同じ進展制約を適用。','',
        '制限したもの: 実際の進展がない待ちの予測更新だけによる延長と、同じ条件の決定的失敗の反復。目的/準備を進める待ち、実際に役立つ維持、確率上有効な試行は残す。固定人格/価値は変更しないが、制約前の全手段に対する最大主義の選択自由は狭まる。','',
        '限界: 診断の意味はゲームが供給する。危険しかない/支持された手がない局面では無理に動かさずunresolvedと記録する。今回の診断で0という結果も、全ゲームの停滞ゼロや最適判断の保証ではない。遅延成果による未来モデル学習は追加していない。','',
        '競り/ハゲタカ/ギャンブルは既存経路の対照で、新しい待機adapterは接続していない。速度/費用は合否にせず、Colab/GPU/Drive/LLM/モデル訓練は使わない。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    return result
