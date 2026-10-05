"""Fixed-candidate stability confirmation, with isolated CPU game workers.

No speed acceptance criterion. Real victory, persona retention, failed means,
and consistency of immediate versus future targets determine the verdict.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from dataclasses import asdict
import json
import time
from .laboratory import profiles
from .purpose_experiment import source_hashes
from .cross_games import core_hashes
from .validation_experiment import combat,resources,auction,gambling,replay
from .combat_planning import TacticalControl
from .resource_planning import EconomyControl
from .teachers import write_json


def run_job(job):
    genre,scene,persona,seed,variant,rival=job;p=next(p for p in profiles() if p['id']==persona)
    before=json.dumps(p,sort_keys=True)
    if genre=='combat':
        m,g=scene.split('/')
        r=combat(p,seed,m,g,'baseline',learn=variant!='baseline',survival_security=variant!='baseline',
            rival=rival,planner=TacticalControl() if variant=='deliberation' else None)
    elif genre=='resources':r=resources(p,seed,scene,'baseline',planner=EconomyControl() if variant=='deliberation' else None)
    elif genre in ('auction','hagetaka'):r=auction(p,seed,scene,'baseline',game=genre)
    else:r=gambling(p,seed,scene,'baseline')
    assert before==json.dumps(p,sort_keys=True)
    r.update(variant=variant,rival=rival)
    return r,replay(r)


def jobs(seeds):
    from .combat import GOALS
    from .contests_experiment import SCENARIOS as AUCTION
    from .resource_experiment import SCENARIOS as RESOURCE
    from .gambling import SCENARIOS as GAMBLING
    out=[]
    for rival in ('reference','raider','switch'):
        for terrain in ('open','choke'):
            for goal in GOALS:
                for p in profiles():
                    for seed in seeds:
                        for v in ('baseline','experience','deliberation'):out.append(('combat',terrain+'/'+goal,p['id'],seed,v,rival))
    for scene in RESOURCE:
        for p in profiles():
            for seed in seeds:
                for v in ('baseline','deliberation'):out.append(('resources',scene,p['id'],seed,v,'game-owned'))
    for genre,scenes in (('auction',AUCTION),('hagetaka',AUCTION),('gambling',GAMBLING)):
        for scene in scenes:
            for p in profiles():
                for seed in seeds:out.append((genre,scene,p['id'],seed,'baseline','game-owned'))
    return out


def experiment(output,seeds=(90,91),workers=4,progress=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);frozen=source_hashes();core=core_hashes()
    work=jobs(seeds);results={};checks=0;start=time.perf_counter()
    with (output/'trajectories.jsonl').open('w',encoding='utf-8') as stream,ProcessPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(run_job,job):i for i,job in enumerate(work)}
        for future in as_completed(pending):
            r,count=future.result();i=pending[future];checks+=count
            stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush()
            results[i]={k:v for k,v in r.items() if k!='trace'}
            if len(results)%16==0 and progress:progress(f'{len(results)}/{len(work)} games; {checks} real transitions replayed',flush=True)
    assert frozen==source_hashes() and core==core_hashes()
    return finalize(output,[results[i] for i in range(len(work))],checks,frozen,core,seeds,time.perf_counter()-start)


def finalize(output,runs,checks,frozen,core,seeds,elapsed):
    summary=[];comparisons=[]
    for genre in ('combat','resources','auction','hagetaka','gambling'):
        for variant in ('baseline','experience','deliberation'):
            for p in profiles():
                rows=[r for r in runs if (r['genre'],r['variant'],r['profile'])==(genre,variant,p['id'])]
                if not rows:continue
                totals={k:sum(r.get(k,0) for r in rows) for k in
                    ('won','lost','own_proposal_collisions','focal_failed_moves','planning_nodes','planned_ticks','planning_declines','negative_ev_bets','ruined','shortages')}
                summary.append(dict(genre=genre,variant=variant,profile=p['id'],games=len(rows),**totals))
        rows=[r for r in runs if r['genre']==genre]
        index={(r['scenario'],r['profile'],r['seed'],r['rival']):r for r in rows if r['variant']=='baseline'}
        for variant in ('experience','deliberation'):
            for rival in sorted({r['rival'] for r in rows}):
                group=[r for r in rows if r['variant']==variant and r['rival']==rival]
                if not group:continue
                pairs=[(index[(r['scenario'],r['profile'],r['seed'],rival)],r) for r in group]
                comparisons.append(dict(genre=genre,variant=variant,rival=rival,pairs=len(pairs),
                    better=sum(b['win_credit']>a['win_credit'] for a,b in pairs),
                    worse=sum(b['win_credit']<a['win_credit'] for a,b in pairs),
                    same=sum(b['win_credit']==a['win_credit'] for a,b in pairs)))
    verdict={}
    for genre in ('combat','resources'):
        old={s['profile']:s for s in summary if (s['genre'],s['variant'])==(genre,'baseline')}
        new={s['profile']:s for s in summary if (s['genre'],s['variant'])==(genre,'deliberation')}
        no_drop=all(new[k]['won']>=s['won'] for k,s in old.items())
        weak=min(old,key=lambda k:old[k]['won']/old[k]['games']);improved=new[weak]['won']>old[weak]['won']
        no_family_drop=all(c['better']>=c['worse'] for c in comparisons if (c['genre'],c['variant'])==(genre,'deliberation'))
        verdict[genre]=dict(no_persona_win_drop=no_drop,weakest_persona=weak,weakest_improved=improved,
            no_opponent_family_paired_regression=no_family_drop,passed=no_drop and improved and no_family_drop)
    result=dict(format='stability-validation-v1',games=len(runs),seeds=list(seeds),
        seed_status='90/91 reserved before this candidate confirmation; 60/61/80 development; no per-persona tuning',
        source_hashes=frozen,core_hashes=core,elapsed_seconds=elapsed,rule_transitions_replayed=checks,
        summary=summary,comparisons=comparisons,quality_gate=verdict,runs=runs,
        controls=dict(combat=asdict(TacticalControl()),resources=asdict(EconomyControl())),
        performance_gate=False,defaults_changed=False,
        contracts=['fixed personality/value axes and original shared Policy scoring; no persona handicaps',
            'joint forecasts have a separate modeled-horizon target; only selected actual immediate outcomes enter empirical memory',
            'same actual rules, seats, opponents and RNG in paired games; private planning RNG is disjoint',
            'purpose corridor is an explicit approximate competence constraint, not root action legality or psychological proof',
            'each actor strongest value tier must accept a joint plan; incompatible proposals are declined',
            'combat simulates joint turns/roles; resources simulates rivals and production between own turns',
            'no actual rival controller/current intent/private future event schedule in forecasts',
            'controls in auction/hagetaka/gambling retain existing current evidence path; no automatic improvement transfer claim'],
        limitations=['small authored games, two seeds and two terrains; general stable intelligence not proved',
            'purpose scales, rollout tactics and effects remain game-owned; not an automatic universal evaluator',
            'modeled future expectations are not learned/calibrated from delayed outcomes',
            'limited option sets, greedy tactical continuations and small model samples can miss effective plans',
            'future roles are replanned each real turn; not a persistent multi-turn commitment protocol',
            'shared objective cooperation; bargaining between actors with incompatible private goals is not implemented'])
    write_json(Path(output)/'evaluation.json',result)
    lines=['# 賢さの安定：目的の下限・共同予測・経験との分離','',
        f'{len(runs)} games; seeds {list(seeds)}; {checks} real transitions replayed.', '',
        '現在は機能を足して成立させる段階。速度・費用は合否に使わない。勝利/人格/手段の失敗を評価する。', '',
        '|種別|方式|人格|勝利/局数|敗北|味方の移動先競合|実移動失敗|共同案採用tick|辞退tick|',
        '|---|---|---|---|---|---|---|---|---|']
    for s in summary:lines.append(f"|{s['genre']}|{s['variant']}|{s['profile']}|{s['won']}/{s['games']}|{s['lost']}|{s['own_proposal_collisions']}|{s['focal_failed_moves']}|{s['planned_ticks']}|{s['planning_declines']}|")
    lines += ['', '対条件の勝利（改善/悪化/同じ）：']
    for c in comparisons:lines.append(f"- {c['genre']}/{c['variant']}/{c['rival']}: {c['better']}/{c['worse']}/{c['same']}。")
    lines += ['',f'品質ゲート: {verdict}。全人格の勝利維持・最弱人格改善・各相手群で悪化超過なし。差を縮めるだけでは合格にしない。', '',
        '得たもの: 共通DecisionLoopに共同先読みを接続し、未来の想定と即時の実経験を分離。目的達成見込みが最良案から大きく外れる候補を避け、その帯の中では既存人格Policyの最大主義/欲求/リスクで選ぶ。味方の進路競合を予測し、目的担当と援護担当を組み合わせる。', '',
        '制限したもの: 目的の見込みが設定した帯から外れる手段の自由。帯内での損失やこだわりは残す。最大主義を他者の多数決で押し切る協力案は採用しない。人格別の勝率補正はない。', '',
        '維持したもの: 固定Big Five/有限価値、現在の欲求、合法手/同一情報、個体別の経験と選択ticket。未来の仮説・未選択結果・共同成果の因果功績を実観測として学ばない。', '',
        '検証ゲーム専用の部分: 戦闘の移動/射撃/拠点と、経済の生産/資源/勝利を評価するアダプタ。共通の選択/人格/目的帯/現在経験との分離はdeliberation.pyとDecisionLoopで共用。具体的な戦術/価値尺度はゲームが供給する。', '',
        '残る不足: 未見ゲームの評価、未来予測の誤りを遅延結果で学ぶ契約、相手の傾向の学習、互いに両立しない人格の交渉、役割の継続と切替。今回の品質ゲートと人格別結果を超えて完成を主張しない。', '',
        'Colab/GPU/Drive/LLM教師/訓練なし。ローカルCPUの独立対戦を複数workerで進行。速度費用を理由に機能を削る合否判定はしていない。']
    (Path(output)/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    return result
