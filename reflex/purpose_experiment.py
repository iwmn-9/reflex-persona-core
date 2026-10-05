"""Fixed-candidate purpose tests; actual wins gate adoption, not gap alone.

The candidate changes combat effect semantics and connects immediate selected
outcomes to the existing bounded memory. Shared Policy is not retrained.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import platform
import time
import numpy as np
from .laboratory import profiles
from .teachers import write_json
from .cross_games import core_hashes
from .validation_experiment import combat,auction,resources,gambling,replay


def source_hashes():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob('*.py'))}


def experiment(output,seeds=(70,71),progress=None):
    from .combat import GOALS
    from .contests_experiment import SCENARIOS as AUCTION
    from .resource_experiment import SCENARIOS as RESOURCE
    from .gambling import SCENARIOS as GAMBLING
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    frozen=source_hashes();core=core_hashes();runs=[];checks=0;start=time.perf_counter()
    with (output/'trajectories.jsonl').open('w',encoding='utf-8') as stream:
        for rival in ('reference','raider','switch'):
            for terrain in ('open','choke'):
                for goal in GOALS:
                    for p in profiles():
                        original=json.dumps(p,sort_keys=True)
                        for seed in seeds:
                            for variant in ('baseline','candidate'):
                                r=combat(p,seed,terrain,goal,'baseline',rival=rival,
                                    learn=variant=='candidate',survival_security=variant=='candidate')
                                r.update(variant=variant,rival=rival)
                                checks+=replay(r);stream.write(json.dumps(r,ensure_ascii=False)+'\n')
                                runs.append({k:v for k,v in r.items() if k!='trace'})
                        assert original==json.dumps(p,sort_keys=True)
                    if progress:progress(f'{rival}/{terrain}/{goal}: {len(runs)} games',flush=True)
        # Controls use identical shared core. Known gambling odds and future
        # resource proxies never enter combat's immediate empirical channel.
        jobs=[(g,s,auction,dict(game=g)) for g in ('auction','hagetaka') for s in AUCTION]
        jobs += [('resources',s,resources,{}) for s in RESOURCE]
        jobs += [('gambling',s,gambling,{}) for s in GAMBLING]
        for genre,scene,runner,extra in jobs:
            for p in profiles():
                for mode in ('baseline','pressure'):
                    r=runner(p,seeds[0],scene,mode,**extra)
                    r.update(variant=mode,rival='game-owned')
                    checks+=replay(r);stream.write(json.dumps(r,ensure_ascii=False)+'\n')
                    runs.append({k:v for k,v in r.items() if k!='trace'})
            if progress:progress(f'{genre}/{scene}: {len(runs)} games',flush=True)
    assert frozen==source_hashes() and core==core_hashes()
    return finalize(output,runs,checks,frozen,core,seeds,time.perf_counter()-start)


def finalize(output,runs,checks,frozen,core,seeds,elapsed):
    summary=[];pairs=[]
    genres=('combat','auction','hagetaka','resources','gambling')
    for genre in genres:
        for variant in ('baseline','candidate','pressure'):
            for p in profiles():
                rows=[r for r in runs if (r['genre'],r['variant'],r['profile'])==(genre,variant,p['id'])]
                if not rows:continue
                totals={k:sum(r.get(k,0) for r in rows) for k in
                    ('won','lost','learned_uses','regime_resets','focal_failed_moves','decisions','negative_ev_bets','ruined','shortages')}
                summary.append(dict(genre=genre,variant=variant,profile=p['id'],games=len(rows),
                    win_credit=sum(r['win_credit'] for r in rows),**totals))
        rows=[r for r in runs if r['genre']==genre]
        index={(r['scenario'],r['profile'],r['seed'],r['rival']):r for r in rows if r['variant']=='baseline'}
        for rival in sorted({r['rival'] for r in rows}):
            changed=[r for r in rows if r['variant']!='baseline' and r['rival']==rival]
            comparisons=[(index[(r['scenario'],r['profile'],r['seed'],rival)],r) for r in changed]
            pairs.append(dict(genre=genre,rival=rival,pairs=len(comparisons),
                better=sum(b['win_credit']>a['win_credit'] for a,b in comparisons),
                worse=sum(b['win_credit']<a['win_credit'] for a,b in comparisons),
                same=sum(b['win_credit']==a['win_credit'] for a,b in comparisons)))
    base={s['profile']:s for s in summary if (s['genre'],s['variant'])==('combat','baseline')}
    candidate={s['profile']:s for s in summary if (s['genre'],s['variant'])==('combat','candidate')}
    # Deliberately conservative gate: do not call stronger-persona erosion a
    # success. This small authored matrix cannot prove general intelligence.
    no_persona_drop=all(candidate[k]['won']>=v['won'] for k,v in base.items())
    weakest=min(base,key=lambda k:base[k]['won']/base[k]['games'])
    weak_gain=candidate[weakest]['won']>base[weakest]['won']
    opponents_pass=all(p['better']>=p['worse'] for p in pairs if p['genre']=='combat')
    result=dict(format='purpose-validation-v1',seeds=list(seeds),
        seed_status='70/71 fixed before candidate confirmation; 60/61 used for development',
        source_hashes=frozen,core_hashes=core,games=len(runs),rule_transitions_replayed=checks,
        numeric_batch_single_comparisons=sum(r['decisions'] for r in runs),
        elapsed_seconds=elapsed,platform=platform.platform(),summary=summary,comparisons=pairs,runs=runs,
        adoption=dict(default_changed=False,confirmation_gate_passed=no_persona_drop and weak_gain and opponents_pass,
            weakest_persona=weakest,weakest_improved=weak_gain,no_persona_win_drop=no_persona_drop,
            no_opponent_family_paired_regression=opponents_pass),
        contracts=['fixed traits/values, same actual rules and world randomness in paired games',
            'candidate = survival-valued security plus uncertain immediate volley plus selected immediate memory; bundled effects, not isolated causal estimate',
            'actual joint-turn feedback includes others; not causal own-action credit',
            'route/terrain/health/ammo/visibility condition, max 64 memory entries and 4 recent samples',
            'no real rival intents/controller labels/RNG in forecasts; newly-visible actors cannot be selected retroactively',
            'all actual transitions replayed; batch/single checked live; checkpoints restored',
            'control games retain existing game-owned effects; no claim of transferred combat improvement'],
        limitations=['two reserved seeds, two terrains, four goals, three authored rival families; no unseen-commercial-game claim',
            'uniform independent firing model is approximate, not calibrated opponent adaptation',
            'delayed consequences, team intent coordination and personality-consistent multi-step search remain unintegrated',
            'learning is selected on-policy association, not evidence that the action caused joint success',
            'timings include combat decisions and feedback, exclude world simulation and benchmark audit scoring; no large-NPC real-time claim'])
    write_json(Path(output)/'evaluation.json',result)
    lines=['# 人格を保った目的達成と経験接続の確認','',
        f'{len(runs)} games; {checks} actual transitions replayed. Seeds {list(seeds)}.', '',
        '戦闘候補は、主義の安全を生存損失、負傷を欲求/リスクに分け、現在の同時射撃の不確実性を評価する。実際に選んだ手の1ターン後の公開結果だけを既存の有界記憶へ返す。未来成果や未選択の結果は学習しない。', '',
        '|種別|方式|人格|勝利持分/局|敗北|経験使用判断|失敗移動|',
        '|---|---|---|---|---|---|---|']
    for s in summary:lines.append(f"|{s['genre']}|{s['variant']}|{s['profile']}|{s['win_credit']:g}/{s['games']}|{s['lost']}|{s['learned_uses']}|{s['focal_failed_moves']}|")
    lines += ['', '対条件での勝利持分（改善/悪化/同じ）：']
    for p in pairs:lines.append(f"- {p['genre']}/{p['rival']}: {p['better']}/{p['worse']}/{p['same']}。")
    lines += ['',f"確認ゲート: {result['adoption']}。既定は変更しない。最弱人格の改善、各人格の勝利維持、各相手群の対条件で悪化超過なしを同時に要求する。",'',
        '得たもの: 戦闘で即時の実経験を判断/方針と接続した比較可能な候補。途中で相手の方針が変わる対戦、同時移動の失敗、各人格の実勝敗を含め、評価の誤りと経験接続を一緒に検証できる。共有Policy/人格/主義の優先規則は変更しない。', '',
        '捨てた案: 停滞時に優先モードだけを周期的に再抽選する案と、目標進行を欲求へ足すだけの案は開発試験で安定改善せず、今回の候補から外した。防御を違法扱いしたり、慎重な人格を強制解除したりはしない。開発試験の60/61は確認用の70/71と分けた。', '',
        '費用: 即時の不確実な結果が最大8分岐、個体別の有界記憶、JSON/契約検査、実観測の更新。大量NPCの性能合格を意味しない。', '',
        '未解決: 本人の目的への長期価値、経験からの相手行動モデル改善、味方の選択の競合、遅延成果の帰属、どの人格でも最低限賢いという安定性。勝率差だけを縮める修正は採用しない。', '',
        '他ゲーム性の確認は共通コアを保つ対照試験。戦闘の生存評価をオークション等へコピーしていない。正確なギャンブル確率を少数の実運で上書きせず、資源モデルの未来収入を即時実観測にしない。', '',
        'Colab/GPU/Drive・LLM教師・訓練は使わず、ローカルCPUで実行。']
    (Path(output)/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    return result
