"""Economic/multiple-victory headless comparison with replayable public decisions."""
from dataclasses import replace
from pathlib import Path
from collections import Counter
import copy
import json
import time
import numpy as np
from .core import Policy, compile_batch, digest
from .laboratory import profiles
from .resource_world import (World, ROUTES, legal, act, step, winners, terminal,
    potential, goal_progress, achieved, economic_move, make_context, world_record,
    world_from_record)
from .cross_games import core_hashes
from .teachers import write_json


SCENARIOS = {
    'mixed':dict(routes=ROUTES, event='food'),
    'science':dict(routes=('science',), event='ore'),
    'culture':dict(routes=('culture',), event='trade'),
}


def public_event(w, event):
    # Announced only after production round 8, outside the forecast interface.
    if w.round != 8:
        return w
    return replace(w, **{'food':{'food_yield':1}, 'ore':{'ore_yield':0},
                          'trade':{'trade_price':1}}[event])


def run(profile, seed, scenario, horizon, record=True):
    config = SCENARIOS[scenario]
    w = World.start(routes=config['routes'])
    focal = seed % 4
    policy = Policy(); memory = None; trace = []
    times = []; model_nodes = 0; skipped_wins = 0; public_changes = 0
    while not terminal(w):
        viewer = w.turn
        before = w
        if viewer == focal:
            t = time.perf_counter()
            c, stats = make_context(w, profile, seed, len(trace), f'{scenario}-{seed}', memory, horizon)
            d = policy.choose(c)
            times.append((time.perf_counter()-t)*1000)
            model_nodes += stats['model_transitions']
            key = d['action_id']; memory = d['next_state']
            possible_wins = [k for k in legal(w) if focal in winners(act(w, k))]
            skipped_wins += bool(possible_wins and key not in possible_wins)
        else:
            key = economic_move(w)
            c = d = stats = None
        w = step(w, key)
        if w.round != before.round:
            previous = w; w = public_event(w, config['event'])
            public_changes += w != previous
        if record:
            trace.append(dict(before=world_record(before), actor=viewer, action=key,
                context=c, decision=d, forecast=stats, after=world_record(w)))
        else:
            # Tick independence from logging choice: one entry per real action.
            trace.append(None)
    final = w.empires[focal]
    return dict(profile=profile['id'], seed=seed, seat=focal, scenario=scenario,
        horizon=horizon, winner=list(winners(w)), won=focal in winners(w),
        terminated_by='victory' if winners(w) else 'round_limit', rounds=w.round,
        actions=len(trace), focal_actions=len(times), final=world_record(w),
        achieved=list(set(achieved(final)) & set(w.routes)),
        goal_progress=goal_progress(final,w.routes), potential=potential(w,focal),
        shortages=final.shortages, overflow=final.overflow,
        skipped_immediate_wins=skipped_wins, public_changes=public_changes,
        forecast_transitions=model_nodes,
        end_to_end_decision_p50_ms=float(np.median(times)),
        end_to_end_decision_p95_ms=float(np.percentile(times,95)), trace=trace)


def replay(run_record):
    policy = Policy(); checked = 0
    for row in run_record['trace']:
        before = world_from_record(row['before'])
        assert row['actor'] == before.turn
        assert row['action'] in legal(before)
        after = step(before, row['action'])
        if after.round != before.round:
            after = public_event(after, SCENARIOS[run_record['scenario']]['event'])
        assert world_record(after) == row['after']
        if row['context'] is not None:
            d = policy.choose(row['context'])
            assert d == row['decision']
            assert d['action_id'] == row['action']
            checked += 1
    return checked


def probe():
    w = World.start()
    # Fixed legal public economic snapshots. Deliberate probes, not held-out data.
    states = [w]
    for _ in range(18):
        if terminal(w): break
        w = step(w, economic_move(w,1))
        states.append(w)
    rows = []; batch_checks = 0; distinct = 0
    for i, w in enumerate(states):
        if terminal(w): continue
        contexts = [make_context(w,p,91,i,'resource-probe',horizon=4)[0] for p in profiles()]
        singles = [Policy().choose(c) for c in contexts]
        batch = compile_batch(contexts)
        together = Policy().decide(batch).records(batch)
        assert together == singles
        batch_checks += len(singles)
        choices = [d['action_id'] for d in singles]
        distinct += len(set(choices)) > 1
        rows.append(dict(round=w.round, turn=w.turn, actions=choices))
    return dict(snapshots=len(rows), snapshots_with_personality_difference=distinct,
                batch_decisions_checked=batch_checks, rows=rows)


def experiment(root, seeds=4, progress=print):
    root = Path(root); root.mkdir(parents=True,exist_ok=True)
    original = core_hashes(); runs = []
    for scenario in SCENARIOS:
        for profile in profiles():
            for seed in range(seeds):
                for horizon in (1,4):
                    r = run(profile,seed,scenario,horizon)
                    r['replayed_persona_decisions'] = replay(r)
                    runs.append(r)
        progress(f'resource-world: {scenario} complete, {len(runs)} games')
    assert original == core_hashes()
    paired = []
    for a,b in zip(runs[::2],runs[1::2]):
        assert (a['profile'],a['seed'],a['scenario']) == (b['profile'],b['seed'],b['scenario'])
        paired.append(dict(profile=a['profile'],seed=a['seed'],scenario=a['scenario'],
            immediate_win=a['won'],forecast_win=b['won'],
            immediate_progress=a['goal_progress'],forecast_progress=b['goal_progress'],
            immediate_shortage=a['shortages'],forecast_shortage=b['shortages']))
    by_condition = []
    for scenario in SCENARIOS:
        for horizon in (1,4):
            subset = [r for r in runs if r['scenario']==scenario and r['horizon']==horizon]
            by_condition.append(dict(scenario=scenario,horizon=horizon,games=len(subset),
                wins=sum(r['won'] for r in subset), completed=sum(r['terminated_by']=='victory' for r in subset),
                mean_progress=float(np.mean([r['goal_progress'] for r in subset])),
                mean_shortage=float(np.mean([r['shortages'] for r in subset])),
                mean_overflow=float(np.mean([r['overflow'] for r in subset])),
                skipped_immediate_wins=sum(r['skipped_immediate_wins'] for r in subset),
                median_decision_ms=float(np.median([r['end_to_end_decision_p50_ms'] for r in subset]))))
    summaries = [{k:v for k,v in r.items() if k!='trace'} for r in runs]
    result = dict(schema='resource-world-evaluation-v1',games=len(runs),conditions=by_condition,
        win_pairs=dict(better=sum(not a['immediate_win'] and a['forecast_win'] for a in paired),
            worse=sum(a['immediate_win'] and not a['forecast_win'] for a in paired),
            same=sum(a['immediate_win']==a['forecast_win'] for a in paired)),
        paired=paired, runs=summaries, personality_probe=probe(), core_hashes=original,
        replayed_persona_decisions=sum(r['replayed_persona_decisions'] for r in runs),
        development_evaluation=True,
        effect_boundary=dict(root_persona_effects_only=True,root_resource_cost_only=True,
            future_goal_supply_weight=.65,weight_is_calibrated_probability=False,
            nonterminal_potential_scale=.6),
        skipped_immediate_wins_total=sum(r['skipped_immediate_wins'] for r in runs),
        limitations=['Authored economic stress test, not Civ or commercial board-game strength',
            'Game-authored production/liquidity/threshold proxy, not learned universal value',
            'Solo greedy future; rival responses and territory depletion by rivals NOT forecast',
            '1/4 modeled own turns are estimates, not actual multiplayer search',
            'Seeds change personality randomness and focal seats; only one authored board',
            'No training or automatic factual-learning validation in this experiment'])
    write_json(root/'evaluation.json',result)
    write_json(root/'trajectories.json',dict(runs=runs))
    lines = ['# 複数資源・複数勝利条件の経済ストレス検証','',
        '自作4人ゲーム。Civや既存商用ボードゲームの再現・性能証明ではない。GUIなし。',
        '資源は食料/木材/鉱石/資金、研究/文化/軍事も別管理。生産設備、維持費、貯蔵上限、技術前提、共有領地、他者領地への侵攻がある。',
        '科学は技術3段＋研究14、文化は記念碑2＋文化20、勢力は領地4＋軍事6。行動時の達成者は即勝利、生産時の同時達成は同着。混合条件ではどれか1つ、科学/文化条件では指定経路だけ有効。',
        '8ラウンド後に供給/市場の変化が公開される。その予定時刻は判断や予測へ渡さない。変更の直後から現供給を使う。',
        '', '## 比較','',
        '本人だけ同じ固定人格Policyで1/4自己手番の予測を比較。相手3人は公開情報のみの同じ経済代理評価の4手モデルを使う専用比較方策。ゲームの有効ルートは全員共通。',
        '4手モデルでは相手を停止し、現在供給を維持した仮想自己手番を読む。全根候補を残し、内側は代理評価の貪欲1案。勝率推定や多人数の実未来とは呼ばない。',
        '人格の価値/様式/資源費用は今選ぶ根行動だけから計算する。将来の目的/供給効果は0.65で混ぜる工学的仮置きで、校正済み確率ではない。研究/文化の消費予算と永続する技術/設備を区別し、安全は過剰な蓄財でなく維持費不足の回避として評価する。',
        '判断入力のfactsは数値採点では読まれない。意味はAdapterの結果22特徴へ明示変換する。文章のルール理解は未実装。',
        '', '|条件|自己手番予測|局数|本人勝利|勝利で終了|平均条件進捗|平均不足累計|判断中央値ms|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in by_condition:
        lines.append(f"|{r['scenario']}|{r['horizon']}|{r['games']}|{r['wins']}|{r['completed']}|{r['mean_progress']:.3f}|{r['mean_shortage']:.2f}|{r['median_decision_ms']:.2f}|")
    lines += ['',f"対応比較の勝敗：{result['win_pairs']}。進捗は勝利条件への距離の代理で、勝率と別。",
        f"再生・人格再採点：{result['replayed_persona_decisions']}判断。共通core/runtimeバイト不変。",
        f"同局面の人格差：{result['personality_probe']['snapshots_with_personality_difference']}/{result['personality_probe']['snapshots']}。バッチ一致：{result['personality_probe']['batch_decisions_checked']}判断。",'',
        '## 得たもの・費用・限界','',
        '- 得たもの：維持費/貯蔵/前提投資/他者との有限資源競争/複数の勝利条件を同じ人格コアで動かす検証環境。全選択と実遷移を再生できる。',
        '- 追加費用：4手の各候補で仮想行動を作る。時間は入力変換/予測/Policy込み、相手の判断/ゲーム進行/描画/IOを含まない。通常バッチの速度とは別。',
        '- 限界：ゲーム側で設備・流動性・勝利条件進捗の代理評価を設計した。ルールだけから価値を獲得したわけではない。',
        '- 開発中の修正：初版96局を見て、仮想未来の人格利益を根行動へまとめる誤り、研究予算の消費を知性低下とする写像、未完了の資源蓄積を勝利に近い報酬にする写像を修正した。この再測定は独立した未知環境の性能評価ではない。初版はbackups/resource-world-before-effect-boundary.zipへ保全。',
        '- 限界：相手が先に土地を取る未来、相手の方針変更、人格に沿った自分の未来行動は未統合。この省略で4手が悪化しても隠さない。',
        '- 限界：手番/人格乱数を変えた少数条件、1つの自作盤面。未見ゲーム・人間相手・Civ規模の証拠ではない。',
        '- 即時勝利の見送り：'+str(result['skipped_immediate_wins_total'])+'回。人格の主義優先も含み、その損と能力不足の判別は未完成。最低限の合理性を全面保証したとは扱わない。',
        '- 学習/運用：実行はローカルCPU。Colab/GPU/Driveを起動せず、教師生成/採用/学習は行わない。既存ゲームの機能は削っていない。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
