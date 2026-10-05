"""Goofspiel matches, exact conditional diagnostics and unapproved materials."""
from pathlib import Path
import copy
import json
import platform
import time
import numpy as np
from .core import Policy, compile_batch, digest
from .laboratory import profiles
from .board_planning import core_audit
from .cross_games import core_hashes
from .monte_carlo import RolloutBudget
from .teachers import prompt, write_json
from .goofspiel import Position, referee, reserve_bid, observe, exact_two_rounds, RULES


def controls():
    return {'reflex':None,**{f'mc{n}':RolloutBudget(samples=n,max_steps=13,min_samples=4,rollout_policy='random') for n in (8,32)}}


def snapshot(c,stats,method,profile,teachers):
    tag=[c['facts']['goal'],method,profile]
    if any(r['tag']==tag for r in teachers): return
    teachers.append(dict(id=f'goofspiel-{len(teachers):03d}',tag=tag,context=copy.deepcopy(c),context_hash=digest(c),
        prompt=prompt(c),quality='awaiting_generation',source='pre_decision_snapshot_no_gold',
        evaluation_provenance=copy.deepcopy(stats)))


def play(players,seed,profile_index,goal,method,opponent,teachers=None,cards=13):
    order='ascending' if seed%2==0 else 'descending'; s=Position.start(players,cards,order)
    seat=seed%players; profile=profiles()[profile_index]; policy=Policy(); memory=None
    rng=np.random.default_rng(60000+seed*37+players); trace=[]; durations=[]; budget=controls()[method]
    while not s.terminal:
        start=time.perf_counter()
        c,stats=observe(s,seat,profile,goal,seed,s.round,f'goof-{players}-{seed}',memory,budget)
        d=policy.choose(c); durations.append((time.perf_counter()-start)*1000)
        if teachers is not None: snapshot(c,stats,method,profile['id'],teachers)
        own=int(d['action_id'].split(':')[1]); bids=[]
        # All opponent actions are computed from the same pre-reveal state.
        for actor,h in enumerate(s.hands):
            bid=own if actor==seat else (h[int(rng.integers(len(h)))] if opponent=='random' else reserve_bid(s,actor,rng))
            bids.append(bid)
        after=s.play(tuple(bids)); referee(s,bids,after)
        chosen=stats.get('actions',{}).get(d['action_id'])
        trace.append(dict(round=s.round,prize=s.prizes[s.round],public_hands=[list(h) for h in s.hands],
                          scores_before=list(s.scores),joint_bids=bids,scores_after=list(after.scores),
                          candidate_action=d['action_id'],mode=d['next_state']['mode'],
                          samples=stats['completed_samples'],nodes=stats.get('additional_nodes',0),
                          fallback=budget is not None and not stats['used'],prediction=chosen))
        s=after; memory=d['next_state']
    share,winners=s.share(seat)
    return dict(players=players,seed=seed,prize_order=order,profile=profile['id'],seat=seat,goal=goal,method=method,
                opponent=opponent,scores=list(s.scores),own_score=s.scores[seat],win_share=share,won=seat in winners,
                discarded=s.discarded,finished=s.terminal,decision_ms=durations,trace=trace)


def exact_probe(cases=8):
    targets=[]; diagnostics=[]
    for players in (3,4,6):
        for seed in range(cases):
            s=Position.start(players,order='ascending' if seed%2==0 else 'descending')
            rng=np.random.default_rng(90000+seed*43+players)
            while len(s.hands[0])>2:
                bids=tuple(h[int(rng.integers(len(h)))] for h in s.hands)
                child=s.play(bids); referee(s,bids,child); s=child
            seat=seed%players; exact=exact_two_rounds(s,seat)
            public=dict(hands=[list(h) for h in s.hands],scores=list(s.scores),prizes=list(s.prizes),
                        round=s.round,discarded=s.discarded,viewer=seat)
            for name,value in exact.items():
                targets.append(dict(id=f'exact-{players}-{seed}-{name}',observation=public,action_id=name,**value,
                    source='exact_conditional_value_target',assumption='independent uniform opponent bids; final round forced',
                    quality='factual_model_target_not_reviewed_action_label',split='test' if seed%4==3 else 'train',
                    family=f'goof-{players}-{seed}'))
            for n in (8,32):
                budget=RolloutBudget(samples=n,min_samples=4,rollout_policy='random',max_steps=13)
                # Score and winner-share statistics are measured in the SAME
                # rollouts; no need to repeat identical trials for both goals.
                _,stats=observe(s,seat,profiles()[0],'score',seed,11,f'exact-{players}-{seed}',budget=budget)
                for name,value in exact.items():
                    estimate=stats['actions'][name]
                    diagnostics.append(dict(players=players,seed=seed,samples=n,action_id=name,
                        score_error=estimate['mean_score']-value['mean_score'],
                        share_error=estimate['win_share']-value['win_share']))
    summary=[]
    for n in (8,32):
        rows=[r for r in diagnostics if r['samples']==n]
        summary.append(dict(samples=n,action_estimates=len(rows),
            score_rmse=float(np.sqrt(np.mean([r['score_error']**2 for r in rows]))),
            share_rmse=float(np.sqrt(np.mean([r['share_error']**2 for r in rows])))))
    return targets,dict(cases_per_player_count=cases,summary=summary,rows=diagnostics)


def population_probe():
    """Live simultaneous personality batches, without forced profile branches."""
    results=[]; policy=Policy(); all_profiles=profiles()[:4]
    for players in (3,4,6):
        s=Position.start(players); memories=[None]*players; trace=[]
        roster=[all_profiles[i%4] for i in range(players)]
        while not s.terminal:
            contexts=[observe(s,i,roster[i],'score',73,s.round,f'batch-{players}',memories[i])[0] for i in range(players)]
            b=compile_batch(contexts); decisions=policy.decide(b).records(b)
            separate=[policy.choose(c) for c in contexts]
            assert decisions==separate,'batch/individual decisions differ'
            bids=tuple(int(d['action_id'].split(':')[1]) for d in decisions)
            after=s.play(bids); referee(s,bids,after)
            trace.append(dict(round=s.round,joint_bids=list(bids),profiles=[p['id'] for p in roster]))
            memories=[d['next_state'] for d in decisions]; s=after
        results.append(dict(players=players,finished=s.terminal,scores=list(s.scores),
                            differing_bid_rounds=sum(len(set(r['joint_bids']))>1 for r in trace),trace=trace))
    # Counterfactual: identical situation, change only profile (no distinct RNG).
    s=Position.start(4); rows=[]
    for i,p in enumerate(all_profiles):
        c=observe(s,0,p,'score',73,0,'same-situation')[0]
        rows.append(dict(profile=p['id'],action=policy.choose(c,False)['action_id']))
    return dict(batch_individual_mismatches=0,live_runs=results,same_situation=rows)


def experiment(root,seeds=2,exact_cases=8,progress=None):
    if type(seeds) is not int or seeds<1 or type(exact_cases) is not int or exact_cases<1:
        raise ValueError('positive case counts required')
    root=Path(root); root.mkdir(parents=True,exist_ok=True); start=time.perf_counter(); runs=[]; teachers=[]
    for players in (3,4,6):
        for seed in range(seeds):
            for profile in (0,3):
                for opponent in ('random','reserve'):
                    for goal in ('score','win_share'):
                        for method in controls():
                            runs.append(play(players,seed,profile,goal,method,opponent,teachers))
        if progress: progress(f'{players}人戦完了: 累計{len(runs)}局')
    targets,probe=exact_probe(exact_cases); population=population_probe(); groups=[]
    for players in (3,4,6):
        for opponent in ('random','reserve'):
            for goal in ('score','win_share'):
                for method in controls():
                    rows=[r for r in runs if (r['players'],r['opponent'],r['goal'],r['method'])==(players,opponent,goal,method)]
                    times=[t for r in rows for t in r['decision_ms']]
                    groups.append(dict(players=players,opponent=opponent,goal=goal,method=method,games=len(rows),
                        mean_score=float(np.mean([r['own_score'] for r in rows])),
                        mean_win_share=float(np.mean([r['win_share'] for r in rows])),
                        decision_p50_ms=float(np.median(times)),decision_p95_ms=float(np.percentile(times,95))))
    result=dict(game='goofspiel',rules=RULES,variant='public used bids; known alternating ascending/descending prizes; ties discarded',
        games=len(runs),finished=sum(r['finished'] for r in runs),referee_mismatches=0,
        fallback_decisions=sum(t['fallback'] for r in runs for t in r['trace']),core_hashes=core_hashes(),
        core_unchanged=core_audit(),groups=groups,exact_probe=probe,population_probe=population,
        value_target_rows=len(targets),teacher_requests=len(teachers),accepted_llm_rows=0,
        elapsed_seconds=time.perf_counter()-start,environment=dict(platform=platform.platform(),numpy=np.__version__),
        protocol=dict(seeds=seeds,players=[3,4,6],profiles=['growth','ego'],cards=13,
            seat='seed modulo players; NOT exhaustive seats',forecast='all future actors uniform independent; not actual opponent model',
            paired='same seed/profile/seat/order/opponent across 6 goal-method combinations',
            inference='small paired deterministic cases, no population win-rate confidence claim',
            training='exact conditional value targets and factual trajectories only; no fitted/approved universal policy'))
    write_json(root/'evaluation.json',result); write_json(root/'trajectories.json',runs)
    for filename,rows in (('value_targets.jsonl',targets),('teacher_requests.jsonl',teachers)):
        (root/filename).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows),encoding='utf-8')
    lines=['# Goofspiel: 同時入札・多人数の評価実験','',
        f'公式実装の規則: [OpenSpiel Goofspiel]({RULES})。全員が1〜13の札を1枚ずつ同時に出す。単独最高入札が得点を取り、最高同額なら得点札を破棄する。',
        '使用済み札は公開、得点札は既知の昇順/降順。隠れた手札版は今回扱わない。独自の規則実装を独立審判で検査し、OpenSpiel本体は依存に追加していない。','',
        f'{result["games"]}局/{result["finished"]}局完了、審判不一致0、MC予算不足の反射復帰{result["fallback_decisions"]}回。3/4/6人、2人格、2種類の相手、得点/優勝持分の2目的、反射/MC8/MC32を同じ条件で比較。',
        '各方法の未来入札仮説は全員一様ランダム。実相手reserveは得点札に近い札を使う別方策なので、モデルずれを含む。実相手の現在の札、人格、将来の乱数は評価へ渡さない。',
        'scoreは自分の点数を増やす目的、win_shareは最高得点者の持分を増やす目的。反射のwin_shareは現在の点差代理値、MCは終局持分。公式win_loss数値とは異なる。人格の価値/欲求を合わせた最終選択は目的単独の最大化ではない。','',
        '## 条件別の実対戦結果','',
        '|人数|相手|目的|方法|局数|平均点数|平均優勝持分|判断p50 ms|',
        '|---|---|---|---|---|---|---|---|']
    lines += [f'|{g["players"]}|{g["opponent"]}|{g["goal"]}|{g["method"]}|{g["games"]}|{g["mean_score"]:.2f}|{g["mean_win_share"]:.3f}|{g["decision_p50_ms"]:.2f}|' for g in groups]
    lines+=['','## 全列挙できる終盤で評価誤差を測る','',
        '各人数の2ラウンド残りの局面を固定。相手の現在の札を全列挙し、最後の札は全員強制。独立一様入札を仮定した期待点数/優勝持分の正確な参照値であり、最適戦略/Nash均衡の正解ではない。','',
        '|各候補の標本数|行動期待値の比較数|点数RMSE|持分RMSE|','|---|---|---|---|']
    lines += [f'|{r["samples"]}|{r["action_estimates"]}|{r["score_rmse"]:.4f}|{r["share_rmse"]:.4f}|' for r in probe['summary']]
    lines += ['', '## 得たもの・削ったもの・方向性','',
        '- 得たもの: 同時入札にも同じ人格コアを接続。3/4/6人の混合バッチと個別判断が一致。終局だけでなく各ラウンドの得点も記録し、評価値の学習に使える条件付きの数値参照を追加。',
        '- 得たもの: 目的の定義と人格の選択を分離。同時に選ぶ相手の札を見せず、全候補に同数の完了標本を与える既存MCを再利用。',
        '- 今回省いたもの: 隠れた手札、未知の得点順、交渉/同盟/報復、均衡探索。評価誤差を分離するための実験範囲で、コアからこれらを削除したわけではない。',
        '- 費用/近似: 各候補8/32回の終局対戦は反射より高価。未来の自分も一様ランダムという近似があり、実際に将来も人格判断する行動価値とは一致しない。8結果圧縮は平均を保つが損失分布を近似する。',
        '- 方向性: 通常の数値反射/並列性と、人格と独立した読み予算を維持。コアとruntime本体は変更なし。安全性/権力/入札余力などの効果はゲーム側の手設計で、心理的妥当性や汎用学習を証明しない。',
        '- 学習状態: value_targets.jsonlは仮説に条件づけた数値ターゲット、教師の好ましい行動ラベルではない。軌跡/教師相談は未採用。学習済みの汎用コアや評価モデルはまだ作っていない。',
        '- 全員人格戦と同じ状況の4人格比較はevaluation.json内population_probe。違う札が出ただけで合理性/人格の再現性が完成したとは扱わない。',
        '- 観測した弱点: 同じ初期状態でgrowth/egoの反射は低い得点札にも最高札を使う。現在得点の貪欲評価と弱い将来余力代理値の限界で、人格として合理的な損かどうかは未検証。コアの最低限の知能を達成した証拠にはしない。',
        '- 標本精度と強さは別: 終盤の全列挙参照では32標本のRMSEを8標本と比較できるが、実相手の仮説ずれ/人格の重みづけ/局面次第で実対戦の得点や優勝持分が悪化しうる。単調な改善を保証しない。',
        '- 実行はローカルCPU。Colab/GPUは今回起動せず、新しいクラウドセッション消費なし。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
