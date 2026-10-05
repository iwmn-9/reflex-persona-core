"""Public-only hypothesis predictions, paired matches and model-shift probes."""
from collections import Counter
import copy
import json
import math
from pathlib import Path
import time
import numpy as np
from .core import Policy,digest
from .laboratory import profiles
from .monte_carlo import RolloutBudget
from .goofspiel import Position,referee,reserve_bid,observe
from .goofspiel_beliefs import PublicBidBeliefs
from .goofspiel_reading import decide
from .board_planning import core_audit
from .cross_games import core_hashes
from .teachers import prompt,write_json

OPPONENTS=('random','reserve','high','switch','unseen','mixed')
METHODS=('reflex','read_reflex','mc16','read_mc16')


def actual_bid(s,actor,rng,controller):
    """Driver-only controls. Names/probabilities are never sent to the learner."""
    h=s.hands[actor]
    if controller=='mixed': controller=('random','reserve','high')[actor%3]
    if controller=='random': return h[int(rng.integers(len(h)))]
    if controller=='reserve': return reserve_bid(s,actor,rng)
    if controller=='high': return max(h)
    if controller=='switch': return max(h) if s.round<6 else min(h)
    if controller!='unseen': raise ValueError('unknown driver controller')
    # A withheld soft score-opportunity policy, absent from the four hypotheses.
    # Its live decisions see only public hands/prizes; no current sealed bids.
    prize=s.prizes[s.round]; future=sum(s.prizes[s.round+1:])/max(1,sum(s.prizes[s.round:]))
    utility=[prize*float(np.prod([sum(c<b for c in other)/len(other) for i,other in enumerate(s.hands) if i!=actor]))
             -.4*b*future for b in h]
    weights=np.exp((np.array(utility)-max(utility))/1.5); weights/=weights.sum()
    return h[int(rng.choice(len(h),p=weights))]


def snapshot(c,stats,method,profile,teachers):
    if not stats.get('reading_applied'): return
    tag=[c['facts']['goal'],method,profile]
    if any(r['tag']==tag for r in teachers): return
    teachers.append(dict(id=f'opponent-reading-{len(teachers):03d}',tag=tag,context=copy.deepcopy(c),
        context_hash=digest(c),prompt=prompt(c),quality='awaiting_generation',source='pre_decision_snapshot_no_gold',
        observed_beliefs=copy.deepcopy(stats.get('beliefs',{}))))


def play(players,seed,goal,method,opponent,teachers=None,cards=13):
    if method not in METHODS or opponent not in OPPONENTS: raise ValueError('unknown comparison condition')
    s=Position.start(players,cards,'ascending' if seed%2==0 else 'descending'); seat=seed%players
    profile=profiles()[0 if seed%2==0 else 3]; memory=None; tracker=PublicBidBeliefs(players,seat,contextual=False,responsive=False)
    rng=np.random.default_rng(110000+43*seed+players); trace=[]; times=[]; policy=Policy()
    budget=RolloutBudget(samples=16,min_samples=4,rollout_policy='random',max_steps=13) if 'mc16' in method else None
    while not s.terminal:
        forecast=tracker.snapshot(); start=time.perf_counter()
        if method.startswith('read_'):
            c,d,stats=decide(s,seat,profile,goal,seed,s.round,f'reading-{players}-{seed}',memory,budget,forecast)
        else:
            c,ev=observe(s,seat,profile,goal,seed,s.round,f'reading-{players}-{seed}',memory,budget,common_random=True)
            d=policy.choose(c); stats=dict(reason='disabled',reading_applied=False,action_changed=False,
                model_evaluated=False,total_nodes=ev.get('additional_nodes',0),node_cap=budget.max_nodes if budget else 0,
                baseline_evaluation=ev)
        times.append((time.perf_counter()-start)*1000)
        if teachers is not None: snapshot(c,stats,method,profile['id'],teachers)
        own=int(d['action_id'].split(':')[1])
        bids=tuple(own if i==seat else actual_bid(s,i,rng,opponent) for i in range(players))
        # Reveal and update only AFTER every live actor has committed.
        after=s.play(bids); referee(s,bids,after); observations=tracker.reveal(s,bids)
        assert stats['total_nodes']<=stats['node_cap']
        trace.append(dict(round=s.round,prize=s.prizes[s.round],scores_before=list(s.scores),
                          public_hands=[list(h) for h in s.hands],joint_bids=list(bids),scores_after=list(after.scores),
                          reading=stats,observations=observations,mode=d['next_state']['mode']))
        memory=d['next_state']; s=after
    share,winners=s.share(seat)
    return dict(players=players,seed=seed,profile=profile['id'],seat=seat,goal=goal,method=method,opponent=opponent,
                scores=list(s.scores),own_score=s.scores[seat],discarded=s.discarded,win_share=share,won=seat in winners,
                finished=s.terminal,trace=trace,decision_ms=times)


def prediction_probe(seeds=8):
    """Identical fixed public histories for prediction loss vs uniform baseline."""
    rows=[]; histories=[]
    for players in (3,4,6):
        for controller in OPPONENTS:
            for seed in range(seeds):
                s=Position.start(players,order='ascending' if seed%2==0 else 'descending')
                tracker=PublicBidBeliefs(players,0,contextual=False,responsive=False); rng=np.random.default_rng(120000+seed*47+players)
                trace=[]
                while not s.terminal:
                    bids=tuple(s.hands[0][int(rng.integers(len(s.hands[0])))] if i==0 else actual_bid(s,i,rng,controller)
                               for i in range(players))
                    observations=tracker.reveal(s,bids)
                    for actor,r in observations.items():
                        row=dict(players=players,controller=controller,seed=seed,round=s.round,actor=actor,**r)
                        rows.append(row)
                    after=s.play(bids); referee(s,bids,after)
                    trace.append(dict(round=s.round,joint_bids=list(bids),beliefs_before={i:r['before'] for i,r in observations.items()},
                                      beliefs_after={i:r['after'] for i,r in observations.items()})); s=after
                histories.append(dict(players=players,controller=controller,seed=seed,trace=trace))
    groups=[]
    for controller in OPPONENTS:
        for phase,lo,hi in (('all_unforced',0,12),('after_evidence',3,12),('after_switch',6,12)):
            subset=[r for r in rows if r['controller']==controller and lo<=r['round']<hi and not r['forced']]
            groups.append(dict(controller=controller,phase=phase,predictions=len(subset),
                learned_log_loss=float(np.mean([r['log_loss'] for r in subset])),
                uniform_log_loss=float(np.mean([r['uniform_log_loss'] for r in subset])),
                mean_trust=float(np.mean([r['before']['confidence'] for r in subset]))))
    return dict(seeds=seeds,histories=len(histories),groups=groups),histories


def experiment(root,seeds=2,prediction_seeds=8,progress=None):
    if any(type(n) is not int or n<1 for n in (seeds,prediction_seeds)): raise ValueError('positive seed counts required')
    root=Path(root); root.mkdir(parents=True,exist_ok=True); start=time.perf_counter(); runs=[]; teachers=[]
    for players in (3,4,6):
        for seed in range(seeds):
            for opponent in OPPONENTS:
                for goal in ('score','win_share'):
                    for method in METHODS:
                        runs.append(play(players,seed,goal,method,opponent,teachers))
            if progress: progress(f'{players}人 seed {seed} 完了、累計{len(runs)}局')
    prediction,histories=prediction_probe(prediction_seeds); groups=[]; paired=[]
    for opponent in OPPONENTS:
        for goal in ('score','win_share'):
            for method in METHODS:
                subset=[r for r in runs if (r['opponent'],r['goal'],r['method'])==(opponent,goal,method)]
                durations=[t for r in subset for t in r['decision_ms']]
                groups.append(dict(opponent=opponent,goal=goal,method=method,games=len(subset),
                    mean_score=float(np.mean([r['own_score'] for r in subset])),
                    mean_win_share=float(np.mean([r['win_share'] for r in subset])),
                    decision_p50_ms=float(np.median(durations)),decision_p95_ms=float(np.percentile(durations,95)),
                    reading_changes=sum(t['reading']['action_changed'] for r in subset for t in r['trace'])))
    for learned in [r for r in runs if r['method'].startswith('read_')]:
        base=next(r for r in runs if (r['players'],r['seed'],r['goal'],r['opponent'],r['method'])==
                  (learned['players'],learned['seed'],learned['goal'],learned['opponent'],learned['method'][5:]))
        paired.append(dict(players=learned['players'],seed=learned['seed'],goal=learned['goal'],opponent=learned['opponent'],
                           method=learned['method'],score_delta=learned['own_score']-base['own_score'],
                           share_delta=learned['win_share']-base['win_share']))
    reasons=Counter(t['reading']['reason'] for r in runs if r['method'].startswith('read_') for t in r['trace'])
    result=dict(games=len(runs),finished=sum(r['finished'] for r in runs),referee_mismatches=0,
        core_unchanged=core_audit(),core_hashes=core_hashes(),groups=groups,paired=paired,prediction_probe=prediction,
        gate_reasons=dict(reasons),teacher_requests=len(teachers),accepted_llm_rows=0,elapsed_seconds=time.perf_counter()-start,
        protocol=dict(players=[3,4,6],cards=13,seeds=seeds,methods=list(METHODS),actual_opponents=list(OPPONENTS),
            hypotheses=['uniform','reserve','high','low'],profile_order='growth on even seed; ego on odd, coupled to prize order',
            fair_comparison='same candidate persona/seat/prize order/driver RNG; common MC inverse-CDF variates',
            budget='16 completed trials per root per model; both estimates share one 20000-transition cap',
            virtual_learning='belief weights frozen within forecast; only later real revealed bids update trackers',
            no_oracle='actual controller labels/current sealed bids/future driver RNG never supplied to predictor',
            limits='small paired conditions, not calibrated trust or an independent population win-rate claim',
            transfer='generic categorical tracker + game-authored hypotheses; not automatically learned arbitrary-game beliefs'))
    write_json(root/'evaluation.json',result); write_json(root/'trajectories.json',runs); write_json(root/'prediction_histories.json',histories)
    (root/'teacher_requests.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in teachers),encoding='utf-8')
    lines=['# 公開行動からの相手仮説と、必要なときの読み','',
        f'{len(runs)}局中{result["finished"]}局完了、独立審判との不一致0。3/4/6人、同じ人格コア、反射/MC16の各々について推測なし/ありを比較した。','',
        '## 得たもの・費用・方向性','',
        '- 得たもの: 相手ごとの有限4仮説を、後から公開された行動の尤度で更新する汎用Tracker。ゲーム側が状況依存の分布を定義。性格の診断や真の相手方策の取得ではない。',
        '- 得たもの: 観測3回未満なら一様予測。仮説の混合に一様予測を残し、過去の重みを0.9倍で減衰。最近の実予測が外れると信頼度を下げる。強制行動は証拠として数えない。信頼度式は手設計で、校正済み確率ではない。',
        '- 得たもの: 仮説分布を反射の即時結果とMCの相手入札へ接続。根の重みを仮想対戦中に固定し、架空の観測で実Trackerを育てない。',
        '- 無駄の抑制: どの将来の手札/得点札でも一様予測との差の上限が5%未満なら二重評価を省く。信頼している相手がランダム型なら、信頼度が高くても読みを再実行しない。',
        '- 方針変更の制御: 優勝目的で数学的に勝ちが確定なら読みを省略。現在先行し、仮説下でも元の行動が有利を維持するなら切替えない。維持判定は推定持分0.6以上・標本内Wilson下限0.5超のゲーム用目安。点数目的では優勝安泰でも追加点に意味があるので同じ保護を強制しない。',
        '- 方針変更の制御: 人格の同じPolicyで両モデルを評価。切替えには価値階層の更新、または利益0.025＋目的の標準誤差に応じた余裕を要求。全人格特徴の不確実性やモデル誤りを保証する閾値ではない。',
        '- 費用: 読みが有効なときは基本評価と相手仮説評価の両方を行う。追加遷移は1つの予算内だが、分布構築/JSON/評価も増える。人格・通常バッチコア/runtimeは変更しない。',
        '- 省いたもの: 仮説数の無制限増加、相手の隠れた性格への接続、交渉や相手同士/手をまたぐ行動の相関、未知の手札の推測、仮想世界からの自己学習。仮説は各状態の行動分布を混合する方式で、隠れた人物タイプを仮想対戦の開始時に固定抽選する方式ではない。',
        '- 方向性: 「情報が薄いうちは人格に沿う通常判断、観測で傾向が読めれば必要時に対応」の追加。汎用コアをゲーム専用の勝率最大化へ置換していない。教師の行動候補は未採用、価値モデルの学習は未実施。','',
        '## 固定履歴の、次の公開入札の予測','',
        '評価には過去の公開履歴だけを使う。random/reserve/highは既知仮説、switchは6ラウンド後に強い札→弱い札、unseenは未収録の得点機会方策、mixedは相手ごとに違う方策。予測の負の対数尤度は低いほど良い。現在の行動を更新に使う前の確率で計測。最後の強制札を除外。','',
        '|実相手|期間|予測数|仮説予測の損失|一様予測の損失|平均信頼度|','|---|---|---|---|---|---|']
    lines += [f'|{g["controller"]}|{g["phase"]}|{g["predictions"]}|{g["learned_log_loss"]:.3f}|{g["uniform_log_loss"]:.3f}|{g["mean_trust"]:.3f}|' for g in prediction['groups']]
    lines+=['','## 実対戦と計算費用','',
        '各群は3人数×seed。人数/同じseedを再利用する少数比較で、一般的な勝率の保証ではない。上昇と悪化を両方記録。個別の対応条件の差はevaluation.jsonのpaired。','',
        '|相手|目的|方法|局数|平均得点|優勝持分|判断p50 ms|読みで変更した手|','|---|---|---|---|---|---|---|---|']
    lines += [f'|{g["opponent"]}|{g["goal"]}|{g["method"]}|{g["games"]}|{g["mean_score"]:.2f}|{g["mean_win_share"]:.3f}|{g["decision_p50_ms"]:.2f}|{g["reading_changes"]}|' for g in groups]
    lines+=['','方針保持/切替えの内訳: `'+json.dumps(dict(reasons),ensure_ascii=False)+'`。',
        '元の人格方策を数学的な最適解へ変更した実験ではない。予測が正しくても人格の損を伴う選択や近似的な将来方策が成績へ影響する。仮説外・相関した相手への強さは別に確認が必要。',
        '実行はローカルCPUのみ。Colab/GPUは起動していない。ソース/検証は復元ノートブックへ同梱する。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
