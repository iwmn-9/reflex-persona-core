"""Headless four-player single-kyoku test, with explicit game-side proxies."""
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
import time
import numpy as np
from .core import Policy,compile_batch,digest
from .laboratory import profiles
from .cross_games import core_hashes
from .mahjong_adapter import (ENGINE_VERSION,PublicView,action_id,action_type,
    candidates,choose,engine,make_context,reference,shanten,remove_tile)
from .teachers import write_json


def audit(env):
    """Driver-only hidden-state conservation check, never used for choosing."""
    tiles=[t for hand in env.hands for t in hand]+[t for river in env.discards for t in river]+list(env.wall)
    assert len(tiles)==136 and set(tiles)==set(range(136))
    assert not any(env.melds) # scope: every controller follows the closed protocol
    assert sum(env.scores())+1000*env.riichi_sticks==100000


def play(seed,profile_index,method,opponent,probes=None):
    if method not in ('random','efficiency','core','core_own_draw'):raise ValueError('Unknown method')
    env=engine().RiichiEnv(game_mode='4p-red-single',seed=seed)
    observations=env.reset();seat=seed%4;profile=profiles()[profile_index]
    memory=None;trace=[];timings=[];initial=env.scores();audit(env)
    for tick in range(600):
        if env.done():break
        actions={};entries=[]
        for player,obs in sorted(observations.items()):
            legal=obs.legal_actions();view=PublicView.from_observation(obs)
            start=time.perf_counter();stats=None;decision=None
            if player==seat and method.startswith('core'):
                selected,c,decision,stats=choose(obs,profile,seed,tick,f'mahjong-{seed}',memory,
                    lookahead_samples=16 if method=='core_own_draw' else 0)
                memory=decision['next_state'];timings.append((time.perf_counter()-start)*1000)
                # Individual vs batch equivalence tested on saved same-state probes.
            else:
                selected=reference(obs,seed,tick,method if player==seat else opponent)
                if player==seat:timings.append((time.perf_counter()-start)*1000)
            assert action_id(selected) in {action_id(a) for a in legal}
            actions[player]=selected
            if probes is not None and method=='efficiency' and player==seat and any(action_type(a)=='Discard' for a in legal):
                if len(probes)<160:probes.append(dict(seed=seed,tick=tick,opponent=opponent,observation=obs.serialize_to_base64()))
            row=dict(player=player,own=player==seat,public=asdict(view),action=action_id(selected),
                     engine_legal=[action_id(a) for a in legal],decision=decision,stats=stats)
            entries.append(row)
        before=len(env.mjai_log);observations=env.step(actions);audit(env)
        trace.append(dict(tick=tick,decisions=entries,events=env.mjai_log[before:]))
    else:raise AssertionError('A single kyoku failed to terminate within cap')
    scores=env.scores();best=max(scores);credits=[1/sum(x==best for x in scores) if x==best else 0 for x in scores]
    events=[e for t in trace for e in t['events']]
    return dict(seed=seed,seat=seat,profile=profile['id'] if method.startswith('core') else None,
        method=method,opponent=opponent,finished=env.done(),scores=scores,win_share=credits[seat],
        point_delta=scores[seat]-initial[seat],rank=env.ranks()[seat],riichi_sticks=env.riichi_sticks,
        agari=sum(e.get('type')=='hora' and e.get('actor')==seat for e in events),
        deal_in=sum(e.get('type')=='hora' and e.get('actor')!=seat and e.get('target')==seat for e in events),
        riichi=sum(e.get('type')=='reach' and e.get('actor')==seat for e in events),
        trace=trace,decision_ms=timings,steps=len(trace))


def same_state_probes(pool,limit=60):
    # Prioritize real public-riichi situations, then cover ordinary decisions.
    parsed=[]
    for row in pool:
        obs=engine().Observation.deserialize_from_base64(row['observation'])
        v=PublicView.from_observation(obs)
        parsed.append((any(v.riichi[i] for i in range(4) if i!=v.player),row,obs))
    parsed.sort(key=lambda x:not x[0]);rows=[];batch_checks=0
    for threat,source,obs in parsed[:limit]:
        choices=[];contexts=[];decisions=[]
        for profile in profiles():
            a,c,d,s=choose(obs,profile,source['seed'],source['tick'],'same-state-probe')
            choices.append(dict(profile=profile['id'],action=action_id(a),features=s['selected']['features'],
                                exposure=s['selected']['exposure']))
            contexts.append(c);decisions.append(d)
        b=compile_batch(contexts);ds=Policy().decide(b,stochastic=False).records(b)
        assert [d['action_id'] for d in ds]==[d['action_id'] for d in decisions]
        assert [d['next_state'] for d in ds]==[d['next_state'] for d in decisions]
        batch_checks+=1
        a,c,d,s=choose(obs,profiles()[0],source['seed'],source['tick'],'same-state-probe',lookahead_samples=16)
        rows.append(dict(seed=source['seed'],tick=source['tick'],opponent=source['opponent'],threatened=threat,
            public=asdict(PublicView.from_observation(obs)),choices=choices,
            own_draw_choice=action_id(a),lookahead_changed=action_id(a)!=choices[0]['action'],
            personality_difference=len({x['action'] for x in choices})>1))
    return dict(cases=len(rows),personality_differences=sum(r['personality_difference'] for r in rows),
        threatened_cases=sum(r['threatened'] for r in rows),lookahead_changes=sum(r['lookahead_changed'] for r in rows),
        batch_equal_checks=batch_checks,rows=rows)


def discard_metric(row):
    """Measure actual public discard for ALL controllers, not only core logs."""
    if not row['action'].startswith('Discard:'):return None
    v=PublicView(**row['public']);tile=int(row['action'].split(':')[1])
    current=shanten(remove_tile(v.hand,tile))
    best=min(shanten(remove_tile(v.hand,int(a.split(':')[1]))) for a in row['engine_legal'] if a.startswith('Discard:'))
    return dict(worsens=current>best,threatened=any(v.riichi[i] for i in range(4) if i!=v.player),exposure=v.exposure(tile))


def experiment(root,seeds=4,progress=None):
    if type(seeds) is not int or seeds<1:raise ValueError('Positive seed count required')
    if version('riichienv')!=ENGINE_VERSION:raise RuntimeError('Use pinned optional engine version')
    root=Path(root);root.mkdir(parents=True,exist_ok=True);start=time.perf_counter();runs=[];pool=[]
    for seed in range(seeds):
        for opponent in ('random','efficiency'):
            for method in ('random','efficiency'):runs.append(play(seed,0,method,opponent,pool))
            for i in range(4):
                for method in ('core','core_own_draw'):runs.append(play(seed,i,method,opponent))
        if progress:progress(f'麻雀 seed {seed}: {len(runs)}局終了（4人の1局、門前の初期基線）')
    groups=[]
    for opponent in ('random','efficiency'):
        for method in ('random','efficiency','core','core_own_draw'):
            for profile in ([p['id'] for p in profiles()] if method.startswith('core') else [None]):
                rr=[r for r in runs if (r['opponent'],r['method'],r['profile'])==(opponent,method,profile)]
                own=[d for r in rr for t in r['trace'] for d in t['decisions'] if d['own']]
                discards=[m for m in (discard_metric(d) for d in own) if m is not None]
                threatened=[d for d in discards if d['threatened']]
                groups.append(dict(opponent=opponent,method=method,profile=profile,games=len(rr),
                    agari=sum(r['agari'] for r in rr),deal_in=sum(r['deal_in'] for r in rr),riichi=sum(r['riichi'] for r in rr),
                    mean_points=float(np.mean([r['point_delta'] for r in rr])),mean_share=float(np.mean([r['win_share'] for r in rr])),
                    p50_ms=float(np.median([v for r in rr for v in r['decision_ms']])),
                    p95_ms=float(np.quantile([v for r in rr for v in r['decision_ms']],.95)),
                    shanten_worsening=sum(d['worsens'] for d in discards),
                    worsening_without_riichi=sum(d['worsens'] for d in discards if not d['threatened']),
                    threatened_discards=len(threatened),genbutsu_to_all=sum(d['exposure']==0 for d in threatened)))
    paired=[]
    for r in runs:
        if r['method']!='core_own_draw':continue
        old=next(o for o in runs if (o['seed'],o['opponent'],o['profile'],o['method'])==(r['seed'],r['opponent'],r['profile'],'core'))
        paired.append(dict(seed=r['seed'],opponent=r['opponent'],profile=r['profile'],point_delta=r['point_delta']-old['point_delta'],
                           share_delta=r['win_share']-old['win_share']))
    result=dict(games=len(runs),finished=sum(r['finished'] for r in runs),groups=groups,paired=paired,
        probes=same_state_probes(pool),core_hashes=core_hashes(),elapsed_seconds=time.perf_counter()-start,
        protocol=dict(engine='riichienv',version=ENGINE_VERSION,mode='4p-red-single',seeds=seeds,
            scope='all controllers: legal win/riichi/discard/pass; no calls, kans, abort selection; one kyoku, not hanchan',
            game_evaluation='handwritten game-side shanten/ukeire progress and genbutsu exposure proxies',
            proxy_version=3,progress_formula='.90*(6-shanten)/6+.10*ukeire/unseen_count',
            exposure_cost='.02 * public exposure; an uncalibrated common cost, not P(ron)',
            own_draw='16 samples from public unseen tile pool; optimal next own discard; no opponent progression or terminal EV',
            future_progress='complete shape 1, otherwise .90*(6-next_shanten)/6; tenpai below completion',
            truth='full engine legal actions, yaku/furiten/payment processing; driver conservation checks, not independent scoring referee',
            limits='not rule-only/general score discovery/trained policy/opponent reading/retained rolling plan; exposures not calibrated probabilities',
            llm='no generation, adoption or training',runtime='local CPU only; no Colab launched'))
    write_json(root/'evaluation.json',result);write_json(root/'trajectories.json',runs)
    write_json(root/'probe_observations.json',pool);write_report(root,result)
    return result


def write_report(root,r):
    p=r['probes'];paired=r['paired']
    nonthreat=sum(g['worsening_without_riichi'] for g in r['groups'] if g['method'].startswith('core'))
    lines=['# 麻雀で人格判断コアを試す：門前・4人・1局の初期基線','',
        '麻雀は難しいが、既存の日本式麻雀エンジンを使い、同じ人格コアへ接続できた。今回の成功は接続と限定した判断の検査であり、麻雀の強さや汎用スコアの自動獲得を達成したとは扱わない。','',
        f'今回の範囲では立直脅威のない場面でのシャンテン悪化は人格コア方式全体で{nonthreat}回。同じ実局面{p["cases"]}件のうち人格差は{p["personality_differences"]}件。次ツモ評価は点差で{sum(x["point_delta"]>0 for x in paired)}改善/{sum(x["point_delta"]==0 for x in paired)}同じ/{sum(x["point_delta"]<0 for x in paired)}悪化だった。4 seedを複数人格で反復し、同じ行動になる人格もあるため、これを独立した多数の棋力証拠とは数えない。','',
        '## 得たもの・削ったもの','',
        '- 得たもの: [RiichiEnv](https://github.com/smly/RiichiEnv) 0.4.10の実際の4人日本式麻雀を描画なしで進行。和了の役/フリテン/支払い/立直の制限は外部エンジンへ任せる。',
        '- 得たもの: 自分の牌と公開捨牌/立直だけを人格評価へ渡す。実際の他家手牌や山順は見せない。全4人格を同じ局面で比較し、数値バッチと個別の一致も検査。',
        '- 加えたもの: 麻雀側のシャンテン数、受入れ枚数、立直相手の現物かどうかによる露出指数。手書きのゲーム別評価であり、従来のルール/終局報酬だけの基線とは別。人格コア本体は変更しない。',
        '- 加えたもの: 公開情報から未見牌を16回抽選し、自分の次の1ツモ後の最善打牌まで評価する対照。局の終わり、相手の進行、役/得点の期待値を読む探索ではない。',
        '- 削ったもの: 初期比較では全員の鳴き/カン/途中流局の選択を対象外とし、門前の打牌/立直/和了/パスにそろえた。外部エンジンの合法手は記録し、制限を隠さない。半荘の順位目的も保留。',
        '- 残るもの: 非立直の危険、筋/壁/待ち/打点の推測、立直棒と順位を含む終局EV、他者傾向の学習、継続計画の評価。完成形だけの先読み値には役やフリテンの保証がない。','',
        '## 評価契約','',
        '`progress = .90 * (6-shanten)/6 + .10 * ukeire/unseen_count`。シャンテン1段の重みが受入れ全範囲より大きい。立直と打牌にも同じ次ツモ条件を適用し、立直の役獲得には小さな固定加点、拘束には固定露出費用を置く。いずれもゲーム用仮定で、学習/校正済み期待値ではない。',
        '受入れを重く見すぎた初版で、立直のない場面でも手進行を壊す選択が出たため、上記の重み制約を加えた。危険回避のために手を崩す選択は別に残す。任意先読みを打牌だけへ加えて立直との比較が不公平になる問題も修正した。初版のコード/結果はローカルbackupへ保全。',
        '達成主義には進行代理値、安定主義には立直相手への未知の放銃露出を渡す。同じ進行なら不要な露出を避けられるよう、全人格に露出×.02の小さい共通費用も渡す。現物はルールに基づくが、露出指数を放銃確率や実安全率と読み替えない。欲求や未対応の主義効果は発明しない。',
        '次ツモ後の代理値は完成形1、それ以外は`.90*(6-next_shanten)/6`。テンパイだけなら.90で、完成と区別する。立直に使うテンパイ形では、非和了ツモをそのまま捨てても同じ最小シャンテンを維持できる。相手への放銃や待ち/打点の変化はこの代理値で評価しない。',
        '実対戦では3人の機械的ランダム（合法和了/立直は取る）または手進行参照（最小シャンテン/最大受入れ）と比較。同じseed/席/相手で直接評価と次ツモ評価を比較するが、行動後の履歴は分岐する。少数seedの結果で強さを断定しない。','',
        '## 実行結果','',f'{r["games"]}局、終了{r["finished"]}。毎遷移で136枚の物理牌保存・門前・得点と立直棒の保存を検査。独立した得点審判を実装したとは扱わない。','',
        '|相手|方式|人格|局数|和了|放銃|立直|平均点差|最高得点持分|p50 ms|p95 ms|シャンテン悪化|立直脅威なしの悪化|脅威下打牌|全宣言者の現物|',
        '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for g in r['groups']:
        lines.append(f'|{g["opponent"]}|{g["method"]}|{g["profile"]}|{g["games"]}|{g["agari"]}|{g["deal_in"]}|{g["riichi"]}|{g["mean_points"]:.0f}|{g["mean_share"]:.3f}|{g["p50_ms"]:.2f}|{g["p95_ms"]:.2f}|{g["shanten_worsening"]}|{g["worsening_without_riichi"]}|{g["threatened_discards"]}|{g["genbutsu_to_all"]}|')
    lines+=['','最高得点持分（JSONのwin_share）は、この1局の終了時に最も点数を持つ人へ配分した指標。同点は分割し、和了率や半荘の勝率とは区別する。',
        '',f'同じ局面{p["cases"]}件、人格間で打牌等が異なった局面{p["personality_differences"]}件。そのうち立直脅威を持つ検査対象は{p["threatened_cases"]}件。バッチ/個別一致{p["batch_equal_checks"]}件。次ツモ評価で挑戦志向の選択が変わった局面{p["lookahead_changes"]}件。','',
        '現時点の人格差は主に安定志向と手進行志向の違い。未対応の慈善/権力等を無理に麻雀の効果へ割り当てず、4人格すべてが豊かに異なるとは主張しない。',
        f'次ツモ評価と直接評価の同条件{len(paired)}比較、点差で改善{sum(x["point_delta"]>0 for x in paired)}・同じ{sum(x["point_delta"]==0 for x in paired)}・悪化{sum(x["point_delta"]<0 for x in paired)}。これは短い自己手牌評価の比較で、前回のRolling Horizon維持方式の強さを測った結果ではない。','',
        '通常の数値Policyだけの速度と、牌効率計算/JSON/乱数を含むこのadapterの時間は別。CPU・シャンテンキャッシュと実行順に依存し、後に実行した人格はキャッシュの恩恵を受ける。公平な冷キャッシュ性能比較や大規模な多数NPCの実効速度は未検証。',
        '実軌跡のeventsは審査用の世界ログで、非公開のツモも含む。教師入力や人格の観測へそのまま渡さない。同じ局面のprobe_observationsは一人分にマスクした観測。',
        '','人格コア/runtime不変。LLM生成/採用/学習なし。Colab起動/Drive反映なし。']
    (Path(root)/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
