"""Situation-driven opponent changes and own adversity, tested separately."""
from collections import Counter
from itertools import permutations,product
import copy
import json
from pathlib import Path
import time
import numpy as np
from .core import Policy,digest
from .laboratory import profiles
from .monte_carlo import RolloutBudget
from .goofspiel import Position,referee,consequence,make_context
from .goofspiel_beliefs import PublicBidBeliefs,under_pressure
from .goofspiel_reading import decide
from .planning import compress_outcomes
from .board_planning import core_audit
from .cross_games import core_hashes
from .teachers import prompt,write_json

MODELS=('legacy4','responsive4','responsive6')
CONTROLLERS=('scheduled','state_attack','state_conserve','unseen','random')


def observer(players,viewer,model):
    if model not in MODELS: raise ValueError('unknown inference ablation')
    return PublicBidBeliefs(players,viewer,contextual=model=='responsive6',responsive=model!='legacy4')


def regime(s,actor,controller):
    if controller=='scheduled': return 'high' if s.round<6 else 'low'
    if controller=='state_attack': return 'high' if under_pressure(s,actor) else 'reserve'
    if controller=='state_conserve': return 'reserve' if under_pressure(s,actor) else 'high'
    if controller in ('random','unseen'): return controller
    raise ValueError('unknown driver controller')


def bid(s,actor,mode,u):
    h=s.hands[actor]
    if mode=='high': return max(h)
    if mode=='low': return min(h)
    if mode=='reserve':
        distances=[abs(c-s.prizes[s.round]) for c in h]; best=min(distances)
        options=[c for c,d in zip(h,distances) if d<=best+1]
        return options[min(int(u*len(options)),len(options)-1)]
    if mode=='random': return h[min(int(u*len(h)),len(h)-1)]
    if mode!='unseen': raise ValueError('unknown driver mode')
    future=sum(s.prizes[s.round+1:])/max(1,sum(s.prizes[s.round:]))
    # Withheld soft opportunity response, not one of the six model kernels.
    risk_cost=.1 if under_pressure(s,actor) else .7
    values=[s.prizes[s.round]*float(np.prod([sum(c<b for c in other)/len(other)
               for i,other in enumerate(s.hands) if i!=actor]))-risk_cost*b*future for b in h]
    w=np.exp((np.array(values)-max(values))/1.5); w/=w.sum()
    return h[min(int(np.searchsorted(np.cumsum(w),u,side='right')),len(h)-1)]


def prediction_probe(seeds=8):
    rows=[]; histories=[]
    for players in (3,4,6):
        for controller in CONTROLLERS:
            for seed in range(seeds):
                s=Position.start(players,order='ascending' if seed%2==0 else 'descending')
                trackers={m:observer(players,0,m) for m in MODELS}; rng=np.random.default_rng(130000+seed*53+players)
                trace=[]; last_mode=None; last_change=None
                while not s.terminal:
                    mode=regime(s,1,controller); changed=last_mode is not None and mode!=last_mode
                    if changed: last_change=s.round
                    age=None if last_change is None else s.round-last_change
                    phase='before_any_change' if age is None else 'at_change' if age==0 else 'first_two_after_change' if age<=2 else 'later_after_change'
                    modes=['random',mode]+[('reserve' if i%2 else 'random') for i in range(2,players)]
                    bids=tuple(bid(s,i,m,float(rng.random())) for i,m in enumerate(modes))
                    diagnostics={m:t.reveal(s,bids)['player-1'] for m,t in trackers.items()}
                    for m,d in diagnostics.items():
                        rows.append(dict(players=players,controller=controller,seed=seed,round=s.round,model=m,
                            phase=phase,mode=mode,changed=changed,**d))
                    after=s.play(bids); referee(s,bids,after)
                    trace.append(dict(round=s.round,scores_before=list(s.scores),public_hands=[list(h) for h in s.hands],
                        joint_bids=list(bids),target_mode=mode,changed=changed,phase=phase,observations=diagnostics))
                    s=after; last_mode=mode
                histories.append(dict(players=players,controller=controller,seed=seed,trace=trace))
    groups=[]
    for controller in CONTROLLERS:
        for phase in ('all','at_change','first_two_after_change','later_after_change'):
            for m in MODELS:
                subset=[r for r in rows if r['controller']==controller and r['model']==m and not r['forced'] and (phase=='all' or r['phase']==phase)]
                if not subset: continue
                groups.append(dict(controller=controller,phase=phase,model=m,predictions=len(subset),
                    log_loss=float(np.mean([r['log_loss'] for r in subset])),
                    uniform_log_loss=float(np.mean([r['uniform_log_loss'] for r in subset])),
                    mean_trust=float(np.mean([r['before']['confidence'] for r in subset]))))
    shocks={m:[r for r in rows if r['model']==m and r['changed'] and not r['forced'] and r['before']['confidence']>0] for m in MODELS}
    change_trust={m:dict(events=len(v),immediate_trust_drops=sum(r['after']['confidence']<r['before']['confidence'] for r in v),
                        mean_before=float(np.mean([r['before']['confidence'] for r in v])) if v else None,
                        mean_after=float(np.mean([r['after']['confidence'] for r in v])) if v else None) for m,v in shocks.items()}
    return dict(seeds=seeds,histories=len(histories),groups=groups,change_trust=change_trust),histories


def adversity_positions(players):
    """Two legal histories; same remaining hands/prizes, own/rival lead swapped."""
    result=[]
    for swap in (False,True):
        s=Position.start(players,7)
        for own,other in zip((2,1,3,4),(1,4,2,3)):
            bids=(own,)+(other,)*(players-1)
            if swap: bids=(bids[1],bids[0])+bids[2:]
            after=s.play(bids); referee(s,bids,after); s=after
        result.append(s)
    return tuple(result)


def exact_small_actions(base):
    """3 players x 3 cards: 216 terminal paths, no finite MC noise.

    Opponent permutations and own subsequent permutations are uniform. Goal
    win-share expectations are exact; compressing the persona effect outcomes
    retains means but still approximates within-bucket risk as usual.
    """
    if len(base.hands)!=3 or any(len(h)!=3 for h in base.hands): raise ValueError('3 players / 3 remaining cards required')
    packed={}; values={}
    for own in base.hands[0]:
        samples=[]; shares=[]
        own_orders=list(permutations(c for c in base.hands[0] if c!=own))
        worlds=list(product(own_orders,permutations(base.hands[1]),permutations(base.hands[2])))
        for rest,a,b in worlds:
            orders=((own,)+rest,a,b); s=base
            for tick in range(3): s=s.play(tuple(h[tick] for h in orders))
            e=consequence(base,s.scores,0,own,'win_share',terminal=True); e['p']=1/len(worlds); samples.append(e)
            shares.append(s.share(0)[0])
        groups={}
        for row in samples:
            key=digest({k:v for k,v in row.items() if k!='p'})
            if key not in groups: groups[key]=dict(row)
            else: groups[key]['p']+=row['p']
        packed[f'BID:{own}']=compress_outcomes(list(groups.values()))
        values[f'BID:{own}']=float(np.mean(shares))
    return packed,values


def score_sensitive_positions():
    # A selected diagnostic fixture, not a held-out performance benchmark.
    orders=(((1,2,3,4),(1,4,6,2),(5,7,1,3)),
            ((3,2,1,4),(2,4,6,1),(7,1,5,3)))
    result=[]
    for history in orders:
        s=Position.start(3,7)
        for t in range(4):
            bids=tuple(h[t] for h in history); after=s.play(bids); referee(s,bids,after); s=after
        result.append(s)
    return tuple(result)


def own_adversity_probe():
    rows=[]
    for case,states in (('symmetric_control',adversity_positions(3)),('score_sensitive_diagnostic',score_sensitive_positions())):
        forecasts=[exact_small_actions(s) for s in states]
        for p in profiles()[:4]:
            for seed in range(4):
                ds=[]
                for s,(outcomes,values) in zip(states,forecasts):
                    c=make_context(s,0,p,'win_share',seed,4,'exact-adversity',outcomes=outcomes)
                    d=Policy().choose(c,stochastic=False)
                    ds.append(dict(action=d['action_id'],expected_share=values[d['action_id']],
                                   goal_values=values,mode=d['next_state']['mode'],scores=list(s.scores)))
                rows.append(dict(case=case,profile=p['id'],seed=seed,leading=ds[0],trailing=ds[1],changed=ds[0]['action']!=ds[1]['action']))
    return dict(pairs=len(rows),changed=sum(r['changed'] for r in rows),rows=rows,
        control='same personality, hands, remaining prizes, scope/seed and decision mode; only legal past score outcome changes',
        limits='two selected legal situation pairs x 4 personalities x 4 mode seeds, not 32 independent game cases; no automatic panic/risk rule')


def play(players,seed,controller,model,teacher_sink=None):
    s=Position.start(players,order='ascending' if seed%2==0 else 'descending'); seat=seed%players; target=(seat+1)%players
    tracker=observer(players,seat,model); profile=profiles()[0 if seed%2==0 else 3]; memory=None; trace=[]; times=[]
    rng=np.random.default_rng(140000+seed*59+players); last_mode=None
    budget=RolloutBudget(samples=16,min_samples=4,max_steps=13,rollout_policy='random')
    while not s.terminal:
        start=time.perf_counter()
        c,d,stats=decide(s,seat,profile,'win_share',seed,s.round,f'adaptive-{players}-{seed}',memory,budget,tracker.snapshot())
        times.append((time.perf_counter()-start)*1000)
        if teacher_sink is not None and model=='responsive6' and stats['reading_applied'] and len(teacher_sink)<12:
            tag=[profile['id'],stats['reason']]
            if not any(r['tag']==tag for r in teacher_sink):
                teacher_sink.append(dict(id=f'adaptive-reading-{len(teacher_sink):03d}',tag=tag,context=copy.deepcopy(c),
                    context_hash=digest(c),prompt=prompt(c),quality='awaiting_generation',source='pre_decision_snapshot_no_gold'))
        mode=regime(s,target,controller); switched=last_mode is not None and mode!=last_mode
        bids=[]
        for i in range(players):
            u=float(rng.random())  # one driver-only variate per actor and round
            bids.append(int(d['action_id'].split(':')[1]) if i==seat else bid(s,i,mode if i==target else ('reserve' if i%2 else 'random'),u))
        after=s.play(tuple(bids)); referee(s,bids,after); observations=tracker.reveal(s,bids)
        trailing=s.scores[seat]<max(s.scores); pressure=under_pressure(s,seat)
        trace.append(dict(round=s.round,scores_before=list(s.scores),public_hands=[list(h) for h in s.hands],joint_bids=bids,
            target_mode=mode,target_switched=switched,own_trailing=trailing,own_pressure=pressure,
            own_action=d['action_id'],reading=stats,observations=observations,scores_after=list(after.scores)))
        assert stats['total_nodes']<=budget.max_nodes
        last_mode=mode; memory=d['next_state']; s=after
    share,_=s.share(seat)
    return dict(players=players,seed=seed,seat=seat,target=target,controller=controller,model=model,profile=profile['id'],
                scores=list(s.scores),own_score=s.scores[seat],discarded=s.discarded,win_share=share,finished=s.terminal,
                decision_ms=times,trace=trace)


def experiment(root,seeds=4,prediction_seeds=8,progress=None):
    if any(type(n) is not int or n<1 for n in (seeds,prediction_seeds)): raise ValueError('positive seed counts required')
    root=Path(root); root.mkdir(parents=True,exist_ok=True); start=time.perf_counter(); runs=[]; teachers=[]
    for players in (3,4,6):
        for seed in range(seeds):
            for controller in ('state_attack','state_conserve','unseen'):
                for m in MODELS: runs.append(play(players,seed,controller,m,teachers))
        if progress: progress(f'{players}人の劣勢時変更検査完了、累計{len(runs)}局')
    prediction,histories=prediction_probe(prediction_seeds); own=own_adversity_probe(); groups=[]
    for controller in ('state_attack','state_conserve','unseen'):
        for m in MODELS:
            subset=[r for r in runs if (r['controller'],r['model'])==(controller,m)]
            ts=[t for r in subset for t in r['trace']]; durations=[d for r in subset for d in r['decision_ms']]
            groups.append(dict(controller=controller,model=m,games=len(subset),mean_score=float(np.mean([r['own_score'] for r in subset])),
                mean_share=float(np.mean([r['win_share'] for r in subset])),decision_p50_ms=float(np.median(durations)),
                own_trailing_decisions=sum(t['own_trailing'] for t in ts),changes_while_trailing=sum(t['own_trailing'] and t['reading']['action_changed'] for t in ts),
                target_switches=sum(t['target_switched'] for t in ts)))
    paired=[]
    for learned in [r for r in runs if r['model']!='legacy4']:
        old=next(r for r in runs if (r['players'],r['seed'],r['controller'],r['model'])==(learned['players'],learned['seed'],learned['controller'],'legacy4'))
        paired.append(dict(players=learned['players'],seed=learned['seed'],controller=learned['controller'],model=learned['model'],
                           score_delta=learned['own_score']-old['own_score'],share_delta=learned['win_share']-old['win_share']))
    result=dict(games=len(runs),finished=sum(r['finished'] for r in runs),referee_mismatches=0,groups=groups,paired=paired,
        prediction_probe=prediction,own_adversity_probe=own,core_unchanged=core_audit(),core_hashes=core_hashes(),
        teacher_requests=len(teachers),accepted_llm_rows=0,elapsed_seconds=time.perf_counter()-start,
        protocol=dict(players=[3,4,6],seeds=seeds,prediction_seeds=prediction_seeds,models=list(MODELS),goal='win_share',
            pressure_rule='gap to highest public score >= half of maximum prize value; game-owned proxy for material disadvantage',
            hypotheses='4 global bases + 2 finite public-score-conditioned bases',
            inference='surprise below half-uniform probability weakens previous logs by retention x 0.25; a decaying surprise penalty reduces trust',
            distinction='own choice under adversity vs predicting opponent situation-driven changes are separate tests',
            fairness='paired seed/seat/personality/prizes, one driver variate per actor/round, common MC variates; live histories diverge after choices',
            limits='small fixed-case comparisons, not calibrated confidence or universal strategy-switching intelligence; no trained value model'))
    write_json(root/'evaluation.json',result); write_json(root/'trajectories.json',runs); write_json(root/'prediction_histories.json',histories)
    (root/'teacher_requests.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in teachers),encoding='utf-8')
    lines=['# 劣勢に応じる方針変更と、相手の変化の予測','',
        '旧検査は6手目に最大札→最小札へ変更する時計条件だった。旧モデルの変更後区間の予測損失1.872は、一様予測1.421より悪かった。変化後の再学習が遅い点を改善し、状況条件の変更を別に検査する。','',
        '## 得たもの・費用・残る限界','',
        '- 得たもの: 一様予測の半分未満の確率しか与えていなかった公開行動を見たら、過去の重みを強く弱める。外れの余波を1つのスカラーで減衰させ、直後の過信を抑える。次の方策を1回の観測で確定しない。',
        '- 得たもの: 公開得点から不利が大きい状況を判定し、温存→強い札、強い札→温存の2条件付き仮説を追加。全6仮説で有限のまま。人物の性格を変更する処理ではない。',
        '- 得たもの: こちら自身の劣勢時の選択を、同じ人格/手札/残りの得点札/seedの合法局面対で検査。相手の傾向推定とは分ける。',
        '- 費用/近似: 仮説が4→6になり、少数観測ではどの方策かを絞りにくい。急な外れを雑音と区別できず、誤検知で良い仮説も弱める可能性がある。閾値と信頼度は未校正の設計値。',
        '- 省いたもの: 無制限の状況分類、万能な負け時のリスク上昇、架空の結果での実観測学習。旧実験はlegacy4を明示して保持する。',
        '- 方向性: 性格は固定し、公開状況に応じた選択/行動見込みを更新。人格コア/runtimeは変更なし。不利だから必ず方針転換するのではなく、目的と人格に照らして選ぶ。','',
        '不利の代理値は、最多得点との差が得点札最大値の半分以上。順位そのものや実勝率の真値ではない。ゲームがこの基準を変える。状況変更を起こす対象は1人、他者は別方策。実相手の方策名/現在の未公開入札は予測へ渡さない。','',
        '## 同じ公開履歴の変更後予測','',
        'legacy4=旧4仮説、responsive4=急な外れへの見直しのみ、responsive6=見直し＋状況条件仮説。低い予測損失が良い。強制の最後の札は除外。予測は現在の行動を更新する前の確率で測定。','',
        '|実相手|期間|モデル|予測数|予測損失|一様損失|信頼度|','|---|---|---|---|---|---|---|']
    lines += [f'|{g["controller"]}|{g["phase"]}|{g["model"]}|{g["predictions"]}|{g["log_loss"]:.3f}|{g["uniform_log_loss"]:.3f}|{g["mean_trust"]:.3f}|' for g in prediction['groups']]
    lines+=['','状況条件を持つ6仮説でも万能ではない。時計での変更と公開状況での変更を分け、変更直後/その後/未収録相手を別々に評価する。既定実験では、劣勢による攻めへの変更は予測しやすくなった一方、時計変更後の最初の2手は6仮説が一様予測より悪かった。外れへの見直しのみの4仮説が6仮説より良い条件もある。']
    lines+=['','## 実対戦での劣勢と切替え','',f'{len(runs)}局すべて終了、独立審判との不一致0。各群は3人数×seedの少数比較。仮説の切替えが正しい/勝てる保証として扱わない。','',
        '|相手|モデル|局数|平均得点|優勝持分|p50 ms|劣勢で判断した回数|その中の読みで変更|相手の切替回数|',
        '|---|---|---|---|---|---|---|---|---|']
    lines += [f'|{g["controller"]}|{g["model"]}|{g["games"]}|{g["mean_score"]:.2f}|{g["mean_share"]:.3f}|{g["decision_p50_ms"]:.2f}|{g["own_trailing_decisions"]}|{g["changes_while_trailing"]}|{g["target_switches"]}|' for g in groups]
    lines+=['','旧4仮説との同条件比較（改善/同じ/悪化は優勝持分）:']
    for m in MODELS[1:]:
        subset=[r for r in paired if r['model']==m]
        lines.append(f'- {m}: {len(subset)}対中 {sum(r["share_delta"]>0 for r in subset)}改善 / {sum(r["share_delta"]==0 for r in subset)}同じ / {sum(r["share_delta"]<0 for r in subset)}悪化。')
    lines+=['','予測を直すことと、良い対処を選ぶことは別。成績が悪化する条件が残るため、6仮説への追加を総合的な強さの改善とは扱わない。']
    lines+=['','## 自分の劣勢だけを変えた対照','',
        '3人・各7枚の合法履歴から、手札/残り得点札を固定した得点だけの違う2種類の対を作る。対称な手札の対は全候補の期待持分が同じになる負の対照。非対称な手札の対は、相手の点数だけが変わって自分が先行→劣勢となる診断用の選択局面。一様継続の216終局経路を全列挙し、期待持分を計算。人格特徴は同じで、決定乱数を使わない。結果の8枠圧縮は人格リスクをなお近似する。',
        f'2種類×4人格×4モードseedの{own["pairs"]}対のうち、選択が変わったのは{own["changed"]}対。同じ診断局面を再利用した検査で、32種類の独立ゲーム局面ではない。選択差を確認するため選んだ局面なので未見性能の検証にしない。明細はevaluation.jsonのown_adversity_probe。',
        '固定したgrowth人格の診断例では、先行時に札5、劣勢時に札6を選んだ。劣勢時の優勝持分の期待値は札5が0.750、札6が0.806。これは宣言した一様継続仮説での全列挙値で、実相手に対する真の勝率ではない。',
        '変わらないことも誤りとは限らない。負けているだけで見込みの薄い大勝負へ強制的に飛びつく設計にはしていない。',
        '実行はローカルCPUのみ。Colab/GPUのセッションは起動していない。教師相談は未採用。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
