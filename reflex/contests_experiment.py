"""Reproducible public-feedback contests; learning is online prediction, not training."""
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
import json
import math
import time
import numpy as np
from .core import Policy, digest
from .laboratory import profiles
from .cross_games import core_hashes
from .teachers import write_json
from .contests import Public, PRIZES, PublicBeliefs, target, settle, decide, public_from_record

SCENARIOS={'pattern':('value','high','frugal'),
           'pressure':('pressure','frugal','value'),
           'responsive':('response','pressure','frugal')}
MODES=('uniform','learned','gated')


def opening(game, seed, episode, players=4):
    rng=np.random.default_rng(int(digest([game,seed,episode,'world-deck'])[:16],16))
    deck=list(PRIZES if game=='hagetaka' else (2,3,4,5,6,7)*2)
    rng.shuffle(deck)
    categories=tuple(('science','culture','power')[int(x)] for x in rng.integers(0,3,len(deck)))
    hands=(tuple(range(1,16)),)*players if game=='hagetaka' else ((),)*players
    s=Public(game,hands,(0,)*players,(0 if game=='hagetaka' else 18,)*players,
             int(deck[0]),categories[0],tuple(sorted(deck[1:])),episode=episode)
    return s,tuple(deck),categories


def terminal_scores(s):
    return np.asarray(s.scores,dtype=float)+(.25*np.asarray(s.budgets) if s.game=='auction' else 0)


def run(game,profile,seed,scenario,mode,episodes=2,record=True,samples=32,capital_pricing=True):
    seat=seed%4; observer=PublicBeliefs(4,seat); memory=None; tick=0
    games=[]; trace=[]; timings=[]; reasons=Counter(); predictions=[]; nodes=0
    for episode in range(episodes):
        s,deck,categories=opening(game,seed,episode)
        memory=None; changed=0; losses=0; collisions=0; payments=0
        controllers=iter(SCENARIOS[scenario]); names={i:next(controllers) for i in range(4) if i!=seat}
        for j in range(len(deck)):
            before=s
            t=time.perf_counter()
            c,d,stats=decide(s,seat,profile,seed,tick,memory,observer,mode,samples,capital_pricing)
            timings.append((time.perf_counter()-t)*1000); memory=d['next_state']; tick+=1
            # All opponents choose from the SAME public state. Actual controllers
            # exist only here, outside the actor/prediction interface.
            bids=tuple(int(d['action_id'].split(':')[1]) if i==seat else target(s,i,names[i]) for i in range(4))
            for actor,b in enumerate(bids):
                if actor==seat or len(s.legal(actor))<2: continue
                forecast=observer.forecast(s,actor)
                predictions.append(dict(learned=-math.log(max(forecast[b],1e-12)),
                    uniform=math.log(len(s.legal(actor))),scenario=scenario))
            audit=observer.reveal(s,bids)
            after,winner=settle(s,bids)
            reasons[stats['reason']]+=1; nodes+=stats['joint_resolutions']; changed+=stats['action_changed']
            losses+=max(0,before.scores[seat]-after.scores[seat])
            collisions+=sum(b==bids[seat] for b in bids)>1
            payments+=before.budgets[seat]-after.budgets[seat]
            if j+1<len(deck):
                after=replace(after,prize=int(deck[j+1]),category=categories[j+1],remaining=tuple(sorted(deck[j+2:])))
            if record:
                trace.append(dict(before=asdict(before),bids=list(bids),winner=winner,
                    context=c,decision=d,reading=stats,observed=audit,after=asdict(after)))
            s=after
        totals=terminal_scores(s); winners=np.flatnonzero(totals==totals.max()).tolist()
        games.append(dict(episode=episode,score=float(totals[seat]),margin=float(totals[seat]-max(x for i,x in enumerate(totals) if i!=seat)),
            won=seat in winners,win_credit=1/len(winners) if seat in winners else 0,
            winners=winners,negative_points=losses,collision_rounds=collisions,payment=payments,
            unused_budget=s.budgets[seat],unawarded_last_pot=s.carry,changes=changed,final=asdict(s)))
    return dict(game=game,profile=profile['id'],seed=seed,seat=seat,scenario=scenario,mode=mode,capital_pricing=capital_pricing,
        episodes=games,reasons=dict(reasons),joint_resolutions=nodes,
        decision_p50_ms=float(np.median(timings)),decision_p95_ms=float(np.percentile(timings,95)),
        prediction_logloss=float(np.mean([x['learned'] for x in predictions])),
        uniform_logloss=float(np.mean([x['uniform'] for x in predictions])),trace=trace)


def replay(row):
    checked=0; observer=PublicBeliefs(4,row['seat'])
    for item in row['trace']:
        s=public_from_record(item['before']); after,winner=settle(s,tuple(item['bids']))
        assert winner==item['winner']
        expected=public_from_record(item['after'])
        assert replace(after,prize=expected.prize,category=expected.category,remaining=expected.remaining)==expected
        assert Policy().choose(item['context'])==item['decision']
        assert item['context']['scope']['npc']==f"player-{row['seat']}"
        audit=observer.reveal(s,tuple(item['bids']))
        assert audit==item['observed']
        checked+=1
    return checked


def experiment(output,seeds=4,episodes=2,samples=32,progress=None):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    before=core_hashes(); runs=[]; checks=0
    trajectory=output/'trajectories.jsonl'
    with trajectory.open('w',encoding='utf-8') as stream:
        for game in ('hagetaka','auction'):
            for scenario in SCENARIOS:
                for profile in profiles():
                    for seed in range(seeds):
                        for mode in MODES:
                            row=run(game,profile,seed,scenario,mode,episodes,samples=samples)
                            checks+=replay(row); stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                            runs.append({k:v for k,v in row.items() if k!='trace'})
                if progress: progress(f'{game}/{scenario}: {len(runs)*episodes} games, {checks} public decisions replayed')
    summary=[]
    for game in ('hagetaka','auction'):
        for scenario in SCENARIOS:
            for mode in MODES:
                group=[r for r in runs if (r['game'],r['scenario'],r['mode'])==(game,scenario,mode)]
                games=[g for r in group for g in r['episodes']]; reasons=Counter()
                for r in group: reasons.update(r['reasons'])
                summary.append(dict(game=game,scenario=scenario,mode=mode,games=len(games),
                    win_credit=sum(g['win_credit'] for g in games),wins=sum(g['won'] for g in games),
                    mean_score=float(np.mean([g['score'] for g in games])),mean_margin=float(np.mean([g['margin'] for g in games])),
                    action_changes=sum(g['changes'] for g in games),negative_points=sum(g['negative_points'] for g in games),
                    collision_rounds=sum(g['collision_rounds'] for g in games),payment=sum(g['payment'] for g in games),
                    prediction_logloss=float(np.mean([r['prediction_logloss'] for r in group])),
                    uniform_logloss=float(np.mean([r['uniform_logloss'] for r in group])),
                    campaign_median_p50_ms=float(np.median([r['decision_p50_ms'] for r in group])),
                    joint_resolutions=sum(r['joint_resolutions'] for r in group),gate_reasons=dict(reasons)))
    paired=[]
    for r in runs:
        if r['mode']=='uniform': continue
        b=next(x for x in runs if (x['game'],x['scenario'],x['profile'],x['seed'],x['mode'])==(r['game'],r['scenario'],r['profile'],r['seed'],'uniform'))
        for x,y in zip(r['episodes'],b['episodes']):
            paired.append(dict(game=r['game'],scenario=r['scenario'],profile=r['profile'],seed=r['seed'],episode=x['episode'],mode=r['mode'],
                margin_delta=x['margin']-y['margin'],win_credit_delta=x['win_credit']-y['win_credit']))
    assert before==core_hashes()
    result=dict(format='public-contests-v1',games=len(runs)*episodes,seeds=seeds,episodes=episodes,samples=samples,
        replayed_decisions=checks,core_before=before,core_after=core_hashes(),summary=summary,paired=paired,runs=runs,
        limitations=['development scenarios, not held-out intelligence evidence',
            'current-round MC and hand/budget commitment proxy, not full-game win EV',
            'auction future capital price assumes equal shares of remaining common item value; not calibrated',
            'finite public behavior templates; imperfect change detection; confidence is heuristic',
            '8-outcome compression loses some distribution/correlation detail',
            'public perfect bid ledger; unresolved final Hagetaka pot remains unawarded',
            'authored common-value auction, not a named board game or private-value equilibrium',
            'reported campaign wall times can include other local jobs; not isolated throughput benchmarks',
            'same seeded decks; public feedback diverges, so paired results are not same-state causal effects'])
    write_report(result,output)
    return result


def write_report(result,output):
    output=Path(output); summary=result['summary']; checks=result['replayed_decisions']
    result['auction_capital_price']='remaining common-value equal share / own cash, clipped .25..2'
    for note in ('auction future capital price assumes equal shares of remaining common item value; not calibrated',
                 'reported campaign wall times can include other local jobs; not isolated throughput benchmarks'):
        if note not in result['limitations']: result['limitations'].append(note)
    write_json(output/'evaluation.json',result)
    lines=['# 公開履歴によるハゲタカ・有限予算オークション','',
        f'{result["games"]} games / {checks} judgments replayed. Shared core/runtime unchanged. No LLM or GPU.', '',
        '|game|opponents|reading|games|win credit|mean margin|changed actions|median p50 ms|',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for s in summary:
        lines.append(f'|{s["game"]}|{s["scenario"]}|{s["mode"]}|{s["games"]}|{s["win_credit"]:.2f}|{s["mean_margin"]:.2f}|{s["action_changes"]}|{s["campaign_median_p50_ms"]:.2f}|')
    lines+=['','uniform: no reading. learned: always score with observed predictions. gated: only change for material persona benefit; preserve currently favorable one-round forecast.',
        '', '## 得たものと費用', '',
        'ゲーム名に依存しない人格Policyへ、公開情報だけの入札分布を接続。相手は固定傾向、得点で不利になると変更、過去の公開入札に応答する3条件。選択前の同一盤面で全員が決め、公開後だけ観測する。',
        '読みを使う場合は通常評価に加え、同じ候補を再評価する。各モデルは全合法候補×32標本（最大480解決）、両モデル最大960。1ゲーム先の勝率、他者の心理の真実、ゲーム横断の自動評価は得ていない。',
        '競り資金の代理価格は max(0.25, 残商品価値合計 / 人数 / 残資金)、上限2。終局資金価値を下限に、将来の商品獲得に使える資金を評価する。価値の等分を仮定した未校正の機会費用で、次の山順や実相手方策を入力しない。',
        '性格を変えず、行動の予測を更新する。学習・忘却は既存の有限仮説と減衰窓。正しい学習を保証せず、誤読も結果に残す。',
        '', '## 解釈上の制限', '']+[f'- {x}' for x in result['limitations']]
    if result.get('capital_price_comparison'):
        lines+=['','## 資金評価の修正前後（同じ開発条件）','',
            '|mode|games per version|before win credit|after win credit|before mean margin|after mean margin|',
            '|---|---:|---:|---:|---:|---:|']
        for row in result['capital_price_comparison']:
            lines.append(f'|{row["mode"]}|{row["games"]}|{row["before_win_credit"]:.2f}|{row["after_win_credit"]:.2f}|{row["before_margin"]:.2f}|{row["after_margin"]:.2f}|')
        lines+=['','再現: `run(game="auction", ..., capital_pricing=False)` は終局資金価値0.25と取得量による旧達成評価。既定Trueは将来資金価値と純獲得による達成評価。2つを一緒に変えた結果で、独立の寄与は未分離。修正前576局はローカル保存、修正後576局が上表の対象。',
            'ハゲタカでは読み常用の優勝持分が大きく、利益ゲートは有益な変更も抑える。競りでは今回の開発条件で資金評価が改善したが、全方式とも平均点差は負。人間並みの競り判断に到達したとは扱わない。']
    lines+=['','Hagetaka rules: https://www.amigo-spiele.de/kartenspiele/hols-der-geier_1943_1210',
            'Zero-carry convention: https://mobius-games.co.jp/25th/rule.html','']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
