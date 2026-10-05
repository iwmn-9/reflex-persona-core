"""Experiments for controlled lookahead across two different tabletop games."""
from collections import Counter
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
import time
from zipfile import ZipFile
import numpy as np
from .core import Policy, compile_batch, digest
from .cross_games import core_hashes
from .laboratory import profiles
from .planning import PlanningBudget
from .board_models import (ConnectPosition, ConnectAdapter, connect_referee, ThanksPosition,
                           ThanksAdapter, thanks_referee_score, observe, CONNECT_RULES, THANKS_RULES)
from .teachers import prompt, write_json

REFLEX=PlanningBudget(enabled=False,depth=1,max_nodes=0)
SHORT=PlanningBudget(depth=3,max_nodes=128,width=3,chance_samples=2)
LONG=PlanningBudget(depth=4,max_nodes=512,width=5,chance_samples=3)
CONNECT_FULL=PlanningBudget(depth=4,max_nodes=4096,width=7,chance_samples=1)


def budgets_for(budget,players):
    values=[budget]*players if isinstance(budget,PlanningBudget) else list(budget)
    if len(values)!=players or not all(isinstance(b,PlanningBudget) for b in values): raise ValueError('one planning budget per player required')
    return values


def budget_record(budget):
    return asdict(budget) if isinstance(budget,PlanningBudget) else dict(per_player=[asdict(b) for b in budget])


def preserve_teacher(c,sink,stats):
    if sink is None or len(sink)>=24: return
    tag=(c['scope']['game'],stats['reached_depth'],c['scope']['npc'],c['state']['mode'])
    if sum(r['tag'][:2]==list(tag[:2]) for r in sink)>=3: return
    if any(r['tag']==list(tag) for r in sink): return
    sink.append(dict(id=f'planning-{len(sink):03d}',tag=list(tag),context=copy.deepcopy(c),
                     context_hash=digest(c),prompt=prompt(c),quality='awaiting_generation',
                     planning=dict(config=stats['config'],reached_depth=stats['reached_depth']),
                     source='pre_decision_snapshot_no_gold'))


def run_connect(seed,budget,order=(0,3),teacher_sink=None):
    adapter=ConnectAdapter(); state=ConnectPosition(); roster=[profiles()[i] for i in order]
    controls=budgets_for(budget,2)
    memories=[None,None]; policy=Policy(); trace=[]; durations=[]
    for tick in range(42):
        if adapter.terminal(state): break
        actor=state.turn; start=time.perf_counter()
        c,stats=observe(adapter,state,roster[actor],controls[actor],seed,tick,f'connect-{seed}',memories[actor])
        b=compile_batch([c]); decision=policy.decide(b).records(b)[0]
        durations.append((time.perf_counter()-start)*1000)
        preserve_teacher(c,teacher_sink,stats); name=decision['action_id']; after=state.play(name)
        winner,legal,grid=connect_referee(state)
        if tuple(legal)!=state.legal() or winner!=state.winner(): raise AssertionError('connect referee mismatch')
        w2,legal2,g2=connect_referee(after)
        col=int(name.split(':')[1]); changed=[(x,y) for y in range(6) for x in range(7) if grid[y][x]!=g2[y][x]]
        if changed!=[(col,state.heights[col])] or w2!=after.winner() or legal2!=after.legal(): raise AssertionError('bad gravity transition')
        memories[actor]=decision['next_state']
        trace.append(dict(tick=tick,actor=actor,profile=roster[actor]['id'],action=name,planning=stats,
                          mode=decision['next_state']['mode'],before=list(state.discs),after=list(after.discs)))
        state=after
    if not adapter.terminal(state): raise AssertionError('42 drops must end the game')
    winner=state.winner()
    return dict(game=adapter.game,seed=seed,players=2,roster=[p['id'] for p in roster],budget=budget_record(budget),
                winner_seats=[] if winner is None else [winner],winner_profiles=[] if winner is None else [roster[winner]['id']],
                scores=[],turns=len(trace),trace=trace,decision_ms=durations,truncated=False)


def run_thanks(seed,players,budget,offset=0,teacher_sink=None):
    # Real shuffled/removed cards belong only to this simulation driver.
    deck=np.random.default_rng(seed+1729).permutation(np.arange(3,36))[:24].tolist()
    state=ThanksPosition.start(players,deck[0],first=seed%players); index=1
    controls=budgets_for(budget,players)
    adapter=ThanksAdapter(sample_seed=seed+991); roster=[profiles()[(i+offset)%4] for i in range(players)]
    memories=[None]*players; trace=[]; times=[]; policy=Policy(); initial_supply=sum(state.chips)
    ledger=[11 if players<=5 else 9 if players==6 else 7]*players
    for tick in range(2000):
        if adapter.terminal(state): break
        if adapter.chance(state):
            state=state.draw(deck[index]); index+=1
        actor=state.turn; old_scores=state.scores(); start=time.perf_counter()
        # Observer stock is reconstructed from public transactions, not private stash.
        view=replace(state,chips=tuple(ledger))
        c,stats=observe(adapter,view,roster[actor],controls[actor],seed,tick,f'thanks-{seed}-{players}',memories[actor])
        b=compile_batch([c]); d=policy.decide(b).records(b)[0]; times.append((time.perf_counter()-start)*1000)
        preserve_teacher(c,teacher_sink,stats); after=state.play(d['action_id'])
        if d['action_id']=='PASS': ledger[actor]-=1
        else: ledger[actor]+=state.pot
        if tuple(ledger)!=after.chips: raise AssertionError('public transaction ledger mismatch')
        if sum(after.chips)+after.pot!=initial_supply or min(after.chips)<0: raise AssertionError('counter conservation mismatch')
        if after.scores()!=tuple(thanks_referee_score(cards,chips) for cards,chips in zip(after.cards,after.chips)):
            raise AssertionError('independent chain scoring mismatch')
        taken=sum(map(len,after.cards)); revealed=len(after.seen)
        if revealed!=taken+int(after.card is not None) or taken+after.remaining+int(after.card is not None)!=24:
            raise AssertionError('card/reveal conservation mismatch')
        if d['action_id']=='PASS':
            if after.card!=state.card or after.turn!=(actor+1)%players or after.pot!=state.pot+1: raise AssertionError('bad refusal')
        elif after.turn!=actor or after.chips[actor]!=state.chips[actor]+state.pot or state.card not in after.cards[actor]:
            raise AssertionError('taker must retain turn and receive pot')
        memories[actor]=d['next_state']
        trace.append(dict(tick=tick,actor=actor,profile=roster[actor]['id'],action=d['action_id'],planning=stats,
                          mode=d['next_state']['mode'],card=state.card,pot_before=state.pot,chips_before=state.chips,
                          chips_after=after.chips,scores_before=old_scores,scores_after=after.scores(),target=d['target']))
        state=after
    done=adapter.terminal(state); scores=state.scores()
    winners=[i for i,s in enumerate(scores) if s==min(scores)] if done else []
    return dict(game=adapter.game,seed=seed,players=players,roster=[p['id'] for p in roster],budget=budget_record(budget),
                winner_seats=winners,winner_profiles=[roster[i]['id'] for i in winners],scores=scores,
                turns=len(trace),trace=trace,decision_ms=times,truncated=not done)


def tactical_probe():
    # Search-independent fixed public games: forks can require more than a reply.
    positions=[]
    for seed in range(10):
        rng=np.random.default_rng(seed+600); p=ConnectPosition()
        for tick in range(14):
            if not p.legal(): break
            if tick in (8,10,12): positions.append((seed,tick,p))
            p=p.play(p.legal()[int(rng.integers(len(p.legal())))])
    adapter=ConnectAdapter(); rows=[]; profile=profiles()[0]
    for seed,tick,p in positions:
        choices={}
        for name,budget in (('reflex',REFLEX),('short',SHORT),('long',LONG),('full4',CONNECT_FULL)):
            c,stats=observe(adapter,p,profile,budget,seed,tick,f'probe-{seed}')
            d=Policy().choose(c,False)
            choices[name]=dict(action=d['action_id'],reached_depth=stats['reached_depth'],nodes=stats['additional_nodes'])
        # Exhaustive 4-ply, objective-focused game reference; not a solved engine.
        from .board_models import connect_features
        side=p.turn
        def reference(s,depth):
            if not depth or adapter.terminal(s): return connect_features(s,side)[0]
            values=[reference(s.play(a),depth-1) for a in s.legal()]
            return max(values) if s.turn==side else min(values)
        reference_scores={a:reference(p.play(a),3) for a in p.legal()}
        optimum=max(reference_scores.values())
        rows.append(dict(seed=seed,tick=tick,discs=p.discs,heights=p.heights,turn=p.turn,choices=choices,
                         reference_scores=reference_scores,regret={k:float(optimum-reference_scores[v['action']]) for k,v in choices.items()}))
    return rows


def core_audit():
    backup=Path(__file__).resolve().parents[1]/'backups/foundation-before-controlled-planning.zip'
    if not backup.exists(): return dict(unchanged_verified=None,reason='historical backup not bundled')
    with ZipFile(backup) as z:
        for name in ('core.py','runtime.py'):
            if z.read('reflex/'+name)!=Path(__file__).with_name(name).read_bytes(): raise AssertionError('shared reflex implementation changed')
    return dict(unchanged_verified=True)


def experiment(root,seeds=2):
    if type(seeds) is not int or not 1<=seeds<=8: raise ValueError('seed count 1..8 required')
    root=Path(root); root.mkdir(parents=True,exist_ok=True); audit=core_audit(); hashes=core_hashes()
    runs=[]; teachers=[]; start=time.perf_counter()
    for seed in range(seeds):
        for budget in (REFLEX,SHORT,LONG):
            for order in ((0,3),(3,0)):
                runs.append(run_connect(seed,budget,order,teachers))
            for n in (3,5,7): runs.append(run_thanks(seed,n,budget,offset=seed,teacher_sink=teachers))
        for order in ((0,3),(3,0)): runs.append(run_connect(seed,CONNECT_FULL,order,teachers))
    runs.append(run_connect(0,(REFLEX,CONNECT_FULL)))
    runs.append(run_thanks(0,5,(REFLEX,SHORT,LONG,SHORT,REFLEX)))
    if hashes!=core_hashes(): raise AssertionError('shared core changed during evaluation')
    return save_results(root,runs,teachers,seeds,elapsed_seconds=time.perf_counter()-start)


def save_results(root,runs,teachers,seeds,elapsed_seconds=0,probes=None,provenance=None):
    """Canonical aggregation permits rechecking one changed adapter without rerunning the others."""
    root=Path(root); root.mkdir(parents=True,exist_ok=True); start=time.perf_counter()
    audit=core_audit(); hashes=core_hashes()
    probes=tactical_probe() if probes is None else probes
    groups=[]
    for game in ('connect_four','no_thanks_basic'):
        for budget in ((REFLEX,SHORT,LONG,CONNECT_FULL) if game=='connect_four' else (REFLEX,SHORT,LONG)):
            group=[r for r in runs if r['game']==game and r['budget']==asdict(budget)]
            rows=[t for r in group for t in r['trace']]; times=[d for r in group for d in r['decision_ms']]
            groups.append(dict(game=game,budget=asdict(budget),games=len(group),finished=sum(not r['truncated'] for r in group),
                               players=sorted(set(r['players'] for r in group)),decisions=len(rows),
                               reached_depths=dict(Counter(str(t['planning']['reached_depth']) for t in rows)),
                               exhausted=sum(t['planning']['budget_exhausted'] for t in rows),
                               max_additional_nodes=max(t['planning']['additional_nodes'] for t in rows),
                               p50_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),
                               wins_by_profile=dict(Counter(p for r in group for p in r['winner_profiles']))))
    if hashes!=core_hashes(): raise AssertionError('shared core changed during evaluation')
    probe_summary={name:dict(mean_regret=float(np.mean([r['regret'][name] for r in probes])),
                             optimal_choices=sum(r['regret'][name]<1e-9 for r in probes),cases=len(probes))
                   for name in ('reflex','short','long','full4')}
    mixed=[dict(game=r['game'],players=r['players'],controls=r['budget'],roster=r['roster'],
                finished=not r['truncated'],turns=r['turns'],winner_profiles=r['winner_profiles']) for r in runs if 'per_player' in r['budget']]
    result=dict(core_audit=audit,core_hashes=hashes,seeds=seeds,groups=groups,probe_summary=probe_summary,mixed_controls=mixed,
                tactical_probe=probes,referee_mismatches=0,truncated_games=sum(r['truncated'] for r in runs),
                teacher_requests=len(teachers),accepted_llm_rows=0,elapsed_seconds=elapsed_seconds+time.perf_counter()-start,
                execution_provenance=provenance or 'single experiment execution',
                environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform()),
                limits=['finite beam search can miss a good future move; greater budget is not a guaranteed intelligence increase',
                        'opponents modeled with neutral priorities; no access to real hidden personality',
                        'No Thanks hypothetical draws use a small public-information sample, not the real future deck',
                        'outcome compression preserves expectation, not full personality-conditioned risk distribution',
                        'node budget excludes already required reflex root construction and game callback work; not a millisecond deadline',
                        'game adapters supply meaning/evaluation; no learned universal game understanding'])
    write_json(root/'evaluation.json',result); write_json(root/'trajectories.json',runs)
    (root/'teacher_requests.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False,allow_nan=False)+'\n' for c in teachers),encoding='utf-8')
    lines=['# 先読みを制御する土台と異なる卓上ゲーム', '', '## 得たもの', '',
           '- 人格コアを変更せず、任意のゲーム状態/遷移/観測/評価/相手仮説へ接続する先読みを追加。通常の判断と別モジュール。',
           '- enabled、depth、max_nodes、width、chance_samplesを独立設定。賢さという単一の人格値は追加しない。ゲーム側が使用条件と評価基準を決める。',
           '- 同一対戦内で個体別の探索予算も設定し、性格と計算能力を別にした混在条件を実行する。',
           '- 「他者が得をするカードを奪う効果」は実際に得をする場合だけ計上。単なる高コストの引受けを妨害の価値へ二重計上しない。',
           '- 全ての根の合法候補を保持。深いところの幅制限は明示。予算切れでは全候補について完了した深さへ戻し、一部の候補だけ深く評価する偏りを避ける。',
           '- 中立の相手仮説で予測し、自分の将来の選択には同じ本人の人格を使う。実対戦の相手の隠れた性格は渡さない。',
           f'- コネクトフォー（標準7×6・2人）: {CONNECT_RULES}',
           f'- No Thanks! 基本ゲーム（3/5/7人）: {THANKS_RULES}',
           '', '|ゲーム|深さ/追加ノード上限/幅|完了/局数|実際に完了した深さの判断数|予算切れ|境界込みp50/p95 ms|', '|---|---|---:|---|---:|---:|']
    for g in groups:
        b=g['budget']; lines.append(f"|{g['game']}|{b['depth']}/{b['max_nodes']}/{b['width']}|{g['finished']}/{g['games']}|{g['reached_depths']}|{g['exhausted']}|{g['p50_ms']:.2f}/{g['p95_ms']:.2f}|")
    lines+=['',f"審判の不一致0、打切り{result['truncated_games']}局。コア/永続ランタイムのハッシュは変更なし。", '',
            f"同一盤面{len(probes)}件の自作4 ply全幅参照に対する比較: {probe_summary}",
            f"異なる計算予算を持つ個体の混在: {sum(r['finished'] for r in mixed)}/{len(mixed)}局完了。", '',
            '幅を絞った設定はこの参照比較で単手より悪い例があった。4手/追加4096ノード/幅7のConnect Four設定も検証し、将来の全合法応答を含めた場合の改善と計算費用を記録する。設定は試作の診断に使ったもので、独立した未見評価ではない。', '',
            '## 削ったもの・後回し', '',
            '- 深い探索の全分岐は幅制限で削る。No Thanksの全未観測カード分布も少数標本に近似する。どちらも計算量を抑える代わりに見落としや予測誤差が残る。',
            '- 8結果を超える分布は最悪の目的結果と7群へ圧縮。質量/期待値は保つが、群内の損失と人格ごとのリスク分布を近似する。',
            '- 今回は学習による相手推定、同盟交渉、LLMへの人格機構の組込みを追加していない。旧コリドールは単手の比較として残す。',
            '', '## 方向性への影響', '',
            'depthには今の自分の1手を含み、1は通常の単手判断。追加ノード上限は根の単手候補を構築した後の遷移/確率分岐を数える。ゲーム側の評価処理や時間そのものの上限ではない。多人数の各仮説の手番は自分の都合でなくその観測者の評価で選ぶ。相手は中立の仮定で、正確に相手を読めたという意味ではない。',
            '', '「多く読む＝必ず賢い」とは扱わない。tactical_probeは同一盤面の各設定を、自作の4 ply全幅ミニマックス評価と比較する。これは完全解でも人間の知能測定でもない。数値/手順の正本はevaluation.json/trajectories.json。',
            '',f'判断前の教師相談材料{len(teachers)}件。生成/採用/学習はまだ行っていない。ローカルCPUで実行し、Colabは起動していない。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
