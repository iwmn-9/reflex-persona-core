"""Balanced real matches: reflex / bounded search / flat terminal Monte Carlo.

Separate actual match outcomes from model estimates. Opponent controllers and
the driver's real future deck are never supplied to an evaluation model.
"""
from collections import Counter
from dataclasses import asdict, replace
import copy
import json
from pathlib import Path
import platform
import time
import numpy as np
from .core import Policy, digest
from .laboratory import profiles
from .board_models import (ConnectPosition, ConnectAdapter, connect_features, connect_referee,
                           ThanksPosition, ThanksAdapter, thanks_referee_score, observe)
from .board_planning import REFLEX, SHORT, core_audit
from .cross_games import core_hashes
from .monte_carlo import RolloutBudget, wilson
from .rollout_boards import observe_rollouts, connect_tactical, thanks_tactical
from .teachers import prompt, write_json


def methods(samples=(8,32)):
    if not samples or len(set(samples))!=len(samples): raise ValueError('distinct sample counts required')
    result={'reflex':REFLEX,'search3':SHORT}
    for n in samples: result[f'mc{n}']=RolloutBudget(samples=n,min_samples=min(n,4),max_nodes=50000)
    return result


def minimax2(state,rng):
    side=state.turn
    def score(s):
        if s.winner() is not None: return 1. if s.winner()==side else -1.
        return 0. if sum(s.heights)==42 else connect_features(s,side)[0]
    values={}
    for a in state.legal():
        child=state.play(a)
        values[a]=min((score(child.play(b)) for b in child.legal()),default=score(child))
    best=max(values.values()); names=[a for a,v in values.items() if abs(v-best)<1e-12]
    return names[int(rng.integers(len(names)))]


def _candidate(adapter,state,profile,control,seed,tick,episode,memory):
    start=time.perf_counter()
    if isinstance(control,RolloutBudget): c,stats=observe_rollouts(adapter,state,profile,control,seed,tick,episode,memory)
    else: c,stats=observe(adapter,state,profile,control,seed,tick,episode,memory)
    d=Policy().choose(c)
    elapsed=(time.perf_counter()-start)*1000
    return c,d,stats,elapsed


def _snapshot(c,stats,method,sink,profile):
    if sink is None: return
    tag=[c['scope']['game'],method,profile]
    if any(r['tag']==tag for r in sink): return
    sink.append(dict(id=f'mc-comparison-{len(sink):03d}',tag=tag,context=copy.deepcopy(c),
                     context_hash=digest(c),prompt=prompt(c),quality='awaiting_generation',
                     evaluation_provenance=copy.deepcopy(stats['config']),
                     source='pre_decision_snapshot_no_gold'))


def _result(game,seed,players,seat,profile,method,opponent,control,winners,scores,trace,times,first_prediction):
    won=seat in winners; share=(1/len(winners) if won else 0.)
    if game=='connect_four' and not winners: share=.5
    return dict(game=game,seed=seed,players=players,candidate_seat=seat,profile=profile['id'],method=method,
                opponent=opponent,config=asdict(control),won=won,win_share=share,draw=not winners,
                scores=list(scores),own_score=scores[seat] if scores else None,winner_seats=winners,
                finished=True,turns=len(trace),trace=trace,decision_ms=times,
                first_prediction=first_prediction)


def play_connect(seed,seat,profile_index,method,control,opponent,teacher_sink=None):
    state=ConnectPosition(); setup_rng=np.random.default_rng(10000+seed*31)
    # New held-out seeds, six legal setup drops shared by all comparison methods.
    for _ in range(6): state=state.play(state.legal()[int(setup_rng.integers(len(state.legal())))])
    initial=state; adapter=ConnectAdapter(); profile=profiles()[profile_index]
    rng=np.random.default_rng(20000+seed*37+seat); trace=[]; times=[]; memory=None; first=None
    for tick in range(42):
        if adapter.terminal(state): break
        actor=state.turn; stats=None
        if actor==seat:
            c,d,stats,elapsed=_candidate(adapter,state,profile,control,seed,tick,f'mc-connect-{seed}',memory)
            name=d['action_id']; memory=d['next_state']; times.append(elapsed)
            _snapshot(c,stats,method,teacher_sink,profile['id'])
            if first is None and 'actions' in stats: first=stats['actions'][name]
        else: name=(minimax2 if opponent=='minimax2' else connect_tactical)(state,rng)
        before=state; winner,legal,grid=connect_referee(before)
        if winner!=before.winner() or legal!=before.legal() or name not in legal: raise AssertionError('connect independent referee mismatch')
        state=state.play(name); w,l,g=connect_referee(state); col=int(name.split(':')[1])
        changes=[(x,y) for y in range(6) for x in range(7) if grid[y][x]!=g[y][x]]
        if changes!=[(col,before.heights[col])] or w!=state.winner() or l!=state.legal(): raise AssertionError('connect transition mismatch')
        trace.append(dict(tick=tick,actor=actor,action=name,evaluation=stats,discs_before=before.discs,discs_after=state.discs))
    if not adapter.terminal(state): raise AssertionError('unfinished connect match')
    w=state.winner(); result=_result(adapter.game,seed,2,seat,profile,method,opponent,control,[] if w is None else [w],[],trace,times,first)
    result['start_discs']=initial.discs; result['start_heights']=initial.heights
    return result


def play_thanks(seed,players,seat,profile_index,method,control,opponent,teacher_sink=None):
    # Only this driver owns the shuffled deck / unseen nine removed cards.
    deck=np.random.default_rng(30000+seed*41).permutation(np.arange(3,36))[:24].tolist()
    state=ThanksPosition.start(players,deck[0],first=seed%players); index=1
    adapter=ThanksAdapter(sample_seed=seed+40000); profile=profiles()[profile_index]
    ledger=list(state.chips); supply=sum(ledger)
    rng=np.random.default_rng(50000+seed*43+seat); trace=[]; times=[]; memory=None; first=None
    for tick in range(2000):
        if adapter.terminal(state): break
        if adapter.chance(state): state=state.draw(deck[index]); index+=1
        actor=state.turn; stats=None
        if actor==seat:
            view=replace(state,chips=tuple(ledger))
            c,d,stats,elapsed=_candidate(adapter,view,profile,control,seed,tick,f'mc-thanks-{seed}-{players}',memory)
            name=d['action_id']; memory=d['next_state']; times.append(elapsed)
            _snapshot(c,stats,method,teacher_sink,profile['id'])
            if first is None and 'actions' in stats: first=stats['actions'][name]
        elif opponent=='random':
            names=state.legal(); name=names[int(rng.integers(len(names)))]
        else: name=thanks_tactical(state,rng)
        before=state; state=state.play(name)
        if name=='PASS': ledger[actor]-=1
        else: ledger[actor]+=before.pot
        if state.chips!=tuple(ledger) or sum(state.chips)+state.pot!=supply or min(state.chips)<0:
            raise AssertionError('public ledger/counter conservation mismatch')
        if state.scores()!=tuple(thanks_referee_score(cards,chips) for cards,chips in zip(state.cards,state.chips)):
            raise AssertionError('independent card scoring mismatch')
        taken=sum(map(len,state.cards))
        if taken+state.remaining+int(state.card is not None)!=24 or len(state.seen)!=taken+int(state.card is not None):
            raise AssertionError('reveal/card conservation mismatch')
        if name=='PASS' and (state.turn!=(actor+1)%players or state.pot!=before.pot+1 or state.card!=before.card):
            raise AssertionError('bad refusal transition')
        if name=='TAKE' and (state.turn!=actor or before.card not in state.cards[actor]): raise AssertionError('bad take transition')
        trace.append(dict(tick=tick,actor=actor,action=name,card=before.card,pot_before=before.pot,
                          chips_before=before.chips,chips_after=state.chips,scores_after=state.scores(),evaluation=stats))
    if not adapter.terminal(state): raise AssertionError('unfinished No Thanks match, no terminal reward assigned')
    scores=state.scores(); winners=[i for i,s in enumerate(scores) if s==min(scores)]
    return _result(adapter.game,seed,players,seat,profile,method,opponent,control,winners,scores,trace,times,first)


def save_results(root,runs,teachers,samples,seeds,elapsed_seconds):
    root=Path(root); root.mkdir(parents=True,exist_ok=True); groups=[]
    keys=sorted({(r['game'],r['players'],r['opponent'],r['method']) for r in runs})
    for game,n,opponent,method in keys:
        rs=[r for r in runs if (r['game'],r['players'],r['opponent'],r['method'])==(game,n,opponent,method)]
        times=[t for r in rs for t in r['decision_ms']]
        evaluated=[t['evaluation'] for r in rs for t in r['trace'] if t['evaluation'] is not None]
        mc=[s for s in evaluated if 'completed_samples' in s]
        first=[r for r in rs if r['first_prediction'] is not None and r['first_prediction']['win_rate'] is not None]
        groups.append(dict(game=game,players=n,opponent=opponent,method=method,games=len(rs),
                           wins=sum(r['won'] for r in rs),win_rate=sum(r['won'] for r in rs)/len(rs),
                           binomial_interval95_reference=wilson(sum(r['won'] for r in rs),len(rs)),
                           win_share=float(np.mean([r['win_share'] for r in rs])),draws=sum(r['draw'] for r in rs),
                           mean_score=float(np.mean([r['own_score'] for r in rs])) if game=='no_thanks_basic' else None,
                           p50_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),
                           additional_nodes=sum(s['additional_nodes'] for s in evaluated),
                           completed_sample_counts=dict(Counter(str(s['completed_samples']) for s in mc)),
                           fallback_decisions=sum(not s['used'] for s in mc),
                           exhausted_decisions=sum(s.get('budget_exhausted',False) or s.get('length_exhausted',False) for s in mc),
                           first_decision_brier=float(np.mean([(r['first_prediction']['win_rate']-float(r['won']))**2 for r in first])) if first else None))
    aggregate=[]
    for game,method in sorted({(r['game'],r['method']) for r in runs}):
        rs=[r for r in runs if r['game']==game and r['method']==method]
        times=[t for r in rs for t in r['decision_ms']]
        aggregate.append(dict(game=game,method=method,games=len(rs),wins=sum(r['won'] for r in rs),
                              mean_score=float(np.mean([r['own_score'] for r in rs])) if game=='no_thanks_basic' else None,
                              p50_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95))))
    audit=core_audit(); result=dict(core_audit=audit,core_hashes=core_hashes(),seeds=seeds,samples=list(samples),
        games=len(runs),finished=sum(r['finished'] for r in runs),referee_mismatches=0,groups=groups,aggregate=aggregate,
        teacher_requests=len(teachers),accepted_llm_rows=0,elapsed_seconds=elapsed_seconds,
        environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform()),
        protocol=dict(connect_setup='6 legal random drops, seeds 10000+31*seed; new relative to prior probes',
                      connect_profiles=['growth','ego'],connect_seats=[0,1],connect_opponents=['tactical','minimax2'],
                      thanks_profiles='growth/ego alternate by seed',thanks_seats='0 and floor(players/2), not all seat permutations',
                      thanks_opponents=['tactical','random'],rollout_policy='tactical for both own and assumed other moves',
                      randomness='same root-independent trial streams; model chance stream separate from real deck',
                      brier='one first evaluated decision per game, descriptive policy/model mismatch diagnostic; not fitted calibration',
                      intervals='match groups reuse seed/board across profiles/seats; binomial intervals are a descriptive reference, not independent-sample confidence for general strength',
                      timing='adapter + rollout/search + validated Policy; excludes driver/referee/IO; exploratory desktop measurement'),
        limits=['small paired sample, not a proof of improved strength or cross-game learned transfer',
                'future rollout policy may differ from actual future own actions and actual opponent',
                'sampling intervals describe finite draws, not accuracy of the assumed behavioral model',
                'terminal objective comes from true simulated end conditions; value/need/style axes remain adapter-designed',
                '8-outcome compression approximates risk; completed-round budget stopping can affect sample distribution',
                'no MCTS, policy/value training, GPU runtime or automatic teacher approval'])
    write_json(root/'evaluation.json',result); write_json(root/'trajectories.json',runs)
    (root/'teacher_requests.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in teachers),encoding='utf-8')
    lines=['# 終局モンテカルロ評価の比較', '', '## 得たもの', '',
        '- 各根の全合法候補に同数の終局標本を与える任意の評価器。人格コア/永続ランタイムは変更しない。',
        '- 勝利率、同点の按分、期待終局報酬、No Thanksの生の得点、標本誤差と打切りを分けて保存。',
        '- 標本数/追加遷移上限/1試行の遷移上限/最低完了標本数/仮想行動方策を設定可能。',
        '- 途中終了を敗北と数えない。全候補の標本がそろったラウンドだけ採用。最低数未満なら単手評価へ戻す。',
        '- 実山札・実相手コントローラ・隠れた人格を予測へ渡さない。No Thanksは公開集合から未知カードを逐次標本化する。',
        '', f"実対戦 {result['finished']}/{result['games']}局完了、独立審判の不一致0。",
        '', '|ゲーム/人数|比較相手|方法|勝利/局数|平均勝利配分|平均点（小さい方が良い）|判断p50/p95 ms|',
        '|---|---|---|---:|---:|---:|---:|']
    for g in groups:
        score='—' if g['mean_score'] is None else f"{g['mean_score']:.2f}"
        lines.append(f"|{g['game']}/{g['players']}|{g['opponent']}|{g['method']}|{g['wins']}/{g['games']}|{g['win_share']:.3f}|{score}|{g['p50_ms']:.2f}/{g['p95_ms']:.2f}|")
    lines+=['', '勝利数は同点勝者も1勝、平均勝利配分は同点人数で按分。Connect Fourの引分は勝利数0・配分0.5。同じseedの盤面を人格/座席間で再利用するため、対局は独立標本ではない。evaluation.jsonの二項95%区間は記述用の目安で、強さの母集団に対する信頼区間として使わない。数局の差を優劣の確定と扱わない。',
        '', '実対戦の条件混合集計（人数/相手別の内訳は上表）:',
        '', '|ゲーム|方法|勝利/局数|平均点|判断p50/p95 ms|', '|---|---|---:|---:|---:|']
    for g in aggregate:
        score='—' if g['mean_score'] is None else f"{g['mean_score']:.2f}"
        lines.append(f"|{g['game']}|{g['method']}|{g['wins']}/{g['games']}|{score}|{g['p50_ms']:.2f}/{g['p95_ms']:.2f}|")
    lines+=['', '標本数を増やしても実勝利数が増える保証はない。No Thanksの終局objectiveは勝利配分であり、生得点の最小化を直接最適化していないことも含めて解釈する。MCの常時適用や標本増量を自動採用しない。',
        '', '## 削ったもの・費用', '',
        '- 全終局分布は有限標本へ近似。既定のtactical継続は本人の未来の人格判断も近似する。persona設定なら本人のPolicy、他者の中立Policyを使うが重くなる。',
        '- tacticalはConnect Fourの即勝ち/即負け回避＋その他ランダム、No Thanksの即時自得点改善。強い相手・深い戦術を再現する方策ではない。',
        '- 8結果を超える数値分布を圧縮するため、期待値は保つが人格ごとの損失分布を近似する。',
        '- 最大遷移数には破棄した試行も含む。初回根の生成、選択/評価処理、JSON検査、時間そのものの上限は別。',
        '', '## 方向性への影響', '',
        '比較用の標準的なflat Monte Carloを導入した。MCTSの木の成長や価値ネットワークは追加していない。終局目的の見通しと本人による価値づけを分離し、勝率最優先の人格へ書き換えない。通常の反射へ自動適用しない。',
        '', '標本内の勝利率は仮定した継続方策に対する条件付き推定。実対戦の勝率は別列。最初の評価と対戦結果のBrier値も記録するが、実対戦では本人/相手の将来方策が変わるため純粋な評価器の校正試験ではない。',
        '', f'相談用の判断前材料{len(teachers)}件。教師回答の生成・採用・学習は未実施。実行はローカルCPU、Colab起動なし。',
        '', '既存手法の一次資料: https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/mcts.py （RandomRolloutEvaluator）。外部ライブラリ依存を増やさず、標準の終局平均の方式を実装。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def experiment(root,seeds=2,samples=(8,32),progress=None):
    if type(seeds) is not int or not 1<=seeds<=8: raise ValueError('seed count 1..8 required')
    controls=methods(samples); runs=[]; teachers=[]; hashes=core_hashes(); core_audit(); start=time.perf_counter()
    for seed in range(seeds):
        for opponent in ('tactical','minimax2'):
            for profile in (0,3):
                for seat in (0,1):
                    for name,control in controls.items():
                        runs.append(play_connect(seed,seat,profile,name,control,opponent,teachers))
                        if progress: progress(dict(game='connect_four',completed=len(runs),seed=seed,method=name,opponent=opponent))
        for n in (3,5,7):
            for opponent in ('tactical','random'):
                for seat in (0,n//2):
                    for name,control in controls.items():
                        runs.append(play_thanks(seed,n,seat,0 if seed%2==0 else 3,name,control,opponent,teachers))
                        if progress: progress(dict(game='no_thanks_basic',players=n,completed=len(runs),seed=seed,method=name,opponent=opponent))
    if core_hashes()!=hashes: raise AssertionError('shared reflex implementation changed')
    return save_results(root,runs,teachers,samples,seeds,time.perf_counter()-start)
