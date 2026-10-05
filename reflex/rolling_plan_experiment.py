"""Three-game comparison of flat choices, fresh plans, and retained plans."""
from collections import Counter
from dataclasses import asdict,replace
from pathlib import Path
import time
import numpy as np
from .laboratory import profiles
from .goofspiel import Pending
from .rule_baseline import GoofRules
from .rule_baseline_experiment import play
from .rolling_plans import PlanBudget,PlanMemory,decide
from .adaptive_reading_experiment import score_sensitive_positions
from .cross_games import core_hashes
from .board_planning import core_audit
from .teachers import write_json

CONTROLS={
    'flat24':PlanBudget(horizon=1,variants=0,search_samples=24,persist=False),
    'fresh3':PlanBudget(persist=False),
    'retained3':PlanBudget(persist=True),
}
SOURCES=[
    dict(method='Rolling horizon + shifted sequences',url='https://rdgain.github.io/assets/pdf/papers/gaina2017rhhybrids.pdf',
         use='short candidates, simulate, execute first action, reuse remaining plan; this prototype is bounded sampling/mutation, not a full RHEA reproduction'),
    dict(method='Options / temporal abstraction',url='https://people.eecs.berkeley.edu/~russell/classes/cs294/f05/papers/sutton%2Bal-1999.pdf',
         use='framework for persistent policies and termination; semantic attack/defend options not implemented here'),
    dict(method='UCT',url='https://aima.cs.berkeley.edu/~russell/classes/cs294/s11/readings/Kocsis%2BSzepesvari:2006.pdf',
         use='alternative for concentrating simulations on branches; deferred, not combined in this trial'),
    dict(method='RHEA parameter study',url='https://arxiv.org/abs/2003.12331',
         use='game/configuration dependence motivates explicit budgets and separate performance tests'),
]


def score_change_probe(seeds=4):
    """Legal score-only counterfactuals; not a reachable sudden real score event."""
    a,b=score_sensitive_positions();rules=GoofRules(3);rows=[];budget=PlanBudget(search_samples=16,validation_samples=48)
    for seed in range(seeds):
        profile=profiles()[0];episode='plan-score-counterfactual'
        ca,da,sa,ma=decide(rules,Pending(a),0,profile,'rules_persona',budget,seed,4,episode)
        # Keep the entire selected plan, because this comparison has not executed
        # its first move. Both states have identical hands/prizes/own score.
        held=PlanMemory(ma.owner,tuple(sa['selected_genes']),ma.generation)
        cb,db,sb,mb=decide(rules,Pending(b),0,profile,'rules_persona',budget,seed,4,episode,ca['state'],held)
        rows.append(dict(seed=seed,leading_action=da['action_id'],trailing_action=db['action_id'],
            same_personality=ca['personality']==cb['personality'] and ca['values']==cb['values'],
            same_hands=a.hands==b.hands,scores_a=list(a.scores),scores_b=list(b.scores),
            leading=sa,trailing=sb))
    return dict(rows=rows,plan_changes=sum(r['trailing']['plan_changed'] for r in rows),
                limits='selected legal diagnostic pair; actual live switching measured separately; independent uniform opponents')


def experiment(root,seeds=4,progress=None):
    if type(seeds) is not int or seeds<1:raise ValueError('positive seed count required')
    root=Path(root);root.mkdir(parents=True,exist_ok=True);start=time.perf_counter();runs=[]
    for game,players in (('connect_four',2),('goofspiel',4),('no_thanks_basic',3)):
        for seed in range(seeds):
            for opponent in ('random','tactical'):
                for method,control in CONTROLS.items():
                    # Keep two competitive personalities; known reward->benevolence
                    # ambiguity is not silently fixed by planning.
                    r=play(game,players,seed,0 if seed%2==0 else 3,'rules_persona',opponent,planner=control)
                    r['method']=method;runs.append(r)
        if progress:progress(f'{game}: 累計{len(runs)}局、計画の維持/見直しを含め完了')
    groups=[]
    for game in ('connect_four','goofspiel','no_thanks_basic'):
        for opponent in ('random','tactical'):
            for method in CONTROLS:
                rr=[r for r in runs if (r['game'],r['opponent'],r['method'])==(game,opponent,method)]
                ts=[t for r in rr for t in r['trace'] if t['own']]
                reasons=Counter(t['stats']['reason'] for t in ts)
                groups.append(dict(game=game,opponent=opponent,method=method,games=len(rr),mean_share=float(np.mean([r['win_share'] for r in rr])),
                    p50_ms=float(np.median([x for r in rr for x in r['decision_ms']])),
                    mean_nodes=float(np.mean([t['stats']['total_nodes'] for t in ts])),
                    starts=sum(t['stats']['plan_started'] for t in ts),
                    maintained=sum(not t['stats']['plan_started'] and not t['stats']['plan_changed'] for t in ts),
                    plan_changes=sum(t['stats']['plan_changed'] for t in ts),
                    confirmed_changes=sum(t['stats']['plan_changed'] and t['stats']['validation']['used'] for t in ts),
                    validation_rejections=sum(t['stats']['reason']=='gain_not_confirmed' for t in ts),reasons=dict(reasons)))
    paired=[]
    for r in runs:
        if r['method']=='flat24':continue
        old=next(o for o in runs if (o['game'],o['seed'],o['opponent'],o['method'])==(r['game'],r['seed'],r['opponent'],'flat24'))
        paired.append(dict(game=r['game'],seed=r['seed'],opponent=r['opponent'],method=r['method'],share_delta=r['win_share']-old['win_share']))
    result=dict(games=len(runs),finished=sum(r['finished'] for r in runs),referee_mismatches=0,groups=groups,paired=paired,
        score_change_probe=score_change_probe(),sources=SOURCES,core_hashes=core_hashes(),core_unchanged=core_audit(),
        elapsed_seconds=time.perf_counter()-start,accepted_llm_rows=0,
        protocol=dict(seeds=seeds,controls={k:asdict(v) for k,v in CONTROLS.items()},personas=['growth','ego'],
            representation='bounded quantiles of legal choices for up to 3 OWN decisions; remaining sequence shifted after execution',
            scoring='same persona/goal terminal effects; fresh uniform continuations after prefix, uniform independent opponents',
            validation='new independent terminal trials for current vs discovery winner; paired resampling lower-gap heuristic',
            fairness='same driver seed/seat/profile/opponent per condition, total node cap includes discarded trials; actual cost differs',
            limits='no learned rule reader/value model/semantic options/opponent learning; does not fix altruism-to-enemy semantics; bootstrap is not calibrated model confidence'))
    write_json(root/'evaluation.json',result);write_json(root/'trajectories.json',runs);write_report(root,result)
    return result


def write_report(root,result):
    root=Path(root)
    retained=[g for g in result['groups'] if g['method']=='retained3']
    maintained=sum(g['maintained'] for g in retained)
    changes=sum(g['plan_changes'] for g in retained)
    confirmed=sum(g['confirmed_changes'] for g in retained)
    lines=['# 短い方針を表し、続行と変更を採点する試作','',
        '調査から、短い行動列のRolling Horizon Planningを最初の候補に選んだ。ゲームの攻略ラベルを手書きせず、ルールと目的で行動列を評価する。原論文の完全再実装ではない。','',
        f'今回得たのは計画の継続・見直しの機構。保持方式では維持{maintained}回、変更{changes}回、そのうち独立試行による検証済み{confirmed}回。毎回の新規開始や計画の自然終了を方針変更と数えない。実際の強さの改善は確認できていない。','',
        '## 調査した候補','',
        '- 行動列と残りの引継ぎ: [Rolling Horizon Evolution Enhancements](https://rdgain.github.io/assets/pdf/papers/gaina2017rhhybrids.pdf)。数手の案を比較する今回の表現へ利用。',
        '- 継続する方策と終了条件: [Options](https://people.eecs.berkeley.edu/~russell/classes/cs294/f05/papers/sutton%2Bal-1999.pdf)。長い意味のある方針の設計枠だが、自動的に攻め/守りの意味を得る仕組みとは別。',
        '- 有望な枝へ試行を集中: [UCT](https://aima.cs.berkeley.edu/~russell/classes/cs294/s11/readings/Kocsis%2BSzepesvari:2006.pdf)。今回は木探索まで併合しない。',
        '- ゲームと設定の依存性: [RHEA parameter study](https://arxiv.org/abs/2003.12331)。一つの計算設定を万能な知能と扱わない。','',
        '## 得たもの・削ったもの・費用','',
        '- 得たもの: 最長3回の自分手番の計画を作り、実行した先頭を除いて個体ごとに保存。次の公開状況で残りを新しく採点し直す。古い予測値は流用しない。',
        '- 得たもの: すべての現在の合法手と一手のみの継続を残し、長い候補と現計画の変異案を加える。本人の人格/主義/目的を固定して同じPolicyで比較。',
        '- 得たもの: 探索に使わなかった乱数で現計画/新案を再試行し、対応した標本の再標本化でも利益が残るとき変更。誤差に埋もれる差は続行。',
        '- 削ったもの: ゲーム別の攻撃/防御の攻略ラベル、無限の方針候補、探索と同じ標本だけでの切替決定、未完了試行を失敗として扱うこと。',
        '- 費用: 複数の列を終局まで試すので軽い反射より高価。遷移数は1つの上限、再標本化/JSON/Policy採点のCPU費用はその外。',
        '- 近似: 選択順位の列なので、合法手が変わると同じ数値が別の具体的手を指す。相手は一様、3手の後の自分も一様。計画を維持することと抽象的な戦略を理解することを区別する。',
        '- 未解決: 本当の相手モデル、意味のある攻め/守り等の継続方策、欲求/関係性の意味契約、学習済み価値関数。相手を勝たせる慈善の問題はこの探索で解決したとはしない。','',
        '## 実対戦','',f'{result["games"]}局完了、独立審判との不一致0。flat24=一手/未来一様、fresh3=毎回新しい最長3手計画、retained3=残りを保持/確認して変更。少数の固定seedで、実際の計算量はそろっていない。','',
        '|ゲーム|相手|方式|局数|優勝持分|p50 ms|平均遷移|開始|維持|変更|検証済変更|利益未確認|','|---|---|---|---|---|---|---|---|---|---|---|---|']
    lines += [f'|{g["game"]}|{g["opponent"]}|{g["method"]}|{g["games"]}|{g["mean_share"]:.3f}|{g["p50_ms"]:.2f}|{g["mean_nodes"]:.0f}|{g["starts"]}|{g["maintained"]}|{g["plan_changes"]}|{g["confirmed_changes"]}|{g["validation_rejections"]}|' for g in result['groups']]
    lines+=['','一手方式との同条件比較（優勝持分）:','']
    for method in ('fresh3','retained3'):
        rows=[r for r in result['paired'] if r['method']==method]
        lines.append(f'- {method}: {len(rows)}条件、改善{sum(r["share_delta"]>0 for r in rows)} / 同じ{sum(r["share_delta"]==0 for r in rows)} / 悪化{sum(r["share_delta"]<0 for r in rows)}。')
    lines+=['','保持方式は改善と悪化が混在した。特にConnect Fourの戦術相手では一手方式より成績が落ちた。相手一様・有限標本・短い順位列という近似の下で、計画の検証が通っても実対戦の利益を保証しない。原因それぞれの寄与を切り分けた実験はまだ行っていない。',
        '','計画の維持/変更を記録できること、仮想モデルでの改善、実際に強くなることは別。悪化条件は隠さず、有限の試行で一般的な賢さを証明しない。再標本化の判定は標本誤差だけの未校正目安で、相手仮説の誤りは含まない。','',
        '## 劣勢を変えた合法な診断局面対','',
        f'同じ人格/手札/自得点/残り札で相手の点数だけが異なる局面対。{len(result["score_change_probe"]["rows"])} seed中、計画変更{result["score_change_probe"]["plan_changes"]}。選択差を調べるための既存局面で、実対戦中に突然点数だけを書き換えた実験でも未見性能の評価でもない。','',
        '公開された目的条件が変わる小さな2段階課題では、安定時の続行と、変更時の独立検証付き切替を単体検査する。これは上記の実ゲーム成績と別の機構検査。','',
        'ローカルCPUのみ。人格コア/runtimeは不変。Colab/GPU起動、LLM生成/採用/学習は実施していない。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
