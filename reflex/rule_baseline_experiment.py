"""Controlled first baseline; assess rule-driven choices, not learned transfer."""
import copy
import json
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
from .core import digest
from .laboratory import profiles
from .board_models import ConnectPosition,ThanksPosition,connect_referee,thanks_referee_score
from .goofspiel import Position,Pending,referee
from .rollout_boards import connect_tactical,thanks_tactical
from .rule_baseline import MODES,ConnectRules,GoofRules,ThanksRules,decide
from .monte_carlo import RolloutBudget
from .teachers import prompt,write_json
from .board_planning import core_audit
from .cross_games import core_hashes

BUDGET=RolloutBudget(samples=24,min_samples=4,max_nodes=30000,max_steps=512,rollout_policy='random')


def setup(game,players,seed):
    if game=='connect_four':return ConnectRules(),ConnectPosition(),()
    if game=='goofspiel':return GoofRules(players),Pending(Position.start(players)),()
    if game!='no_thanks_basic':raise ValueError('unknown baseline game')
    rng=np.random.default_rng(710000+seed*17+players)
    deck=tuple(int(c) for c in rng.permutation(np.arange(3,36))[:24])
    return ThanksRules(players),ThanksPosition.start(players,deck[0]),deck[1:]


def check(rules,before,chosen,after):
    if rules.game=='goofspiel':referee(before.position,chosen,after.position);return
    if rules.game=='connect_four':
        w,names,grid=connect_referee(before)
        assert names==before.legal() and w==before.winner() and chosen in names
        w,names,grid=connect_referee(after)
        assert names==after.legal() and w==after.winner()
        return
    assert sum(before.chips)+before.pot==rules.total_chips
    assert sum(after.chips)+after.pot==rules.total_chips
    assert chosen in before.legal()
    assert len(before.seen)==24-before.remaining
    for cards,chips,s in zip(after.cards,after.chips,after.scores()):assert thanks_referee_score(cards,chips)==s
    actor=before.turn
    if chosen=='PASS':
        assert after.pot==before.pot+1 and after.chips[actor]==before.chips[actor]-1
        assert after.turn==(actor+1)%rules.players and after.cards==before.cards
    else:
        assert after.card is None and after.turn==actor
        assert after.chips[actor]==before.chips[actor]+before.pot
        assert sorted(after.cards[actor])==sorted(before.cards[actor]+(before.card,))


def rival(rules,state,actor,u,kind):
    """Benchmark driver only. No tactical reference reaches the baseline model."""
    if rules.game=='goofspiel':
        h=state.position.hands[actor]
        if kind=='random':return h[min(int(u*len(h)),len(h)-1)]
        d=[abs(c-state.position.prizes[state.position.round]) for c in h]; best=min(d)
        options=[c for c,x in zip(h,d) if x<=best+1]
        return options[min(int(u*len(options)),len(options)-1)]
    if kind=='random':
        names=rules.legal(state);return names[min(int(u*len(names)),len(names)-1)]
    if rules.game=='no_thanks_basic':return thanks_tactical(state,None)
    # deterministic driver sub-RNG depends on this draw, never the model stream
    return connect_tactical(state,np.random.default_rng(int(u*(2**53))))


def play(game,players,seed,profile_index,mode,opponent,budget=BUDGET,teachers=None,planner=None):
    rules,state,private_deck=setup(game,players,seed); seat=seed%players
    profile=profiles()[profile_index]; memory=None; plan_memory=None;trace=[]; times=[]; draws=0
    rng=np.random.default_rng(720000+seed*19+players); episode=f'rule-baseline-{game}-{players}-{seed}'
    for tick in range(1024):
        if rules.terminal(state):break
        if game!='goofspiel' and rules.chance(state):
            state=state.draw(private_deck[draws]);draws+=1;continue
        obs=rules.observation(state); stats=None; own=False
        if game=='goofspiel' or rules.actor(state)==seat:
            own=True;start=time.perf_counter()
            if planner is None:c,d,stats=decide(rules,state,seat,profile,mode,budget,seed,tick,episode,memory)
            else:
                from .rolling_plans import decide as plan_decide
                c,d,stats,plan_memory=plan_decide(rules,state,seat,profile,mode,planner,seed,tick,episode,memory,plan_memory)
            times.append((time.perf_counter()-start)*1000);memory=d['next_state'];chosen=d['action_id']
            if teachers is not None and len(teachers)<9 and not any(x['tag']==[game,mode] for x in teachers):
                teachers.append(dict(id=f'rule-baseline-{len(teachers):03d}',tag=[game,mode],context=copy.deepcopy(c),
                    context_hash=digest(c),prompt=prompt(c),quality='awaiting_generation',source='pre_decision_snapshot_no_gold'))
        if game=='goofspiel':
            bids=[]
            for actor in range(players):
                u=float(rng.random())
                bids.append(int(chosen.split(':')[1]) if actor==seat else rival(rules,state,actor,u,opponent))
            after=Pending(state.position.play(tuple(bids)));actual=bids
        else:
            actor=rules.actor(state);u=float(rng.random())
            actual=chosen if own else rival(rules,state,actor,u,opponent)
            after=rules.step(state,actual)
        check(rules,state,actual,after)
        trace.append(dict(tick=tick,observation=obs,own=own,action=actual,
                          stats=stats,after=rules.observation(after),
                          mode=memory['mode'] if own else None))
        if own and planner is not None:trace[-1]['plan_memory']=plan_memory.record()
        state=after
    else:raise AssertionError('legal game failed to end within rule-derived practical cap')
    rewards=rules.rewards(state)
    return dict(game=game,players=players,seed=seed,seat=seat,profile=profile['id'],method=mode,opponent=opponent,
        win_share=rewards.credits[seat],scores=list(rewards.scores),finished=rules.terminal(state),turns=len(trace),
        trace=trace,decision_ms=times,config=asdict(budget if planner is None else planner))


def probe_cases(seeds):
    """Rule-legal random histories; identical observed states across profiles/modes."""
    cases=[]
    for game,players in (('connect_four',2),('goofspiel',3),('goofspiel',4),('no_thanks_basic',3)):
        for seed in range(seeds):
            rules,state,private_deck=setup(game,players,seed);rng=np.random.default_rng(730000+seed*23+players)
            draws=0
            for tick in range(60):
                if rules.terminal(state):break
                if game!='goofspiel' and rules.chance(state):state=state.draw(private_deck[draws]);draws+=1;continue
                if tick in (0,4,8,12,20,32):
                    viewer=0 if game=='goofspiel' else rules.actor(state)
                    cases.append((rules,state,viewer,seed,tick))
                if game=='goofspiel':state=Pending(state.position.play(tuple(h[int(rng.integers(len(h)))] for h in state.position.hands)))
                else:state=rules.step(state,rules.legal(state)[int(rng.integers(len(rules.legal(state))))])
    return cases


def personality_probe(seeds=3,budget=BUDGET):
    rows=[]
    for index,(rules,state,viewer,seed,tick) in enumerate(probe_cases(seeds)):
        for mode in MODES:
            choices=[];regrets=[];models=[]
            for p in profiles():
                _,d,stats=decide(rules,state,viewer,p,mode,budget,seed,tick,f'baseline-probe-{index}')
                choices.append(d['action_id']);regrets.append(stats['estimated_goal_regret']);models.append(stats)
            rows.append(dict(case=index,game=rules.game,players=rules.players,seed=seed,tick=tick,method=mode,
                             observation=rules.observation(state),choices=choices,different=len(set(choices))>1,
                             goal_regrets=regrets,evaluations=models))
    groups=[]
    for game in ('connect_four','goofspiel','no_thanks_basic'):
        for mode in MODES:
            subset=[r for r in rows if r['game']==game and r['method']==mode]
            groups.append(dict(game=game,method=mode,cases=len(subset),personality_differences=sum(r['different'] for r in subset),
                mean_estimated_goal_regret=float(np.mean([x for r in subset for x in r['goal_regrets'] if x is not None]))))
    return dict(groups=groups,rows=rows,limits='random legal histories, small selected seed range; all profiles use same outcomes/streams; no stochastic tie draw')


def semantic_summary(rows):
    summaries=[]
    baseline={r['case']:r for r in rows if r['method']=='rules_only'}
    for mode in MODES[1:]:
        subset=[r for r in rows if r['method']==mode]
        for i,p in enumerate(profiles()):
            losses=[r['goal_regrets'][i] for r in subset if r['goal_regrets'][i] is not None]
            summaries.append(dict(method=mode,profile=p['id'],cases=len(subset),
                choice_changes=sum(r['choices'][i]!=baseline[r['case']]['choices'][i] for r in subset),
                mean_estimated_goal_regret=float(np.mean(losses)),max_estimated_goal_regret=max(losses)))
    return summaries


def experiment(root,seeds=4,probe_seeds=3,progress=None):
    if type(seeds) is not int or seeds<1 or type(probe_seeds) is not int or probe_seeds<1:raise ValueError('positive seed counts')
    root=Path(root);root.mkdir(parents=True,exist_ok=True);start=time.perf_counter();runs=[];teachers=[]
    for game,players in (('connect_four',2),('goofspiel',4),('no_thanks_basic',3)):
        for seed in range(seeds):
            for opponent in ('random','tactical'):
                for mode in MODES:runs.append(play(game,players,seed,seed%4,mode,opponent,teachers=teachers))
        if progress:progress(f'{game}: 累計{len(runs)}局、攻略評価を基線へ渡さず完了')
    probes=personality_probe(probe_seeds);groups=[]
    for game in ('connect_four','goofspiel','no_thanks_basic'):
        for opponent in ('random','tactical'):
            for mode in MODES:
                rr=[r for r in runs if (r['game'],r['opponent'],r['method'])==(game,opponent,mode)]
                ts=[t for r in rr for t in r['trace'] if t['own']]
                groups.append(dict(game=game,opponent=opponent,method=mode,games=len(rr),mean_share=float(np.mean([r['win_share'] for r in rr])),
                    p50_ms=float(np.median([x for r in rr for x in r['decision_ms']])),
                    fallback=sum(not t['stats']['used'] for t in ts),decisions=len(ts),
                    mean_estimated_goal_regret=float(np.mean([t['stats']['estimated_goal_regret'] for t in ts if t['stats']['used']]))))
    diversity=[]
    for game in ('connect_four','goofspiel','no_thanks_basic'):
        for mode in MODES:
            rr=[r for r in runs if r['game']==game and r['method']==mode]
            trajectories={digest([t['action'] for t in r['trace']]) for r in rr}
            diversity.append(dict(game=game,method=mode,games=len(rr),unique_action_trajectories=len(trajectories),
                                  limits='different driver seeds/opponents, not evidence of personality or emergent strategy'))
    result=dict(games=len(runs),finished=sum(r['finished'] for r in runs),referee_mismatches=0,groups=groups,
        trajectory_diversity=diversity,personality_probe=probes,semantic_diagnostic=semantic_summary(probes['rows']),core_unchanged=core_audit(),core_hashes=core_hashes(),
        elapsed_seconds=time.perf_counter()-start,teacher_requests=len(teachers),accepted_llm_rows=0,
        protocol=dict(seeds=seeds,probe_seeds=probe_seeds,budget=asdict(BUDGET),games=['connect_four','goofspiel','no_thanks_basic'],
            modes=list(MODES),per_game_requirements='legal/step/terminal/chance sampler/terminal payoff bounds; no strategy or intermediate evaluator',
            difference='bare winner reward vs shared per-player-return meanings; constant personalities, unsupported needs/styles inactive',
            fairness='same seed/seat/profile/opponent setup across modes, common model streams; live histories diverge after choices',
            limits='rule simulator supplied by us, no natural-language rule learning, no opponent learning, no persistent multi-step strategy controller, no universal value model'))
    write_json(root/'evaluation.json',result);write_json(root/'trajectories.json',runs)
    (root/'teacher_requests.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in teachers),encoding='utf-8')
    write_report(root,result)
    return result


def write_report(root,result):
    root=Path(root);groups=result['groups'];probes=result['personality_probe']
    bare=[r for r in probes['rows'] if r['method']=='rules_persona']
    rich=[r for r in probes['rows'] if r['method']=='reward_persona']
    lines=['# ルールと終局目的だけで動く共通ベースライン','',
        '目標は、ゲーム専用の攻略評価を追加せず、固定人格と共通判断基盤をルールへ接続しただけで展開が生まれるか確認すること。ルール文章を学習した実験ではなく、既存の合法手/遷移/勝敗シミュレータを使う。','',
        f'今回の結論: 同じ処理で3種のゲームを終局まで動かす最低線は構築できた。同じ{len(bare)}局面で勝敗のみの人格差は{sum(r["different"] for r in bare)}、利益の意味づけを加えた人格差は{sum(r["different"] for r in rich)}。ただし意味づけには競争相手を勝たせる問題が残る。攻略評価なしの動作、意味ある人格、賢さ/楽しさの合格を分ける。','',
        '## 得たもの・外したもの・残る限界','',
        '- 得たもの: 2人の空間対戦Connect Four、4人の同時入札Goofspiel、3人の拒否/引受けNo Thanksで、同じ終局評価と人格Policyを接続。',
        '- 外したもの: 連結形状/盤面の脅威係数/札の温存係数/カード引受けの攻略評価/ゲーム別人格効果。仮想継続は全員一様で、攻略参照は実対戦の相手にだけ使う。',
        '- 比較: rules_only=中立・勝敗のみ、rules_persona=固定人格・勝敗のみ、reward_persona=固定人格・終局利益に共通の意味づけを追加。',
        '- 必須のゲーム側情報: 合法手、観測、遷移、終局、未観測の確率モデル、勝敗と得点の方向/範囲。戦略を知らずとも実装可能だが、任意のルール文章から自動で得る機能はない。',
        '- 費用: 毎判断の終局標本は反射より高価。小標本と一様相手の仮定で読み違える。予算不足なら専用評価へ戻さず同点となる。',
        '- 意味づけ: 共通利益→達成/安全/権力/慈善という仮定を置いた。競争の利益は本当の安全・友情を測っていない。欲求/様式/成長/関係性は入力不足なので発明しない。',
        '- 方向性: 攻略を載せなくても動く最低線と、意味情報なしでは出ない人格を分ける。多手の方針保持/変更、相手学習、一般的な価値モデルは未実装。人格コア/runtimeは不変。','',
        '## 実対戦','',f'{result["games"]}局完了。独立審判との不一致0。少数seed・人格はseedで巡回するため、人格ごとの棋力や未見性能の証明ではない。','',
        '|ゲーム|相手|基線|局数|優勝持分|p50 ms|標本不足|判断数|モデル内の目的損失|','|---|---|---|---|---|---|---|---|---|']
    lines += [f'|{g["game"]}|{g["opponent"]}|{g["method"]}|{g["games"]}|{g["mean_share"]:.3f}|{g["p50_ms"]:.2f}|{g["fallback"]}|{g["decisions"]}|{g["mean_estimated_goal_regret"]:.3f}|' for g in groups]
    lines+=['','目的損失は同じ仮想モデルでの最大優勝持分と選択の見込みの差。実際の失敗率/真の最適手との差ではない。人格の主義のための損失も含む。','',
        '## 同じ局面での人格差','',
        '全人格に同じ予測結果を渡し、決定乱数を使わない。局面/予測の差を人格差と誤認しない。','',
        '|ゲーム|基線|局面数|人格で選択が分かれた局面|モデル内の平均目的損失|','|---|---|---|---|---|']
    lines += [f'|{g["game"]}|{g["method"]}|{g["cases"]}|{g["personality_differences"]}|{g["mean_estimated_goal_regret"]:.3f}|' for g in probes['groups']]
    lines+=['','## 意味づけの失敗も含めた人格別の内訳','',
        '|基線|人格|局面数|勝敗のみの中立から選択変更|モデル内の最大目的損失|','|---|---|---|---|---|']
    lines += [f'|{g["method"]}|{g["profile"]}|{g["cases"]}|{g["choice_changes"]}|{g["max_estimated_goal_regret"]:.3f}|' for g in result['semantic_diagnostic']]
    lines+=['','選択の差が増えたことだけでは多様な人格が適切に表現できたとはいえない。特に対戦相手の利益まで慈善と解釈すると、二人零和対戦では相手を勝たせる方向に働く。これを人間らしい賢さの合格として扱わない。',
        '結論の範囲: 専用攻略なしの合法な目的評価は共通化できた。意味ある人格には、誰を利益の対象にするか/欲求の充足/行動様式などの最小限の意味契約が残る。これらはルールの勝敗から一意に復元できない。初期基線では無効化するか、仮定として明示する。全ゲーム別の攻略を必須とする結論でも、ルールだけで人格を解釈できる結論でもない。']
    lines+=['','異なるseedで手順が変わるだけでは人格や駆け引きの証明にならない。勝敗のみで差が乏しければ、ルールの勝敗だけでは人格に必要な意味が足りないという結論にする。意味づけを足した差も人間らしさや楽しさの合格ではなく、後でユーザーと相談する材料。','',
            'ローカルCPUのみ。Colab/GPU起動、教師LLM生成、採用、学習は実施していない。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
