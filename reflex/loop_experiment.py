"""Common-loop ablations on observed traps, changing opponents and future proxies."""
from pathlib import Path
import ast
import copy
import json
import time
import numpy as np
from .core import FEATURES,digest,Policy,compile_batch
from .judgment import Binding,ReadControl
from .judgment_experiment import Arena,Commons,Routes,snapshot,LEARNED,play
from .laboratory import profiles
from .planning import vector
from .decision_loop import DecisionLoop,Request
from .loop_predictor import CategoricalReader
from .teachers import write_json

NAMES=('feint','grapple','strike','cycle','counter_previous','broad')


def arena_models(c):
    moves=Arena.moves;history=ast.literal_eval(c['facts']['public_moves'])
    own=ast.literal_eval(c['facts'].get('public_self_moves','[]'))
    uniform={k:1/3 for k in moves}
    favored=lambda target:{k:.9 if k==target else .05 for k in moves}
    models={name:favored(move) for name,move in zip(NAMES[:3],moves)}
    models['cycle']=favored(moves[(moves.index(history[-1])+1)%3]) if history else uniform
    previous=own[-1] if own and own[-1] in moves else None
    models['counter_previous']=favored(next(k for k in moves if Arena.beats[k]==previous)) if previous else uniform
    models['broad']=uniform
    return models


def arena_outcome(c,act,response):return Arena().result(act,response)


class ChangingArena(Arena):
    def __init__(self,case):
        super().__init__();self.case=case;self.name='cyclic_'+case;self.own=[]

    def facts(self):return dict(super().facts(),public_self_moves=str(self.own[-4:]))

    def situation(self):
        return 'after_own_'+(self.own[-1] if self.own else 'none')

    def step(self,key,tick,seed):
        # Rival commits from past actions, never this tick's selected intent.
        prior=self.own[-1] if self.own else None
        if self.case=='reactive' and prior in self.moves:
            rival=next(k for k in self.moves if self.beats[k]==prior)
        else:
            rival=self.moves[(seed+tick//13)%3]
            if self.case=='noisy':
                rng=np.random.default_rng(int(digest(['public-opponent-world',seed,tick])[:16],16))
                if rng.random()<.2:rival=self.moves[int(rng.integers(3))]
        row=self.result(key,rival);self.history.append(rival);self.own.append(key)
        self.score+=row['objective'];self.losses+=row['objective']<0;self.streak=self.streak+1 if row['objective']<0 else 0
        return row,dict(revealed_rival=rival,score=self.score,loss=row['objective']<0)


FACTORIES={'cyclic_arena':Arena,'four_player_commons':Commons,'branching_delivery':Routes,
           'cyclic_reactive':lambda:ChangingArena('reactive'),'cyclic_noisy':lambda:ChangingArena('noisy')}
MODES=('fixed','legacy_reflex','legacy','integrated_reflex','integrated_read')


def make_request(w,p,seed,tick,trace):
    raw=snapshot(w,p,seed,tick)
    exact=w.exact()
    bindings={a['id']:Binding(a['id'],w.situation(),tuple(f for f in LEARNED if f not in exact[a['id']]) if a['id']!='CHECK' else ()) for a in raw['actions']}
    recent=[r['realized']['objective'] for r in trace[-4:]]
    ahead=w.streak==0 and sum(recent)>1.
    return Request(raw,bindings,exact,maintains_advantage=ahead,threatened=w.streak>=2)


def group(game,seeds,mode,turns=60):
    factory=FACTORIES[game]
    if mode in ('fixed','legacy_reflex','legacy'):
        return [dict(play(factory,p,seed,'fixed' if mode=='fixed' else 'experience' if mode=='legacy_reflex' else 'experience_read',turns),mode=mode)
                for seed in seeds for p in profiles()]
    actors=[]
    for seed in seeds:
        for p in profiles():
            w=factory();raw=snapshot(w,p,seed,0)
            reader=CategoricalReader(raw['scope'],NAMES,arena_models,arena_outcome,known_conditionals=True) if isinstance(w,Arena) and mode=='integrated_read' else None
            actors.append(dict(world=w,profile=p,seed=seed,loop=DecisionLoop(raw,predictor=reader),trace=[],times=[]))
    for tick in range(turns):
        requests=[make_request(a['world'],a['profile'],a['seed'],tick,a['trace']) for a in actors]
        t=time.perf_counter()
        decisions=DecisionLoop.decide_batch([(a['loop'],r) for a,r in zip(actors,requests)])
        elapsed=(time.perf_counter()-t)*1000
        for a,request,r in zip(actors,requests,decisions):
            realized,after=a['world'].step(r['decision']['action_id'],tick,a['seed'])
            event={'revealed_action':after['revealed_rival']} if a['loop'].predictor is not None else None
            update=a['loop'].observe(r['ticket'],vector(realized),event)
            a['times'].append(elapsed/len(actors))
            a['trace'].append(dict(tick=tick,raw=request.context,result=r,realized=realized,public_after=after,update=update))
    result=[]
    for a in actors:
        w=a['world'];trace=a['trace']
        result.append(dict(game=game,profile=a['profile']['id'],seed=a['seed'],mode=mode,turns=turns,
            score=w.score,losses=w.losses,deliveries=getattr(w,'deliveries',None),
            late_losses=sum(r['public_after']['loss'] for r in trace if r['tick']>=45),
            reading_nodes=sum(r['result']['reading']['nodes'] for r in trace),
            reading_changes=sum(r['result']['reading']['adopted'] for r in trace),
            extra_model_uses=sum(r['result']['reading'].get('response_model')=='validated_hypotheses' for r in trace),
            decision_trial_comparisons=sum(r['update']['decision_comparison'] is not None for r in trace),
            revoked=sum(bool((r['update']['reading'] or {}).get('revoked'))+bool(r['update']['learning'].get('revoked')) for r in trace),
            p50_ms=float(np.median(a['times'])),checkpoint=a['loop'].record(),trace=trace))
    return result


def replay(run):
    """Fresh single-owner lifecycle and actual game transitions, no batch sharing."""
    w=FACTORIES[run['game']]();p=next(p for p in profiles() if p['id']==run['profile']);seed=run['seed']
    raw=snapshot(w,p,seed,0)
    reader=CategoricalReader(raw['scope'],NAMES,arena_models,arena_outcome,known_conditionals=True) if isinstance(w,Arena) and run['mode']=='integrated_read' else None
    loop=DecisionLoop(raw,predictor=reader);trace=[]
    for tick,old in enumerate(run['trace']):
        req=make_request(w,p,seed,tick,trace);r=loop.decide(req)
        assert req.context==old['raw'] and r==old['result']
        observed,after=w.step(r['decision']['action_id'],tick,seed)
        event={'revealed_action':after['revealed_rival']} if reader is not None else None
        update=loop.observe(r['ticket'],vector(observed),event)
        assert observed==old['realized'] and after==old['public_after'] and update==old['update']
        assert req.context['personality']==loop.personality and req.context['values']==loop.values
        assert r['reading']['nodes']<=16
        trace.append(dict(realized=observed))
    assert loop.record()==run['checkpoint']
    return len(trace)


def combat_identity(seeds=(0,1)):
    """Route/action lifecycle on real combat; SAME estimates, no fake learning.

    Combat's stationary forecasts/future safety proxies are not certified as
    instant learnable outcomes here. Bindings are intentionally empty. Actual
    observed state change resolves each selected ticket, including actor deaths.
    """
    from .combat import Battle,MAPS,GOALS,alive,terminal,route_context,make_context,reference,resolve,battle_record,potential
    from .combat_experiment import run
    from .decision_loop import IntentRequest
    from .examples import effect
    rows=[];checked=0
    for terrain in MAPS:
        for goal in GOALS:
            for p in profiles():
                for seed in seeds:
                    baseline=run(p,seed,terrain,goal,record=False)
                    w=Battle.start(terrain,goal);team=seed%2;loops={};choices_total=[]
                    while not terminal(w):
                        actors=alive(w,team);items=[]
                        for actor in actors:
                            raw=make_context(w,actor,p,seed,'eliminate' if goal=='eliminate' else 'secure')
                            if actor not in loops:loops[actor]=DecisionLoop(raw)
                            def build(route,actor=actor):
                                c=make_context(w,actor,p,seed,route)
                                return Request(c,{a['id']:Binding(a['id'],'combat-proxy',()) for a in c['actions']},
                                    {a['id']:('cost',) for a in c['actions']})
                            items.append((loops[actor],IntentRequest(route_context(w,actor,p,seed),build,.35)))
                        ds=DecisionLoop.decide_batch(items);choices={i:d['decision']['action_id'] for i,d in zip(actors,ds)}
                        choices_total.append(dict(choices))
                        for actor in alive(w,1-team):choices[actor]=reference(w,actor,seed)
                        after,audit=resolve(w,choices,int(digest(['combat-world',seed])[:16],16))
                        for actor,d in zip(actors,ds):
                            key=d['decision']['action_id'];route=loops[actor].strategy.chosen
                            objective=float(np.clip(potential(after,team,route)-potential(w,team,route),-1,1))
                            fee=next(a['outcomes'][0]['cost'] for a in d['context']['actions'] if a['id']==key)
                            realized=effect(objective,needs={'physiology':(after.units[actor].hp-w.units[actor].hp)/9},cost=fee)
                            update=loops[actor].observe(d['ticket'],vector(realized))
                            assert not loops[actor].memory.entries and not update['learning']['scored']
                            checked+=1
                        w=after
                    assert json.loads(json.dumps(battle_record(w)))==json.loads(json.dumps(baseline['final']))
                    rows.append(dict(map=terrain,goal=goal,profile=p['id'],seed=seed,ticks=w.tick,
                        won=baseline['won'],lost=baseline['lost'],decision_identity=True))
    return dict(battles=len(rows),resolved_decisions=checked,rows=rows,
        scope='exact route/action lifecycle identity, not improved combat intelligence or calibrated future-value learning')


def experiment(output,seeds=(0,1,10,11,20,21,30,31),turns=60,progress=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);runs=[];checked=0
    with (output/'trajectories.jsonl').open('w',encoding='utf-8') as f:
        for game in FACTORIES:
            for mode in MODES:
                rows=group(game,seeds,mode,turns)
                for r in rows:
                    if mode.startswith('integrated'):checked+=replay(r)
                    f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
                runs.extend({k:v for k,v in r.items() if k not in ('trace','checkpoint')} for r in rows)
                if progress:progress(f'decision-loop/{game}/{mode}: {len(runs)} episodes, {checked} integrated decisions replayed')
    summaries=[];paired=[]
    index={(r['game'],r['profile'],r['seed'],r['mode']):r for r in runs}
    for split,ss in (('development',tuple(s for s in seeds if s<30)),('held_out_seeds',tuple(s for s in seeds if s>=30))):
        for game in FACTORIES:
            for mode in MODES:
                rows=[r for r in runs if r['game']==game and r['mode']==mode and r['seed'] in ss]
                if not rows:continue
                summaries.append(dict(split=split,game=game,mode=mode,episodes=len(rows),
                    mean_score=float(np.mean([r['score'] for r in rows])),losses=sum(r['losses'] for r in rows),
                    late_losses=sum(r['late_losses'] for r in rows),reading_nodes=sum(r['reading_nodes'] for r in rows),
                    reading_changes=sum(r['reading_changes'] for r in rows),extra_model_uses=sum(r.get('extra_model_uses',0) for r in rows),decision_trial_comparisons=sum(r.get('decision_trial_comparisons',0) for r in rows),revoked=sum(r.get('revoked',0) for r in rows)))
            for reference in ('fixed','legacy_reflex','legacy'):
                for mode in ('integrated_reflex','integrated_read'):
                    rows=[r for r in runs if r['game']==game and r['mode']==mode and r['seed'] in ss]
                    d=[r['score']-index[(game,r['profile'],r['seed'],reference)]['score'] for r in rows]
                    if d:paired.append(dict(split=split,game=game,reference=reference,mode=mode,pairs=len(d),
                        better=sum(v>1e-8 for v in d),same=sum(abs(v)<=1e-8 for v in d),worse=sum(v< -1e-8 for v in d),mean_delta=float(np.mean(d))))
    result=dict(format='decision-loop-v2',episodes=len(runs),seeds=list(seeds),turns=turns,
        summaries=summaries,paired=paired,runs=runs,replayed_decisions=checked,
        limitations=['three rule mechanics, two extra rival behaviors; authored proxies, not broad human-level intelligence',
            'held-out seeds repeat mechanisms/profiles, are correlated, and do not prove novel-game transfer',
            'online one-step predictions only; delayed/terminal planning feedback is intentionally rejected',
            'CDF/Brier diagnostics and extra-model adoption/revocation thresholds are finite engineering controls, not universal accuracy/strength guarantees',
            'known conditional rules must be independently correct; game supplies equivalence, exact masks and public cues',
            'JSON lifecycle/state copies and extra numeric scoring add CPU cost; existing Population fast path is unchanged',
            'future information value CHECK is game-authored, excluded from instant-outcome learning',
            'poor combat proxy/over-caution and real-time integration remain separate unresolved quality issues',
            'empirical error evidence is diagnostic only, not an automatic revocation of all learning; existing bounded forgetting remains',
            'decision comparisons assume rival response independent of own current sealed root; effect table correctness is the game contract'])
    matched=[p for p in paired if (p['reference'],p['mode']) in (('legacy_reflex','integrated_reflex'),('legacy','integrated_read'))]
    result['quality_gate']=dict(passed=all(p['worse']==0 for p in matched),
        criterion='No lower actual purpose score per matched game/profile/seed than the existing corresponding adaptive mode. Descriptive finite-case acceptance, not universal strength proof.',
        comparisons=matched)
    uses=sum(r.get('extra_model_uses',0) for r in runs)
    comparisons=sum(r.get('decision_trial_comparisons',0) for r in runs)
    result['extra_model_validation']=dict(adopted_uses=uses,known_rule_choice_comparisons=comparisons,
        conclusion='No extra-model adoption in these episodes; integration preserves existing strength but does not demonstrate a strength gain.' if uses==0 else 'Extra-model adoption occurred; matched game scores determine finite-case strength claims.',
        final_source_replayed_decisions=checked,checkpoint_control_fields='Actual Policy and ReadControl stored and replayed; changing either on restore is rejected.')
    result['rejected_design']=dict(reason='Waiting for four forecast comparisons before allowing any experience disabled prior adaptation.',
        integrated_read_vs_legacy={'development_seeds_0_1':dict(better=9,same=9,worse=22),
                                 'inspected_seeds_10_11':dict(better=10,same=6,worse=24)},
        resolution='Keep existing adaptive fallback and forgetting unchanged; empirical forecast scores are diagnostic only. Gate additional hypotheses on both response prediction and known-rule immediate-purpose choice comparisons.',
        prediction_only_second_attempt=dict(episodes=600,integrated_reflex=dict(better=4,same=108,worse=8),integrated_read=dict(better=43,same=69,worse=8),
            reason='Forecast improvement did not imply stronger decisions; revoking all empirical adaptation also harmed strength. Seeds20/21 are now development.'))
    result['combat_identity']=combat_identity()
    if progress:progress(f'decision-loop/combat: {result["combat_identity"]["battles"]} battle outcomes identical')
    write_json(output/'evaluation.json',result)
    lines=['# 判断・経験・相手読みを接続する共通ループ','',f'{len(runs)} episodes; {checked} integrated selections independently replayed.',
        '固定人格4種、development seed0/1/10/11/20/21と未使用seed30/31。10/11と20/21は棄却した案で確認済みなので開発条件へ移した。循環対戦・4人資源干渉・配送、さらに反応型/雑音型の相手。選択前の予測を保存し、実際の公開結果で比較してから経験/仮説を更新する。新方式は固定方式だけでなく既存の適応方式と比較し、経験だけ/読みありを対応させる。',
        '', '|split|game|mode|episodes|mean purpose score|losses|late losses|read adopted|revoked|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for g in summaries:lines.append(f'|{g["split"]}|{g["game"]}|{g["mode"]}|{g["episodes"]}|{g["mean_score"]:.3f}|{g["losses"]}|{g["late_losses"]}|{g["reading_changes"]}|{g["revoked"]}|')
    lines+=['','## 実際の目的得点の比較','','|split|game|reference|mode|better|same|worse|mean delta|','|---|---|---|---|---:|---:|---:|---:|']
    for p in paired:lines.append(f'|{p["split"]}|{p["game"]}|{p["reference"]}|{p["mode"]}|{p["better"]}|{p["same"]}|{p["worse"]}|{p["mean_delta"]:.3f}|')
    lines+=['','## 得たもの・制限','',
        'DecisionLoopへ人格Policy、証明された無駄の除外、個体の意図、OutcomeMemory、採用/撤回EvidenceGate、ReadControl、所有者別HypothesisTracker、RouteStateを接続。個体の入力/モデル検証を全て終えてからバッチを確定し、誤った最後の個体が他個体を先に進めない。',
        '既存の経験更新/忘却方式は維持する。予測誤差を根拠に経験全体を止める案は成績を悪化させたため棄却し、経験の比較誤差は診断情報に留める。個体のcheckpoint再開/未観測の明示破棄/公開手掛かりでの条件別失効を用意。',
        '任意の読みは同じ結果対象だけを変更できる。未来・終局の値を即時結果で学習する接続は拒否する。未知の効果モデルは方法条件別の比較を必要とし、ある方法で正しいことから未試行の別方法を信用しない。条件付き効果が既知のルールである場合、相手の実際に公開された選択の予測を既存の直近公開履歴予測と比較する。一様予測だけを弱い基線にしない。',
        '削ったもの/費用：全ての経験を採用待ちにする初案は、既存方式より対応が遅れたため廃棄した。既存の2観測後・上限0.75の経験混合を適応基線として残し、過去4件による忘却と支持外の大きな観測によるリセットを維持する。経験の比較誤差だけで基線を無効にしない。既知の条件付きルールは直近4公開手の控えめな応答分布で使い、追加の有限仮説は基線より予測がよい比較4件以上に加え、同じ公開応答での即時目的効果も比較4件以上でよい場合に切り替える。ここでは既知のルールによる仮想比較を明示し、未選択手の実観測として記憶へ入れない。追加仮説が悪化すると基線へ戻す。優位中も安い応答予測は公開後に採点するが、根の効果評価は読みが必要な場合だけ。常時探索を必須にせず1個体16root/response評価上限。CPUと入力契約は増える。部品の接続を、強さ/人間味の完成とは扱わない。',
        f'戦闘のRouteState→候補効果→Policy→公開結果も同じ経路へ接続。{result["combat_identity"]["battles"]}対戦・{result["combat_identity"]["resolved_decisions"]}実判断の結果が既存方式と一致。戦闘の将来価値を即時報酬で誤学習しないよう、この接続では経験更新対象を宣言せず、戦闘の賢さを改善したとは扱わない。',
        '初案の既存読み方式との比較はseed0/1で9改善・9同じ・22悪化、10/11で10改善・6同じ・24悪化だった。固定方式だけに勝つ案は採用しない。結果を見た10/11を未見のまま扱わず、予測比較だけの第2案も600 episodeで既存方式より8条件悪化した。反射基線を変更せず、追加読みの判断利益も検査する第3案の未使用条件は30/31にした。',
        f'今回の対応した既存適応方式との有限条件ゲート：{"PASS" if result["quality_gate"]["passed"] else "FAIL"}。反射は既存経験のみ、読み付きは既存経験＋読みと比べる。FAILなら既定方式の置換を認めない。未見ルール/商用ゲームや戦闘の弱さを解決した証拠とは扱わない。',
        f'追加仮説の実採用{uses}回、既知ルールで判断差を比較できたのは{comparisons}回。採用なし/成績維持を賢さの増加とは扱わない。Policy係数と読み予算もcheckpointへ保存し、再開時の別設定への変更を拒否する。',
        '', '## 残る課題','']+['- '+s for s in result['limitations']]+['']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    return result
