"""Conditional payback probes on three existing rule engines, without a GUI.

These are controlled decision problems, NOT complete-game win-rate evidence.
Public benchmark continuations are declared and matched; no secret future deck
or real combat RNG is passed to the forecaster. The combat actual seeds differ.
"""
from dataclasses import asdict, replace
from pathlib import Path
from collections import Counter
import copy
import json
import numpy as np
from .core import Policy, TRAITS, compile_batch, digest
from .examples import context, action, effect
from .intertemporal import Branch, forecast, patience
from .deliberation import select
from .decision_loop import DecisionLoop, Request
from .judgment import Binding
from .laboratory import profiles
from .cross_games import core_hashes
from .purpose_experiment import source_hashes


def resource_paths(horizon,gold=2):
    from .resource_world import World, act, step, legal, terminal, world_record
    start=World.start(limit=12)
    start=replace(start,empires=(replace(start.empires[0],stock=(5,4,3,gold)),)+start.empires[1:])
    worth=lambda w:sum(w.empires[0].stock)+w.empires[0].science+w.empires[0].culture
    paths={};traces={}
    for root in legal(start):
        w=start;rows=[];trace=[]
        for t in range(horizon):
            before=w
            if terminal(w):rows.append(effect());continue
            if t==0:w=act(w,root)
            else:
                # Complete a public all-wait production cycle. The first flow
                # pays the root before crediting any future production.
                for _ in range(len(w.empires)):w=step(w,'wait')
            gain=(worth(w)-worth(before))/100
            rows.append(effect(gain,values={'achievement':gain}))
            trace.append(dict(before=world_record(before),after=world_record(w),kind='root' if t==0 else 'production',root=root))
        paths[root]=(Branch(1.,tuple(rows),(1.,)*horizon),);traces[root]=trace
    return paths,traces


def auction_paths(horizon):
    from .contests import Public, settle
    start=Public('auction',((),)*4,(0,)*4,(6,)*4,8,'science',(10,))
    paths={};traces={}
    for root in start.legal(0):
        s=start;rows=[];trace=[]
        for t in range(horizon):
            if t>=2:rows.append(effect());continue
            before=s
            # This benchmark publicly declares rivals' capped bid-3 strategy.
            # The singleton remaining prize is public, so no hidden order is
            # revealed. Full unknown-deck auctions are a separate experiment.
            bid=root if t==0 else min(4,s.budgets[0])
            bids=(bid,)+tuple(min(3,b) for b in s.budgets[1:])
            s,_=settle(s,bids)
            gain=(s.scores[0]-before.scores[0])/30
            paid=(before.budgets[0]-s.budgets[0])/18
            rows.append(effect(gain,values={'achievement':gain,'power':gain},cost=.05*paid))
            if t==0:s=replace(s,prize=10,remaining=())
            trace.append(dict(before=asdict(before),after=asdict(s),bids=bids))
        paths['bid:'+str(root)]=(Branch(1.,tuple(rows),(1.,)*horizon),);traces['bid:'+str(root)]=trace
    return paths,traces


def combat_paths(horizon,hp=3,limit=8,seeds=tuple(range(800,808))):
    from .combat import Battle, Unit, resolve, legal, terminal, potential, battle_record
    start=Battle((Unit(0,3,2,hp=hp),Unit(0,0,0,hp=0),Unit(0,0,4,hp=0),
                  Unit(1,5,2),Unit(1,8,0,hp=0),Unit(1,8,4,hp=0)),goal='eliminate',limit=limit)
    paths={};traces={}
    for root in legal(start,0):
        branches=[];root_traces=[]
        for seed in seeds:
            w=start;rows=[];trace=[]
            for t in range(horizon):
                if terminal(w):rows.append(effect());continue
                before=w;choices={}
                for i,u in enumerate(w.units):
                    if u.hp<=0:continue
                    keys=legal(w,i);shots=[k for k in keys if k.startswith('shoot:')]
                    # Public finite continuation: enemy guards initially,
                    # thereafter both shoot if possible, reload or guard.
                    choices[i]=root if i==0 and t==0 else 'guard' if i==3 and t==0 else shots[0] if shots else 'reload' if 'reload' in keys else 'guard'
                w,events=resolve(w,choices,seed)
                g=(potential(w,0,'eliminate')-potential(before,0,'eliminate'))/2
                health=(w.units[0].hp-before.units[0].hp)/9
                damage=(before.units[3].hp-w.units[3].hp)/9
                cost=.08 if choices.get(0,'').startswith('heal:') else .015 if choices.get(0,'').startswith('shoot:') else .005
                rows.append(effect(g,needs={'physiology':health,'safety':health},
                    values={'achievement':g,'power':damage,'security':-float(w.units[0].hp==0 and before.units[0].hp>0)},cost=cost))
                trace.append(dict(before=battle_record(before),after=battle_record(w),choices=choices,seed=seed,events=events))
            branches.append(Branch(1/len(seeds),tuple(rows),(1.,)*horizon));root_traces.append(trace)
        paths[root]=tuple(branches);traces[root]=root_traces
    return paths,traces


def make_context(name,paths,profile,urgent=.1):
    choices=[]
    for root,branches in paths.items():
        rows=[]
        for b in branches:
            row=copy.deepcopy(b.effects[0]);row['p']=b.probability;rows.append(row)
        choices.append(action(root,*rows))
    c=context(name,choices,{'physiology':urgent,'safety':urgent,'growth':.3},profile['values'],dict(zip(TRAITS,profile['traits'])),mode=None)
    c['facts']['continuation']='public conditional benchmark policy, held constant across roots; absorbing terminal earns zero further flow'
    return c


def planning_paths(paths,confidence):
    return {k:tuple(replace(b,confidence=(1.,)+(confidence,)*(len(b.effects)-1)) for b in bs) for k,bs in paths.items()}


def replay(genre,traces):
    checked=0
    if genre=='resources':
        from .resource_world import act,step,world_from_record,world_record
        for trace in traces.values():
            for row in trace:
                w=world_from_record(row['before'])
                if row['kind']=='root':w=act(w,row['root'])
                else:
                    for _ in range(len(w.empires)):w=step(w,'wait')
                assert digest(world_record(w))==digest(row['after']);checked+=1
    elif genre=='auction':
        from .contests import public_from_record,settle
        for trace in traces.values():
            for row in trace:
                s,_=settle(public_from_record(row['before']),tuple(row['bids']))
                if s.round==1:s=replace(s,prize=10,remaining=())
                assert digest(asdict(s))==digest(row['after']);checked+=1
    else:
        from .combat import battle_from_record,battle_record,resolve
        for branches in traces.values():
            for trace in branches:
                for row in trace:
                    w,e=resolve(battle_from_record(row['before']),{int(k):v for k,v in row['choices'].items()},row['seed'])
                    assert digest(battle_record(w))==digest(row['after']) and digest(e)==digest(row['events']);checked+=1
    return checked


def case_specs():
    for h in (1,4,8,12):
        for q in (1.,.25):
            for u in (.1,.9):
                for gold in (2,4):yield dict(genre='resources',horizon=h,confidence=q,urgency=u,gold=gold)
    for h in (1,2,8):
        for q in (1.,.25):
            for u in (.1,.9):yield dict(genre='auction',horizon=h,confidence=q,urgency=u)
    for h in (1,4,8):
        for hp in (3,6):
            for q in (1.,.5):yield dict(genre='combat',horizon=h,confidence=q,urgency=1-hp/9,hp=hp,limit=8)
    for hp in (3,6):yield dict(genre='combat',horizon=8,confidence=1.,urgency=1-hp/9,hp=hp,limit=1)


def experiment(output,progress=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if any((output/name).exists() for name in ('preregister.json','trajectories.jsonl','evaluation.json')):
        raise FileExistsError('frozen payback evidence exists; use a new output directory')
    fixed=source_hashes();core=core_hashes();rows=[];checks=0;cache={};actual_cache={};specs=list(case_specs())
    personas=profiles()+[dict(id='patient-probe',traits=(.5,.95,.5,.5,.1),values={}),
                         dict(id='impatient-probe',traits=(.5,.05,.5,.5,.9),values={})]
    prereg=dict(format='intertemporal-preregister-v1',integration_base='cc36ac188209bf29ea1b2c5b493e2ca03c378930',
        source_hashes=fixed,core_hashes=core,specs=specs,profiles=personas,
        model_combat_seeds=list(range(800,808)),actual_combat_seeds=[190,191],
        status='functional conditions already explored before upstream integration; not unseen strength evidence')
    (output/'preregister.json').write_text(json.dumps(prereg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    with (output/'trajectories.jsonl').open('w',encoding='utf-8') as stream:
        for index,spec in enumerate(specs):
            genre=spec['genre'];h=spec['horizon'];key=(genre,h,spec.get('hp'),spec.get('limit'),spec.get('gold'))
            if key not in cache:
                paths,traces=resource_paths(h,spec['gold']) if genre=='resources' else auction_paths(h) if genre=='auction' else combat_paths(h,spec['hp'],spec['limit'])
                # Replay the JSON boundary; future stochastic rule seeds
                # used in actual comparison are disjoint from model seeds.
                traces=json.loads(json.dumps(traces))
                if genre=='combat':
                    for branches in traces.values():
                        for trace in branches:
                            for row in trace:row['choices']={int(k):v for k,v in row['choices'].items()}
                checks+=replay(genre,traces);cache[key]=paths
                stream.write(json.dumps(dict(spec=spec,traces=traces),ensure_ascii=False)+'\n')
                actual,tr= combat_paths(h,spec['hp'],spec['limit'],seeds=(190,191)) if genre=='combat' else (paths,traces)
                if genre=='combat':
                    tr=json.loads(json.dumps(tr))
                    for bs in tr.values():
                        for trace in bs:
                            for row in trace:row['choices']={int(k):v for k,v in row['choices'].items()}
                    checks+=replay(genre,tr)
                    stream.write(json.dumps(dict(spec=spec,actual=True,traces=tr),ensure_ascii=False)+'\n')
                actual_cache[key]=actual
            paths=cache[key];actual=actual_cache[key];modeled=planning_paths(paths,spec['confidence'])
            for profile in personas:
                # Horizon/forecast confidence must not change the actor's
                # random mode. Only the underlying public physical setup is
                # included in its observation scope.
                name=f"payback-{genre}-hp{spec.get('hp')}-limit{spec.get('limit')}-gold{spec.get('gold')}"
                c=make_context(name,paths,profile,spec['urgency']);original=copy.deepcopy(c)
                numeric=Policy().decide(compile_batch([c]),False);immediate=Policy().choose(c,False)['action_id']
                root_scores=dict(zip(compile_batch([c]).ids[0],map(float,numeric.scores[0])))
                def planner(cs,ds):return forecast(cs[0],modeled,horizon=h,unit='public model events' if genre=='resources' else 'rounds' if genre=='auction' else 'ticks',target=genre+'-incremental-payback')
                request=Request(c,{k:Binding(k,'public',()) for k in paths},{k:() for k in paths})
                loop=DecisionLoop(c);result=DecisionLoop.decide_batch([(loop,request)],False,planner)[0]
                chosen=result['decision']['action_id'];loop.abandon(result['ticket'])
                assert not loop.memory.entries and c==original
                score=lambda root:sum(b.probability*sum(r['objective'] for r in b.effects) for b in actual[root])
                new=score(chosen);old=score(immediate)
                rows.append(dict(**spec,profile=profile['id'],discount=patience(c),immediate=immediate,selected=chosen,
                    immediate_score_loss=root_scores[immediate]-root_scores[chosen],
                    actual_conditional_objective=new,baseline_conditional_objective=old,
                    actual_gain=new-old,deliberation=result['deliberation'],
                    persona_hash=digest([c['personality'],c['values']])))
            if progress and (index+1)%6==0:progress(f'{index+1}/{len(specs)} conditional scenarios; {len(rows)} choices; {checks} rule transitions replayed',flush=True)
    assert fixed==source_hashes() and core==core_hashes()
    summary=[]
    for genre in ('resources','auction','combat'):
        rs=[r for r in rows if r['genre']==genre]
        summary.append(dict(genre=genre,choices=len(rs),changed=sum(r['selected']!=r['immediate'] for r in rs),
            immediate_sacrifice=sum(r['immediate_score_loss']>1e-9 for r in rs),
            better_conditional_objective=sum(r['actual_gain']>1e-9 for r in rs),
            worse_conditional_objective=sum(r['actual_gain']< -1e-9 for r in rs)))
    result=dict(format='intertemporal-v1',scenarios=len(specs),choices=len(rows),summary=summary,
        rule_transitions_replayed=checks,source_hashes=fixed,core_hashes=core,
        model_combat_seeds=list(range(800,808)),actual_combat_seeds=[190,191],
        conditional_continuations=True,whole_game_strength_verified=False,default_policy_changed=False,
        license_granted=False,cloud_runtime_used=False,training_performed=False,runs=rows)
    result['integration_base']=prereg['integration_base']
    result['preregister_sha256']=digest(prereg)
    (output/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    lines=['# 目先の利得と後の見返り','',f'{len(specs)}条件・{len(rows)}判断。{checks:,}ルール遷移を独立再生。',
        '', 'これは既存の資源管理・競り・戦闘ルール上の、継続手段を固定した条件付き判断試験。全ゲームの勝率比較や、長期の賢さの安定を証明する試験ではない。',
        '', '|機構|判断|変更|目前の主観利得を譲った|後の目的値改善|後の目的値悪化|', '|---|---|---|---|---|---|']
    for r in summary:lines.append('|'+ '|'.join(str(r[k]) for k in ('genre','choices','changed','immediate_sacrifice','better_conditional_objective','worse_conditional_objective'))+'|')
    lines.extend(['','## 得たもの',
        '全候補の同じ期間について、実際に一度だけ得る利益・支払う費用を積み上げ、人格Policyに評価させる任意の共通部品intertemporal.pyを追加。DecisionLoopのplannerへ接続し、即時経験の学習と未来予測を分離する。今回の変更では性格/主義/core/runtime/judgmentを変更しない。公開先の履歴移行・追加実装を保全し、cc36ac188209bf29ea1b2c5b493e2ca03c378930を基準に統合して新しいintertemporal_integratedへ実行した。統合前の結果はローカルの旧intertemporalとbackups/before-upstream-integration.zipへ保全し、統合後の結果と混ぜない。',
        '待てる度合いは既存の勤勉性・神経症傾向・現在の生理/安全の切迫度から仮に算出する。成長欲求や最強主義による選好・損失への態度は従来Policyが担当する。この写像はゲーム設計上の仮案で、心理学的に実証された人格診断ではない。',
        '将来の利益だけを確信度で割り引き、予測された損失と費用は確信度の低さで消さない。明示したモデル分岐の確率と、モデルの信用度は別。割引率が高いこと自体を賢さと呼ばない。',
        '', '## 制限したもの・未解決',
        'stockや終局報酬を毎手の利益として繰り返し計上しない。終局後はゼロの増分を使う。未知の将来をゼロで埋めて完全な予測と偽らない。期間外の将来価値は未実装で、投資回収が期間外なら評価できない。正規化分母は全候補共通で、途中損失の切捨てや都合のよい期間延長をしない。',
        '資源は初手の全合法候補を比較後、全員待機の公開生産継続。流動資産・科学・文化の実増減を評価し、建物の存在への未回収ボーナスは付けない。最適な建築/研究計画や完全な勝利レースの試験ではない。',
        '競りは公開の2景品8/10点と、宣言した相手のbid-3継続。残る景品が1個なので非公開順序を覗かない。資金を使うと次を買えない機会費用を実ルールで生む。未知の山札・相手の方針変更・長期の競りは未検証。',
        '戦闘は公開の初手防御→射撃/装填継続、全合法初手、独立のモデル8seed/実際2seed。性格付きの複数手再判断やチーム全体の共同採用を改善したとはしない。2seedの実結果は小標本で、人格の得意不得意や強さの一般証明ではない。',
        '将来の目的値だけで評価した改善/悪化は、本人の主義・健康・費用を含む総合的な正しさのラベルではない。生の目的値が下がっても人格が合理的に別価値を取る場合がある。誤った予測と人格上の譲歩を区別してさらに検証する必要がある。',
        '期間/確信度の比較では同じ観測scopeと乱数streamを保持し、性格が選ぶmodeの偶然の変更と混同しない。戦闘の現在の生理/安全切迫度は実体力の不足1-hp/9に合わせる。開発初回の比較scopeと固定切迫度を修正前にbackups/intertemporal-before-paired-scope.zipへ保全した。初期の戦闘23件の目的値改善は切迫度を低く固定した条件であり、実体力に合わせると従来判断も先に回復し追加差はゼロ。戦闘改善の実績には採用しない。今回の条件は開発・機能検証であり、完全未使用条件の強さの証明ではない。',
        '348判断は独立348局ではなく、同じ初期局面の期間・確信度・切迫度・人格を変えた反実仮想比較を含む。資源で目前の主観利得を譲ったのは待てる設定の2条件（8/12イベント）で、同じ初期局面の寺院投資。競りの6条件は3人格×2期間で同じ2景品の局面。大標本の汎用強度と扱わない。',
        '標準Policy/既存戦闘・資源plannerを置換しない。任意の1個体向けplanner契約の部品と反例を残す。複数個体の人格付き長期共同計画、継続と再判断のずれ、遅延成果の学習/忘却、学習済み末端価値は未完成。',
        '', 'Colab/GPU/Drive/LLM学習は使用せずローカルCPUのみ。公開は集計のみで私的な全候補軌跡は含めない。復元はbackups/before-intertemporal.zip。'])
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    return result
