"""Fair-information, repeated traps with three executable rule systems.

Synthetic challenge laboratories, not commercial-game strength or humanity
certification. No agent receives driver regime, current rival moves or hidden
blocked route. Game effect semantics below are authored, not learned rules.
"""
from collections import Counter, deque
import copy
import json
from pathlib import Path
import time
import numpy as np
from .core import FEATURES, TRAITS, Policy, compile_batch, digest
from .examples import context, action, effect
from .judgment import Binding, OutcomeMemory, ReadControl, AdaptivePopulation, choose
from .runtime import Population
from .laboratory import profiles
from .planning import vector
from .teachers import write_json

MODES=('fixed','guard','experience','experience_read')
LEARNED=('objective','achievement','power','security','benevolence')


class Arena:
    name='cyclic_arena'
    names=('FEINT','GRAPPLE','STRIKE','GUARD')
    moves=names[:3]
    beats={'STRIKE':'GRAPPLE','GRAPPLE':'FEINT','FEINT':'STRIKE'}

    def __init__(self):
        self.history=[]; self.score=0.; self.losses=0; self.streak=0

    def prior(self):
        return [action(k,effect(.15,values={'achievement':.15,'power':.15,'security':0},cost=.02)) for k in self.moves]+[
            action('GUARD',self.result('GUARD','STRIKE'))]

    def result(self,key,rival):
        if key=='GUARD': return effect(.02,values={'security':.25},cost=.02)
        value=0. if key==rival else .8 if self.beats[key]==rival else -.8
        return effect(value,values={'achievement':value,'power':value,'security':min(value,0)},cost=.02)

    def step(self,key,tick,seed):
        # Driver's move commits before seeing this tick's NPC decision.
        rival=self.moves[(seed%3+(0 if tick<20 else 1 if tick<40 else 2))%3]
        row=self.result(key,rival); self.history.append(rival)
        self.score+=row['objective']; self.losses+=row['objective']<0
        self.streak=self.streak+1 if row['objective']<0 else 0
        return row,dict(revealed_rival=rival,score=self.score,loss=row['objective']<0)

    def facts(self): return dict(public_moves=str(self.history[-4:]),own_score=str(self.score),rules='cyclic contest; GUARD avoids damage but earns little')
    def situation(self): return 'same_ring'
    def exact(self): return {'GUARD':FEATURES,**{k:('cost',) for k in self.moves}}

    def reading(self,c,cap):
        # One-step public-history model only. All root/response transitions
        # counted. No access to the opponent's actual current move or schedule.
        cost=len(self.names)*len(self.moves)
        if len(self.history)<3 or cap<cost: return None,0
        counts=Counter(self.history[-4:]); total=sum(counts.values())+3
        learned=copy.deepcopy(c)
        for a in learned['actions']:
            a['outcomes']=[]
            for rival in self.moves:
                row=self.result(a['id'],rival); row['p']=(counts[rival]+1)/total
                a['outcomes'].append(row)
        learned['facts']['reading']='last four REVEALED moves with one pseudocount per legal response; one-step hypothesis, not current sealed move'
        return learned,cost


class Commons:
    name='four_player_commons'
    names=('CLAIM','SECURE','HELP','WASTEFUL_SECURE')

    def __init__(self):
        self.history=[]; self.score=0.; self.losses=0; self.streak=0

    def result(self,key,rivals):
        claimants=rivals.count('CLAIM')+(key=='CLAIM')
        # Four distinct allocations: rival claims compete; helping transfers
        # to a teammate, explicitly NOT to a competitive opponent.
        gain=1. if key=='CLAIM' and claimants==1 else -.5 if key=='CLAIM' else .38 if 'SECURE' in key else .12
        help=.7 if key=='HELP' else 0.
        return effect(gain,values={'achievement':gain,'power':max(gain,0),
                                  'security':.3 if 'SECURE' in key else min(gain,0),
                                  'benevolence':help},cost=.18 if key=='WASTEFUL_SECURE' else .02)

    def prior(self):
        return [action(k,effect(.85,values={'achievement':.85,'power':.85},cost=.02)) if k=='CLAIM'
                else action(k,self.result(k,('CLAIM',)*3)) for k in self.names]

    def step(self,key,tick,seed):
        rivals=('CLAIM',)*3 if tick<20 or tick>=40 else ('HELP','SECURE','HELP')
        row=self.result(key,rivals); self.history.append(rivals)
        self.score+=row['objective']; self.losses+=row['objective']<0
        self.streak=self.streak+1 if row['objective']<0 else 0
        return row,dict(revealed_rivals=list(rivals),score=self.score,loss=row['objective']<0,
                        teammate_support=row['values']['benevolence'])

    def facts(self): return dict(public_joint_history=str(self.history[-1:]),own_score=str(self.score),rules='four simultaneous allocations; shared claim collisions lose; secure fixed; help supports declared teammate')
    def situation(self):
        # A public cue AFTER reveal, no hidden regime label before it.
        return 'public_claims_'+str(self.history[-1].count('CLAIM')) if self.history else 'unobserved'
    def exact(self): return {k:FEATURES if k!='CLAIM' else ('cost',) for k in self.names}


class Routes:
    name='branching_delivery'
    names=('LEFT','RIGHT','WAIT','ADVANCE','RETURN','CHECK')

    def __init__(self):
        self.room='junction'; self.score=0.; self.losses=0; self.streak=0; self.history=[]; self.deliveries=0
        self.survey=None; self.checks=0

    def prior(self):
        rows=[]
        for k in self.names:
            legal=(self.room=='junction' and k in ('LEFT','RIGHT','WAIT')) or (self.room=='junction' and k=='CHECK' and self.survey is None) or (self.room=='lane' and k=='ADVANCE') or (self.room=='blocked' and k=='RETURN')
            val=.55 if k=='LEFT' else .35 if k=='RIGHT' else .65 if k=='ADVANCE' else -.12 if k=='RETURN' else 0.
            if k in ('LEFT','RIGHT') and self.survey is not None:
                val=-.8 if k==self.survey else val
            if k=='CHECK':
                # Game-authored value of information proxy: gathering can make
                # a currently stalled delivery feasible. NOT immediate reward.
                stalled=len(self.history)>=4 and all(h[0]!='ADVANCE' for h in self.history[-4:])
                val=.2 if stalled else -.03
            rows.append(action(k,effect(val,values={'achievement':val,'power':val,'security':min(val,0)},cost=.02),legal=legal))
        return rows

    def step(self,key,tick,seed):
        if not next(a['legal'] for a in self.prior() if a['id']==key): raise ValueError('illegal route move')
        blocked='LEFT' if tick<20 or tick>=40 else 'RIGHT'
        if key in ('LEFT','RIGHT'):
            fail=key==blocked; value=-.8 if fail else (.55 if key=='LEFT' else .35)
            self.room='blocked' if fail else 'lane'
            if fail and self.survey is not None and self.survey!=blocked: self.survey=None
        elif key=='ADVANCE': value=.65; self.room='junction'; self.deliveries+=1
        elif key=='RETURN': value=-.12; self.room='junction'
        elif key=='CHECK':
            value=-.03; self.survey=blocked; self.checks+=1
        else: value=0.
        row=effect(value,values={'achievement':value,'power':value,'security':min(value,0)},cost=.02)
        self.score+=value; self.losses+=value==-.8; self.streak=self.streak+1 if value==-.8 else 0
        self.history.append((key,self.room))
        return row,dict(room=self.room,score=self.score,loss=value==-.8,deliveries=self.deliveries,checks=self.checks,
                        surveyed_block=self.survey)

    def facts(self): return dict(room=self.room,own_score=str(self.score),recent_visits=str(self.history[-4:]),
                                survey=str(self.survey),rules='choose route; ADVANCE delivers, RETURN leaves blocked lane; CHECK costs 1 turn/.03 score to observe blockage; conditions can change afterward',
                                info_value='CHECK: manual +.2 future proxy after 4 turns without delivery; NOT instant reward; skip instant-outcome learning for this action')
    def situation(self): return self.room+'_survey_'+str(self.survey)
    def exact(self): return {k:FEATURES if k not in ('LEFT','RIGHT','CHECK') else ('cost',) for k in self.names}


WORLDS=(Arena,Commons,Routes)


def snapshot(world,profile,seed,tick,state=None):
    c=context('fair-traps',world.prior(),values=profile['values'],traits=dict(zip(TRAITS,profile['traits'])),mode=None)
    c.update(scope=dict(game=world.name,episode=f'seed-{seed}',npc=profile['id']),seed=seed,tick=tick,
             objective='make own progress while retaining fixed personal values',facts=world.facts())
    for n in c['needs']: c['needs'][n]=dict(supported=False,enabled=False,deficit=None)
    if state is not None: c['state']=copy.deepcopy(state)
    return c


def play(world_type,profile,seed,mode,turns=60):
    if mode not in MODES: raise ValueError('unknown ablation')
    world=world_type(); state=None; trace=[]; memory=None; times=[]
    for tick in range(turns):
        start=time.perf_counter(); raw=snapshot(world,profile,seed,tick,state)
        if memory is None: memory=OutcomeMemory(raw['scope'])
        exact=world.exact()
        bindings={a['id']:Binding(a['id'],world.situation(),tuple(f for f in LEARNED if f not in exact[a['id']]) if a['id']!='CHECK' else ()) for a in raw['actions']}
        c,experience=memory.prepare(raw,bindings) if mode.startswith('experience') else (raw,{})
        d,guard=choose(c,exact if mode!='fixed' else None)
        reading=dict(requested=False,nodes=0,changed=False,reason='reflex')
        if mode=='experience_read':
            b=compile_batch([c]); result=Policy().decide(b)
            available=isinstance(world,Arena) and len(world.history)>=3
            # Advantage protection uses known own performance over recent turns;
            # no future-policy/hidden-state access. A loss breaks the protection.
            maintained=world.streak==0 and sum(r['realized']['objective'] for r in trace[-4:])>1.
            gate=ReadControl(max_nodes=16).request(b,result,np.array([available]),np.array([maintained]),np.array([world.streak>=2]))
            if gate['requested'][0]:
                learned,nodes=world.reading(c,int(gate['nodes'][0])); reading.update(requested=True,nodes=nodes)
                if learned is not None:
                    lb=compile_batch([learned]); lr=Policy().decide(lb); nd=lr.records(lb)[0]
                    old=lb.ids[0].index(d['action_id']); new=lb.ids[0].index(nd['action_id'])
                    gain=float(lr.scores[0,new]-lr.scores[0,old])
                    if not lr.eligible[0,old] or gain>.05:
                        reading.update(changed=nd['action_id']!=d['action_id'],reason='public_hypothesis_gain'); c,d=learned,nd
                    else: reading['reason']='gain_unconfirmed'
        ticket=memory.commit(c,d,bindings) if mode.startswith('experience') else None
        times.append((time.perf_counter()-start)*1000)
        realized,public_after=world.step(d['action_id'],tick,seed)
        update=memory.observe(ticket,vector(realized)) if ticket is not None else None
        trace.append(dict(tick=tick,context=c,decision=d,guard=guard,experience=experience,
                          reading=reading,realized=realized,public_after=public_after,update=update))
        state=d['next_state']
    return dict(game=world.name,profile=profile['id'],seed=seed,mode=mode,turns=turns,
                score=world.score,losses=world.losses,deliveries=getattr(world,'deliveries',None),
                reading_nodes=sum(r['reading']['nodes'] for r in trace),
                reading_changes=sum(r['reading']['changed'] for r in trace),
                late_losses=sum(r['public_after']['loss'] for r in trace if r['tick']>=45),
                p50_ms=float(np.median(times)),trace=trace)


def conflict_probe():
    # Genuine incompatible benefits; no guard should erase generosity or risk.
    rows=[]
    world=Commons()
    for p in profiles():
        c=snapshot(world,p,0,0)
        # Public condition with no rival claims: use actual known rule results.
        for a in c['actions']: a['outcomes']=[world.result(a['id'],('HELP','SECURE','HELP'))]
        c['state']['mode']='principle'; d,s=choose(c,{k:FEATURES for k in world.names},False)
        rows.append(dict(profile=p['id'],action=d['action_id'],removed=s['waste_removed']))
    return rows


def benchmark(n=1000,repeats=7):
    contexts=[snapshot(Commons(),profiles()[i%4],i,0) for i in range(n)]
    b=compile_batch(contexts); masks=np.ones(b.effects.shape[:2]+(len(FEATURES),),dtype=bool)
    from .judgment import avoid_waste
    p=Policy(); timing={}
    for name in ('policy','guard_policy'):
        measurements=[]
        for _ in range(repeats+1):
            t=time.perf_counter(); bb=b if name=='policy' else avoid_waste(b,masks)[0]; p.decide(bb)
            measurements.append((time.perf_counter()-t)*1000)
        timing[name]=float(np.median(measurements[1:]))
    estimated=np.zeros_like(masks)
    for i,ids in enumerate(b.ids):
        for j,k in enumerate(ids):
            if k=='CLAIM': estimated[i,j]=np.array([f in LEARNED for f in FEATURES])
    observed=np.zeros((n,len(FEATURES)))
    for name,pop in (('population_step',Population(contexts)),('adaptive_step_observe',AdaptivePopulation(contexts,estimated))):
        measurements=[]
        for _ in range(repeats+2):
            t=time.perf_counter();pop.step()
            if name=='adaptive_step_observe':pop.observe(observed)
            measurements.append((time.perf_counter()-t)*1000)
        timing[name]=float(np.median(measurements[2:]))
    return dict(npcs=n,actions=4,repeats=repeats,p50_ms=timing,
                excludes='compile, per-owner outcome memory, game transitions, optional reading, IO')


def experiment(root,seeds=8,turns=60,progress=None):
    root=Path(root); root.mkdir(parents=True,exist_ok=True); runs=[]
    for world in WORLDS:
        for seed in range(seeds):
            for p in profiles():
                for mode in MODES: runs.append(play(world,p,seed,mode,turns))
        if progress: progress(f'{world.name}: {seeds*4*4} episodes done')
    groups=[]
    for world in WORLDS:
        for mode in MODES:
            rows=[r for r in runs if r['game']==world.name and r['mode']==mode]
            groups.append(dict(game=world.name,mode=mode,episodes=len(rows),
                               mean_score=float(np.mean([r['score'] for r in rows])),
                               losses=sum(r['losses'] for r in rows),late_losses=sum(r['late_losses'] for r in rows),
                               reading_nodes=sum(r['reading_nodes'] for r in rows),
                               reading_changes=sum(r['reading_changes'] for r in rows),
                               p50_ms=float(np.median([r['p50_ms'] for r in rows]))))
    pairs=[]
    index={(r['game'],r['profile'],r['seed'],r['mode']):r for r in runs}
    for mode in MODES[1:]:
        deltas=[r['score']-index[(r['game'],r['profile'],r['seed'],'fixed')]['score'] for r in runs if r['mode']==mode]
        pairs.append(dict(mode=mode,pairs=len(deltas),better=sum(d>1e-9 for d in deltas),
                          same=sum(abs(d)<=1e-9 for d in deltas),worse=sum(d< -1e-9 for d in deltas)))
    evaluation=dict(seeds=seeds,turns=turns,episodes=len(runs),groups=groups,paired=pairs,
                    conflict=conflict_probe(),benchmark=benchmark(),local_cpu_only=True,
                    inference='synthetic challenges; repeated profiles/seeds are correlated; no general strength proof')
    write_json(root/'evaluation.json',evaluation)
    # Compact JSON keeps replay evidence without duplicating whitespace at every
    # per-turn snapshot. The canonical trace retains ALL selected contexts.
    (root/'trajectories.json').write_text(json.dumps(runs,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf-8')
    lines=['# 人格基線：同じ失敗の反復を減らす共通層','',
           '3つの自作ルール系（循環同時対戦/4人資源干渉/分岐経路と帰還）による診断。市販ゲームや人間同等の知性の合格証ではない。人格/主義固定、同じ合法手と情報。隠れた当手・通路封鎖・方策変更時刻はdriverだけが持つ。20/40手で環境が変わるが、現在のregimeを判断入力に渡さない。','',
           '|ルール系|方式|平均目的得点|損失選択|45手以後の損失|判断中央値ms|','|---|---|---:|---:|---:|---:|']
    lines += [f"|{g['game']}|{g['mode']}|{g['mean_score']:.3f}|{g['losses']}|{g['late_losses']}|{g['p50_ms']:.3f}|" for g in groups]
    lines += ['',f'合計{len(runs)}エピソード。paired/人格葛藤/CPU費用の数値正本はevaluation.json。', '',
              '得たもの：同じ人格Policyへ、選択済み実観測の分布と、確定した同じ利益に対する無駄な費用の除外を追加。失敗は世界の永久故障ではなく、条件別の見込み更新。大きい予測外れでは古い証拠を弱める。支援は宣言された味方への効果とし、競争相手の利益を慈善扱いする既存診断の問題をこの接続で避ける。', '',
              '費用/削ったもの：記憶は1個体64方法条件、結果4標本、重み上限0.75。4標本と8結果への圧縮は分布/相関を失う。状況キーと効果意味はゲーム側で指定。試さなくなった手段の回復は自動で知れず、公開の手掛かりがないなら再発見が遅れる。最低2観測でも統計的確信は保証しない。', '',
              '人格差：葛藤では支援/安全/自己利益の違いを残す。損失件数は全人格の目的ではなく、主義に沿った支援や安全選択を得点損として失敗扱いしない。', '',
              '読み：循環対戦だけに公開済み直近4手の仮説モデルを接続し、全根×合法応答12遷移、上限16。close/損失/脅威と優位維持による共通requestを使用。1手の条件付き読みであり、複数手戦術・相手の心理・普遍的勝率は未達。他2環境はモデルがないため反射。', '',
              '情報の取り直し：経路系には1手と得点0.03を払うCHECKを全方式に同じ条件で用意。現在の封鎖はCHECK実行後にだけ観測し、古い観測が矛盾すれば失効する。4手納品がないとCHECKの情報価値を+0.2とするゲーム側の代理評価を使う。この未来便益を即時費用から学習しない。情報価値の自動推定は未実装で、汎用コアが勝手に調査能力を得たとは扱わない。納品数/調査数と即時得点も軌跡に保存する。', '',
              '方向性：ゲタ/透視で補わず、観測後の手段更新を土台へ加える。誰にでも通用するハメ防止、相手の未収録方策への対応、人間らしさ全体は未証明。反射APIと任意読みを分ける。LLM教師採用/学習、Colab起動/Drive反映は実施していない。']
    lines += ['', '数値配列での実行：AdaptivePopulationは単一prior/固定条件用。同じ経験混合をJSONなしで一括更新し、スナップショット経路との選択一致を検査する。1000個体・4候補の費用はevaluation.jsonのbenchmark。ゲーム進行/JSON境界/任意読み/IOを含まない。広い候補集合、異なるPC、実ゲームの1000人同時処理の速度保証にはしない。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return evaluation


if __name__=='__main__':
    print(json.dumps(experiment('reflex_artifacts/judgment',progress=print),ensure_ascii=False,indent=2))
