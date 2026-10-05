"""Headless official 2/4-player Quoridor; multi-party effects outside shared Policy.

All legal walls are evaluated. Goal distances describe the public wall geometry,
not exact remaining turns: pawn jumps and future adversarial walls can change it.
No opponent traits, winner oracle, language generation or search enters a decision.
"""
from collections import Counter, deque
import copy
from dataclasses import dataclass, replace
from functools import lru_cache
import json
from pathlib import Path
import time
from zipfile import ZipFile
import numpy as np
from .core import MAX_ACTIONS, NEEDS, TRAITS, Policy, compile_batch, digest
from .cross_games import core_hashes
from .examples import action, context, effect
from .laboratory import profiles
from .teachers import prompt, write_json

RULES_URL = 'https://export.gigamic.com/wp-content/uploads/2021/01/INS-RULES-QUORIDOR_10-2016.pdf'
DIRECTIONS = ((0,1),(1,0),(0,-1),(-1,0))
STARTS = ((4,0),(8,4),(4,8),(0,4))


def index(point): return point[0]+9*point[1]
def point(cell): return cell%9,cell//9
def edge(a,b): return tuple(sorted((index(a),index(b))))
def inside(p): return 0 <= p[0] < 9 and 0 <= p[1] < 9
def goal(p,side): return (p[1]==8,p[0]==0,p[1]==0,p[0]==8)[side]


def wall_edges(wall):
    kind,x,y=wall
    if kind=='H': return (edge((x,y),(x,y+1)),edge((x+1,y),(x+1,y+1)))
    return (edge((x,y),(x+1,y)),edge((x,y+1),(x+1,y+1)))


def fits(wall,walls):
    kind,x,y=wall
    if kind not in ('H','V') or type(x) is not int or type(y) is not int or not (0<=x<8 and 0<=y<8): return False
    for k,a,b in walls:
        if (x,y)==(a,b): return False  # same fence or crossing
        if k==kind and ((kind=='H' and y==b and abs(x-a)<2) or (kind=='V' and x==a and abs(y-b)<2)): return False
    return True


@lru_cache(maxsize=2048)
def graph(walls):
    blocked={e for w in walls for e in wall_edges(w)}
    neighbors=[]
    for cell in range(81):
        x,y=point(cell)
        neighbors.append(tuple(index(q) for dx,dy in DIRECTIONS
                               if inside(q:=(x+dx,y+dy)) and edge((x,y),q) not in blocked))
    return tuple(neighbors)


@lru_cache(maxsize=2048)
def goal_maps(walls):
    neighbors=graph(walls); maps=[]
    for side in range(4):
        distances=[999]*81; queue=deque()
        for cell in range(81):
            if goal(point(cell),side): distances[cell]=0; queue.append(cell)
        while queue:
            cell=queue.popleft()
            for other in neighbors[cell]:
                if distances[other]==999: distances[other]=distances[cell]+1; queue.append(other)
        maps.append(tuple(distances))
    return tuple(maps)


@dataclass(frozen=True)
class Position:
    pawns: tuple
    goals: tuple
    stock: tuple
    walls: tuple = ()
    turn: int = 0

    def __post_init__(self):
        n=len(self.pawns)
        if n not in (2,4) or len(self.goals)!=n or len(self.stock)!=n or not 0<=self.turn<n:
            raise ValueError('official Quoridor supports 2 or 4 players')
        if len(set(self.pawns))!=n or not all(inside(p) for p in self.pawns) or len(set(self.goals))!=n or any(s not in range(4) for s in self.goals):
            raise ValueError('invalid pawns or goals')
        if any(type(s) is not int or s<0 or s>20//n for s in self.stock) or sum(self.stock)+len(self.walls)!=20:
            raise ValueError('fence inventory must be conserved')
        checked=[]
        for w in self.walls:
            if not fits(w,checked): raise ValueError('overlapping or crossing fences')
            checked.append(w)

    @classmethod
    def start(cls,players=4,first=0):
        if players not in (2,4): raise ValueError('game rule supports 2 or 4; shared core has no player-count restriction')
        sides=(0,2) if players==2 else tuple(range(4))
        return cls(tuple(STARTS[s] for s in sides),sides,(20//players,)*players,turn=first)

    def distances(self):
        maps=goal_maps(self.walls)
        return tuple(maps[s][index(p)] for p,s in zip(self.pawns,self.goals))

    def winner(self):
        return next((i for i,(p,s) in enumerate(zip(self.pawns,self.goals)) if goal(p,s)),None)

    def moves(self):
        own=index(self.pawns[self.turn]); occupied={index(p) for i,p in enumerate(self.pawns) if i!=self.turn}
        neighbors=graph(self.walls); out=set()
        for near in neighbors[own]:
            if near not in occupied: out.add(near); continue
            x,y=point(own); nx,ny=point(near); dx,dy=nx-x,ny-y; behind=(nx+dx,ny+dy)
            if inside(behind) and index(behind) in neighbors[near] and index(behind) not in occupied:
                out.add(index(behind))
            else:
                # Behind blocked by edge, fence OR third pawn: sideways, never a chain jump.
                for sx,sy in ((-dy,dx),(dy,-dx)):
                    diagonal=(nx+sx,ny+sy)
                    if inside(diagonal) and index(diagonal) in neighbors[near] and index(diagonal) not in occupied:
                        out.add(index(diagonal))
        return tuple('M:%d:%d'%point(c) for c in sorted(out))

    def wall_candidates(self):
        if not self.stock[self.turn]: return ()
        out=[]
        for kind in ('H','V'):
            for x in range(8):
                for y in range(8):
                    w=(kind,x,y)
                    if not fits(w,self.walls): continue
                    maps=goal_maps(tuple(sorted(self.walls+(w,))))
                    if all(maps[s][index(p)]<999 for p,s in zip(self.pawns,self.goals)):
                        out.append('%s:%d:%d'%w)
        return tuple(out)

    def legal(self):
        return () if self.winner() is not None else self.moves()+self.wall_candidates()

    def play(self,name,validated=False):
        if self.winner() is not None: raise ValueError('game already over')
        if not validated and name not in self.legal(): raise ValueError('illegal action')
        kind,x,y=name.split(':'); x=int(x); y=int(y)
        pawns=list(self.pawns); stock=list(self.stock); walls=self.walls
        if kind=='M': pawns[self.turn]=(x,y)
        else: walls=tuple(sorted(walls+((kind,x,y),))); stock[self.turn]-=1
        return Position(tuple(pawns),self.goals,tuple(stock),walls,(self.turn+1)%len(pawns))


def referee_neighbors(p,walls):
    """Independent coordinate fence predicates, no graph/wall_edges/goal_maps use."""
    x,y=p; out=[]
    for dx,dy in DIRECTIONS:
        q=(x+dx,y+dy)
        if not inside(q): continue
        blocked=False
        for kind,a,b in walls:
            if dx and kind=='V' and min(x,q[0])==a and y in (b,b+1): blocked=True
            if dy and kind=='H' and min(y,q[1])==b and x in (a,a+1): blocked=True
        if not blocked: out.append(q)
    return out


def referee_moves(position):
    own=position.pawns[position.turn]; occupied=set(position.pawns)-{own}; out=set()
    for near in referee_neighbors(own,position.walls):
        if near not in occupied: out.add(near); continue
        dx,dy=near[0]-own[0],near[1]-own[1]; behind=(near[0]+dx,near[1]+dy)
        reachable=referee_neighbors(near,position.walls)
        if behind in reachable and behind not in occupied: out.add(behind)
        else:
            for q in reachable:
                if q not in occupied and q!=own and (q[0]-near[0])*dx+(q[1]-near[1])*dy==0: out.add(q)
    return tuple(sorted('M:%d:%d'%p for p in out))


def referee_distances(position):
    out=[]
    for start,side in zip(position.pawns,position.goals):
        visited={start}; queue=deque([(start,0)]); distance=999
        while queue:
            p,d=queue.popleft()
            if goal(p,side): distance=d; break
            for q in referee_neighbors(p,position.walls):
                if q not in visited: visited.add(q); queue.append((q,d+1))
        out.append(distance)
    return tuple(out)


def referee_wall_valid(wall,position):
    kind,x,y=wall
    if kind not in ('H','V') or not (0<=x<8 and 0<=y<8) or not position.stock[position.turn]: return False
    def segments(w):
        k,a,b=w
        return {(k,a,b),(k,a+int(k=='H'),b+int(k=='V'))}
    units=segments(wall)
    for existing in position.walls:
        if existing[1:]==wall[1:] or units & segments(existing): return False
    # Coordinate referee only; no production graph/distance/fit helpers.
    hypothetical=replace(position,walls=tuple(sorted(position.walls+(wall,))),
                         stock=tuple(s-int(i==position.turn) for i,s in enumerate(position.stock)))
    return max(referee_distances(hypothetical))<999


def audit(before,name,after):
    if sorted(before.moves())!=list(referee_moves(before)): raise AssertionError('pawn-move referee mismatch')
    if name.startswith('M:') and name not in referee_moves(before): raise AssertionError('illegal pawn move')
    if not name.startswith('M:'):
        kind,x,y=name.split(':')
        if not referee_wall_valid((kind,int(x),int(y)),before): raise AssertionError('illegal wall by independent referee')
    expected=before.play(name)
    if expected!=after or before.goals!=after.goals: raise AssertionError('transition mismatch')
    distances=referee_distances(after)
    if distances!=after.distances() or max(distances)>=999: raise AssertionError('distance or goal reachability mismatch')


def consequences(before,after):
    """Public geometry -> generic consequences; independent of persona identity.

    Opponents remain a vector in the adapter. Closest-to-goal progress is a rough
    threat proxy, not a probability of winning. Wall cost and self-delay count.
    """
    actor=before.turn; old=before.distances(); new=after.distances(); rivals=[i for i in range(len(old)) if i!=actor]
    delay=[new[i]-old[i] for i in range(len(old))]; progress=-delay[actor]
    lead=min(old[i] for i in rivals); threat_gain=min(new[i] for i in rivals)-lead
    pressure=max((delay[i]*max(0,1-(old[i]-lead)/4) for i in rivals),default=0)
    wall=int(before.walls!=after.walls); harm=sum(max(0,delay[i]) for i in rivals)/(len(rivals)*3)
    margin=lead-old[actor]; win=after.winner()==actor
    # Rule-based immediate threat, not an opponent-model rollout: can the next
    # actor legally reach its goal now? Public pawn jumps count for this check.
    next_finish=not win and any(goal(tuple(map(int,name.split(':')[1:])),after.goals[after.turn]) for name in after.moves())
    utility=(.7*progress+.6*threat_gain-.12*wall)/3
    # Attack interest is discounted when already near an uncontested finish.
    interference=pressure*(.25 if margin>=3 else 1)
    values=dict(achievement=utility,power=(.45*progress+.7*interference-.35*max(0,-progress)-.12*wall)/3,
                security=(progress+.4*threat_gain-.35*wall)/3,
                benevolence=-harm,universalism=-harm,
                self_direction=progress/3)
    if next_finish:
        utility-=.85
        for key in ('achievement','power','security'): values[key]-=.85
    if win: utility=1; values.update(achievement=1,power=1,security=1,self_direction=1)
    values={k:float(np.clip(v,-1,1)) for k,v in values.items()}
    needs=dict(safety=float(np.clip((progress+.4*threat_gain)/3,-1,1)),esteem=float(np.clip(utility,-1,1)))
    if next_finish: needs['safety']=max(-1,needs['safety']-.7)
    style=dict(agreeableness=-min(1,harm),conscientiousness=float(np.clip(progress/4,-1,1)))
    outcome=effect(float(np.clip(utility,-1,1)),needs,values,cost=.09*wall,style=style)
    affected=[i for i in rivals if delay[i]>0]
    target=max(affected,key=lambda i:(delay[i]*max(0,1-(old[i]-lead)/4),-i)) if affected else None
    return outcome,dict(distance_before=list(old),distance_after=list(new),delay=delay,
                        target=target,affected=affected,self_progress=progress,wall=bool(wall),
                        maintains_geometric_lead=margin>=3,next_actor_can_finish=bool(next_finish),utility_proxy=float(utility))


def snapshot(position,profile,seed,tick,episode,state=None):
    names=position.legal()
    if not names: raise ValueError('terminal board cannot request a decision')
    if len(names)>MAX_ACTIONS: raise ValueError('capacity exceeded; refusing to prune legal choices')
    acts=[]; impacts={}
    for name in names:
        after=position.play(name,validated=True); outcome,impact=consequences(position,after)
        a=action(name,outcome); a['target']=None if impact['target'] is None else f"player-{impact['target']}"
        acts.append(a); impacts[name]=impact
    actor=position.turn; distances=position.distances(); rivals=[d for i,d in enumerate(distances) if i!=actor]
    needs=dict(safety=float(np.clip(.3+(distances[actor]-min(rivals))/12,.1,.9)),esteem=.45)
    c=context(episode,acts,needs,profile['values'],dict(zip(TRAITS,profile['traits'])),mode=None)
    for key in NEEDS:
        if key not in needs: c['needs'][key]=dict(supported=False,enabled=False,deficit=None)
    c['scope']=dict(game='quoridor',episode=episode,npc=f'player-{actor}')
    c.update(seed=seed,tick=tick,objective='他者の干渉と自分の負担を考慮し、反対側の辺への到達を目指す')
    if state is not None: c['state']=copy.deepcopy(state)
    c['facts']=dict(pawns=str(position.pawns),goals=str(position.goals),fences_left=str(position.stock),
                    distances=str(distances),distance_meaning='公開の壁だけの最短経路。駒のジャンプや将来の壁による所要手数ではない',
                    estimate='単手の幾何学的効果と直後の手番の合法な先着の有無。相手の性格/実際の次手/勝率は不明')
    for kind in ('H','V'):
        c['facts']['walls_'+kind]=str([(x,y) for k,x,y in position.walls if k==kind])
    # Lossless per-player impact for teacher review; numeric Policy consumes the
    # generic aggregated effects, and target identifies the principal victim only.
    for name,impact in impacts.items():
        c['facts']['impact_'+name]=f"距離変化（player順）:{impact['delay']}; 壁消費:{int(impact['wall'])}; 次の他者が即先着可能:{impact['next_actor_can_finish']}"
    return c,impacts


def play_game(roster,seed,first=0,max_turns=240,neutral=False,teacher_sink=None):
    if type(max_turns) is not int or not 1<=max_turns<=1000: raise ValueError('bounded positive turn limit required')
    position=Position.start(len(roster),first); states=[None]*len(roster); policy=Policy(); trace=[]; times=[]
    episode=f'run-{seed}-{first}-{len(roster)}'; teacher_tags=set()
    for tick in range(max_turns):
        if position.winner() is not None: break
        actor=position.turn; profile=roster[actor]; start=time.perf_counter()
        c,impacts=snapshot(position,profile,seed,tick,episode,states[actor]); b=compile_batch([c]); d=policy.decide(b).records(b)[0]
        times.append((time.perf_counter()-start)*1000); states[actor]=d['next_state']
        name=d['action_id']; after=position.play(name); audit(position,name,after)
        impact=impacts[name]
        trace.append(dict(tick=tick,actor=actor,profile=profile['id'],action=name,mode=d['next_state']['mode'],
                          primary_need=d['next_state']['primary_need'],legal_candidates=len(c['actions']),
                          impact=impact,pawns_before=position.pawns,pawns_after=after.pawns,stock_after=after.stock))
        tag='wall' if impact['wall'] else 'advance'
        if teacher_sink is not None and (profile['id'],tag) not in teacher_tags and len(teacher_sink)<24:
            teacher_tags.add((profile['id'],tag)); case_id=f'quoridor-{len(teacher_sink):03d}'
            teacher_sink.append(dict(case_id=case_id,context=c,context_hash=digest(c),prompt=prompt(c),
                                     quality='awaiting_generation',source='quoridor_pre_decision_snapshot'))
        position=after
    winner=position.winner()
    return dict(players=len(roster),seed=seed,first=first,neutral=neutral,roster=[p['id'] for p in roster],
                winner=None if winner is None else roster[winner]['id'],winner_seat=winner,truncated=winner is None,
                turns=len(trace),trace=trace,decision_ms=times,final_distances=position.distances())


def probe_positions():
    # Explicit public positions, no invented outcome/LLM gold. Same stock for all personas.
    return dict(start=Position.start(4),
                threatening_leader=Position(((4,2),(1,4),(4,6),(2,4)),(0,1,2,3),(5,)*4),
                own_advantage=Position(((3,6),(7,4),(4,7),(1,4)),(0,1,2,3),(5,)*4),
                own_clear_lead=Position(((4,7),(7,4),(4,6),(1,4)),(0,1,2,3),(5,)*4),
                crowded=Position(((4,3),(4,4),(4,5),(3,4)),(0,1,2,3),(5,)*4))


def personality_probe():
    rows=[]; policy=Policy()
    for name,position in probe_positions().items():
        for profile in profiles():
            c,impacts=snapshot(position,profile,7,0,f'probe-{name}')
            # Controlled principle-mode comparison; actual games choose/hold modes themselves.
            c['state'].update(primary_need='esteem',mode='principle',mode_urgency=.45)
            choice=policy.choose(c,False)
            rows.append(dict(position=name,profile=profile['id'],action=choice['action_id'],impact=impacts[choice['action_id']]))
    return rows


def core_change_audit():
    backup=Path(__file__).resolve().parents[1]/'backups/foundation-before-quoridor.zip'
    if not backup.exists(): return dict(source='restored bundle; original archive not included',capacity_only_verified=None)
    with ZipFile(backup) as z: old=z.read('reflex/core.py').decode('utf-8')
    current=Path(__file__).with_name('core.py').read_text(encoding='utf-8')
    if old.replace('MAX_ACTIONS, MAX_OUTCOMES = 32, 8','MAX_ACTIONS, MAX_OUTCOMES = 256, 8')!=current:
        raise AssertionError('unexpected shared core logic change')
    return dict(capacity_only_verified=True,old_max_actions=32,new_max_actions=MAX_ACTIONS,
                note='Policy, RNG and feature contract unchanged; runtime hash separately recorded')


def experiment(root,seeds=3,max_turns=240):
    if type(seeds) is not int or not 1<=seeds<=16: raise ValueError('bounded positive seed count required')
    root=Path(root); root.mkdir(parents=True,exist_ok=True); runs=[]; teachers=[]
    audit_core=core_change_audit(); start=time.perf_counter()
    for seed in range(seeds):
        # Rotate persona seats and balance first mover. Neutral controls repeat identical setup.
        for offset in range(4):
            order=[(i+offset)%4 for i in range(4)]
            for neutral in (False,True):
                runs.append(play_game(profiles(neutral,order),seed,first=seed%4,max_turns=max_turns,neutral=neutral,teacher_sink=teachers if not neutral else None))
        # Two-player geometry/control, same scoring and no player-count branches in consequences.
        for offset in (0,1):
            roster=[profiles()[i] for i in ((0,3) if offset==0 else (3,0))]
            runs.append(play_game(roster,seed,first=seed%2,max_turns=max_turns))
    probe=personality_probe(); groups=[]
    for n,neutral in ((4,False),(4,True),(2,False)):
        group=[r for r in runs if r['players']==n and r['neutral']==neutral]; turns=[t for r in group for t in r['trace']]
        wall=[t for t in turns if t['impact']['wall']]; durations=[d for r in group for d in r['decision_ms']]
        by_profile={}
        for p in ('growth','steady','care','ego'):
            rows=[t for t in turns if t['profile']==p]; w=[t for t in rows if t['impact']['wall']]
            by_profile[p]=dict(decisions=len(rows),walls=len(w),self_delaying_walls=sum(t['impact']['self_progress']<0 for t in w),
                               rivals_delayed=sum(sum(max(0,d) for i,d in enumerate(t['impact']['delay']) if i!=t['actor']) for t in w),
                               wins=sum(r['winner']==p for r in group))
        groups.append(dict(players=n,neutral=neutral,games=len(group),completed=sum(not r['truncated'] for r in group),
                           decisions=len(turns),walls=len(wall),multi_victim_walls=sum(len(t['impact']['affected'])>1 for t in wall),
                           max_candidates=max((t['legal_candidates'] for t in turns),default=0),by_profile=by_profile,
                           decision_p50_ms=float(np.median(durations)),decision_p95_ms=float(np.percentile(durations,95))))
    result=dict(game='official Quoridor 9x9, 2/4 players',rules_url=RULES_URL,seeds=seeds,
                core_audit=audit_core,core_hashes=core_hashes(),groups=groups,personality_probe=probe,
                referee_mismatches=0,truncated_games=sum(r['truncated'] for r in runs),elapsed_seconds=time.perf_counter()-start,
                teacher_requests=len(teachers),accepted_llm_rows=0,
                limitations=['public shortest paths ignore pawn jumps and future fences; not exact win odds',
                             'no opponent tendency learning, alliances, negotiation, retaliation memory or multi-step search here; immediate legal finish checked',
                             'adapter features are hand-authored; not learned transfer or validated psychology',
                             'timing includes adapter/JSON validation/Policy, excludes independent referee and IO',
                             'official game only supports 2/4 players; core target identifiers are not limited to these counts'])
    write_json(root/'evaluation.json',result); write_json(root/'trajectories.json',runs)
    (root/'teacher_requests.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False,allow_nan=False)+'\n' for c in teachers),encoding='utf-8')
    report=['# 多人数の干渉と葛藤：コリドール', '',
            '公式9×9・2人/4人ルールを描画なしで実行。勝者は実際に反対側の辺へ先着した者。上限到達は未完了として区別する。',
            f'ルール一次資料: {RULES_URL}', '', '## 得たもの', '',
            '- 全合法候補から、移動/壁の設置を同じ人格コアで選ぶ。壁は複数の他者と自分の経路にも干渉する。',
            '- 相手ごとの距離変化、複数被害者、有限の壁消費、自己遅延を実際の盤面変化として記録。性格・主義の係数は既存の4人格を固定。',
            '- 次の手番の相手がすぐ先着できるかは、公開の合法移動から検査。距離だけでは見逃すジャンプによる先着も評価に反映する。実際の相手の次手は使わない。',
            '- 相手の影響ベクトルはAdapterで一般的な効果へ集約。単一targetは主な妨害対象で、全員分の影響は相談用factsと軌跡に残す。コアへ多人数専用の重みや分岐を足していない。',
            '- 共通コアの候補上限を32→256へ拡張。候補を黙って間引かない。配列は実際の候補数で確保するため、小さい入力を256枠へ水増ししない。順位付け/乱数/個別状態の仕組みは変更していない。',
            '', '|人数/比較|完了/局数|判断数|壁|複数相手へ影響した壁|最大候補|境界込みp50/p95 ms|', '|---|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        report.append(f"|{g['players']}人 / {'中立' if g['neutral'] else '人格あり'}|{g['completed']}/{g['games']}|{g['decisions']}|{g['walls']}|{g['multi_victim_walls']}|{g['max_candidates']}|{g['decision_p50_ms']:.2f}/{g['decision_p95_ms']:.2f}|")
    report+=['',f"独立した座標規則の合法移動/到達経路検査の不一致0。上限で打ち切った局数: {result['truncated_games']}。", '',
             '人格ごとの壁消費・自己遅延・勝者、同じ盤面の条件付き主義比較はevaluation.json、手順はtrajectories.jsonが正本。勝率は少数の自作方策同士の比較で、棋力保証ではない。',
             '', '## 削ったもの・後回し', '',
             '- 共通コアに4人固定の制約は加えず、検証は2人戦にも広げた。32候補の制約は外した。公式ゲーム自体の人数は2/4人。別ゲームの3/6人などは共通コアへ任意の相手IDで渡せるが、このゲームのルールを勝手に拡張してはいない。',
             '- 今回は深い探索、相手の性格推定、同盟・交渉・恨みの記憶を追加していない。誰を邪魔するかと自己負担の比較は公開盤面の単手評価で行う。',
             '- 経路距離は壁の幾何学に限定。即先着だけは合法ジャンプも含めて検査するが、駒/将来の妨害を含む勝率の精密推定はまだない。',
             '', '## 方向性への影響', '',
             '「分をわきまえる」は自動降参や人格の抑制にしない。先着が近い自分の優位・無意味な壁消費・自分への遠回りを評価に含め、妨害を常に正解にしない。同じ盤面で主義が強いと個人的な不利を選ぶことも残す。数値の対応はゲーム側の初期設計で、判断力の実証は継続課題。',
             '',f"教師相談用の判断前スナップショット{len(teachers)}件。LLM回答生成/採用/学習はこの実験では行っていない。Colabを起動せずローカルCPUで実行。"]
    (root/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    result=experiment(Path(__file__).resolve().parents[1]/'reflex_artifacts/quoridor')
    print(json.dumps(result,ensure_ascii=False,indent=2))
