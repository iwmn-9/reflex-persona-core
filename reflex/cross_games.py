"""Different executable mechanics, one unchanged numerical policy.

These are adapter/integration experiments, not evidence of learned transfer.
Game-side path distances, prerequisite rules and observed opponent frequencies
are explicit. None is inferred from the game name by the decision core.
"""
from collections import Counter, deque
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .core import NEEDS, TRAITS, Policy, compile_batch, digest
from .examples import action, context, effect
from .laboratory import profiles
from .runtime import Population
from .teachers import prompt, write_json


class Game:
    names = ()
    supported = ()

    def __init__(self, profile, seed, variant=0):
        self.profile = copy.deepcopy(profile)
        self.seed = seed
        self.variant = variant
        self.memory = None
        self.tick = 0
        self.observations = []
        self.state = self.initial()

    def needs(self, state):
        raise NotImplementedError

    def branches(self, key):
        raise NotImplementedError

    def aligned(self, before, after, key):
        return {}

    def observation(self):
        # World states in these adapters are public, except Duel's future move.
        return copy.deepcopy(self.state)

    def features(self, before, after, key, probability):
        old, new = self.needs(before), self.needs(after)
        return effect(float(np.clip(self.progress(after)-self.progress(before), -1, 1)),
            {n:float(np.clip(old[n]-new[n], -1, 1)) for n in self.supported},
            self.aligned(before, after, key), p=float(probability),
            cost=0 if self.done(before) else (.01 if key in ('WAIT','REST') else .025))

    def observe(self):
        actions = []
        for key in self.names:
            branches = self.branches(key)
            actions.append(action(key, *(self.features(self.state, after, key, p) for p, after in branches),
                                  legal=self.legal(key), confidence=self.confidence()))
        c = context('cross-game-check', actions, self.needs(self.state), self.profile['values'],
                    dict(zip(TRAITS, self.profile['traits'])), mode=None)
        # Hidden rival configuration cannot affect the actor's random stream.
        scope_variant = 0 if isinstance(self,Duel) else self.variant
        c['scope'].update(game=self.name, episode=f'seed-{self.seed}-variant-{scope_variant}', npc=self.profile['id'])
        c['seed'], c['tick'] = self.seed, self.tick
        c['objective'] = self.objective
        c['facts'] = {('public_'+key):json.dumps(value,sort_keys=True) for key,value in self.observation().items()}
        c['facts'].update(rules=self.rules,estimates=self.estimates())
        for n in NEEDS:
            active = n in self.supported
            c['needs'][n].update(supported=active, enabled=active, deficit=self.needs(self.state)[n] if active else None)
        if self.memory is not None:
            c['state'] = copy.deepcopy(self.memory)
        return c

    def estimates(self):
        return '候補の効果は公開規則から計算した見込み。'

    def confidence(self):
        return 1.

    def advance(self, key):
        if not self.legal(key):
            raise ValueError('game rejected illegal action: '+key)
        options = self.branches(key)
        self.state = copy.deepcopy(options[0][1])
        self.tick += 1

    def summary(self):
        return dict(game=self.name, npc=self.profile['id'], done=self.done(self.state), state=self.observation())


class Navigation(Game):
    name = 'navigation'
    names = ('DOWN','LEFT','REST','RIGHT','UP')
    supported = ('physiology',)
    objective = '障害物を避け、出口へ到達する。疲労時には休める。'
    rules = '7×5の公開地形。壁と外周には移動不可。移動は活動余力を0.12消費、休息は0.4回復。出口到達後は停止。'
    moves = {'DOWN':(0,1),'LEFT':(-1,0),'RIGHT':(1,0),'UP':(0,-1)}

    def initial(self):
        gap = 3 if self.variant == 0 else 1
        self.walls = {(3,y) for y in range(5) if y != gap}
        self.goal = (6,1 if self.variant == 0 else 3)
        self.distances = {self.goal:0}
        queue = deque([self.goal])
        while queue:
            x,y = queue.popleft()
            for dx,dy in self.moves.values():
                q = (x+dx,y+dy)
                if self.open(q) and q not in self.distances:
                    self.distances[q] = self.distances[(x,y)]+1
                    queue.append(q)
        self.initial_distance = self.distances[(0,1)]
        return dict(x=0,y=1,energy=.85,moves=0,rests=0)

    def open(self, point):
        return 0 <= point[0] < 7 and 0 <= point[1] < 5 and point not in self.walls

    def observation(self):
        return dict(self.state, walls=sorted(self.walls), goal=self.goal)

    def estimates(self):
        return '出口までの障害物を考慮した距離はゲーム側BFSで計算。判断コアが地形を解読したとは扱わない。'

    def done(self, state):
        return (state['x'],state['y']) == self.goal

    def needs(self, state):
        return {'physiology':1-state['energy']}

    def progress(self, state):
        return 1-self.distances[(state['x'],state['y'])]/self.initial_distance

    def legal(self, key):
        if key == 'REST':
            return True
        if self.done(self.state):
            return False
        dx,dy = self.moves[key]
        return self.state['energy'] >= .12 and self.open((self.state['x']+dx,self.state['y']+dy))

    def branches(self, key):
        result = copy.deepcopy(self.state)
        if self.legal(key) and not self.done(result):
            if key == 'REST':
                result['energy'] = min(1,result['energy']+.4)
                result['rests'] += 1
            else:
                dx,dy = self.moves[key]
                result['x'] += dx; result['y'] += dy
                result['energy'] = max(0,result['energy']-.12)
                result['moves'] += 1
        return [(1.,result)]

    def aligned(self, before, after, key):
        delta = self.progress(after)-self.progress(before)
        return {'achievement':float(np.clip(delta,-1,1)),
                'security':max(0,self.needs(before)['physiology']-.7)*max(0,after['energy']-before['energy'])}


class Duel(Game):
    name = 'simultaneous_duel'
    names = ('FEINT','GRAPPLE','GUARD','REST','STRIKE')
    supported = ('physiology','safety','esteem')
    objective = '相手の体力を先に尽きさせる同時手番の対戦。相手の当手の選択は見えない。'
    rules = 'STRIKEはGRAPPLE、GRAPPLEはFEINT、FEINTはSTRIKEに勝つ循環。勝ちは相手に0.16損失、負けは自分、引分は両者0.04。攻撃は余力0.12消費。GUARDは0.04損失で0.14回復、RESTは0.10損失で0.35回復。'
    moves = ('FEINT','GRAPPLE','STRIKE')
    beats = {'STRIKE':'GRAPPLE','GRAPPLE':'FEINT','FEINT':'STRIKE'}

    def initial(self):
        return dict(hp=1.,opponent_hp=1.,energy=.8,round=0,wins=0,losses=0)

    def done(self, state):
        return min(state['hp'],state['opponent_hp']) <= 0

    def needs(self, state):
        return dict(physiology=1-state['energy'],safety=1-state['hp'],esteem=max(0,.45-.15*state['wins']))

    def progress(self, state):
        return (1-state['opponent_hp'])+.4*state['hp']

    def legal(self, key):
        return key == 'REST' or (not self.done(self.state) and (key == 'GUARD' or self.state['energy'] >= .12))

    def probabilities(self):
        counts = Counter(self.observations)
        return np.array([counts[k]+1 for k in self.moves],dtype=float)/(len(self.observations)+3)

    def estimates(self):
        return json.dumps(dict(observed_counts=dict(Counter(self.observations)),
            posterior=np.round(self.probabilities(),4).tolist()),sort_keys=True)

    def confidence(self):
        return min(.9,.65+len(self.observations)*.025)

    def resolve(self, key, rival):
        result = copy.deepcopy(self.state)
        if self.done(result) or not self.legal(key):
            return result
        if key == 'REST':
            result['hp'] -= .1; result['energy'] = min(1,result['energy']+.35)
        elif key == 'GUARD':
            result['hp'] -= .04; result['energy'] = min(1,result['energy']+.14)
        else:
            result['energy'] -= .12
            if key == rival:
                result['hp'] -= .04; result['opponent_hp'] -= .04
            elif self.beats[key] == rival:
                result['opponent_hp'] -= .16; result['wins'] += 1
            else:
                result['hp'] -= .16; result['losses'] += 1
        for field in ('hp','opponent_hp','energy'):
            result[field] = max(0,result[field])
        result['round'] += 1
        return result

    def branches(self, key):
        return [(float(p),self.resolve(key,rival)) for p,rival in zip(self.probabilities(),self.moves)]

    def aligned(self, before, after, key):
        hit = before['opponent_hp']-after['opponent_hp']
        loss = before['hp']-after['hp']
        return dict(achievement=hit,power=hit,security=-loss)

    def advance(self, key):
        if not self.legal(key):
            raise ValueError('game rejected illegal action: '+key)
        # A hidden opponent behavior changes; only past observed moves enter input.
        if not self.done(self.state):
            probabilities = (.65,.2,.15) if self.tick < 12 else (.15,.2,.65)
            if self.variant:
                probabilities = tuple(reversed(probabilities))
            u = int(digest([self.seed,self.profile['id'],self.tick,'duel-world'])[:13],16)/2**52
            rival = self.moves[min(int(np.searchsorted(np.cumsum(probabilities),u,side='right')),2)]
            self.state = self.resolve(key,rival)
            self.observations.append(rival)
        self.tick += 1


class Project(Game):
    name = 'prerequisite_project'
    names = ('HELP','REST','TASK_A','TASK_B','TASK_C','WAIT')
    supported = ('physiology','belonging','esteem','growth')
    objective = '前提作業A/Bを終えてCを完成させ、締切までに納品する。別の個体への援助も選べる。'
    rules = 'A/Bは独立、Cは両方の完了後のみ。1作業は余力0.16消費。援助は公開の他者依頼を1件解消し、自己工程は進まない。依頼消化後は援助不可。休息は0.4回復。毎手で締切が1減り、失敗でも終了。'

    def initial(self):
        self.required = (2,2,3) if self.variant == 0 else (3,1,4)
        self.requested_help = 4 if self.variant == 0 else 6
        return dict(a=0,b=0,c=0,energy=.85,time_left=18 if self.variant == 0 else 21,helped=0)

    def complete(self, state):
        return state['c'] == self.required[2]

    def done(self, state):
        return self.complete(state) or state['time_left'] <= 0

    def needs(self, state):
        return dict(physiology=1-state['energy'],belonging=max(0,.65-.25*state['helped']),
                    esteem=0 if self.complete(state) else .45,growth=max(0,.6-(state['a']+state['b']+state['c'])*.1))

    def progress(self, state):
        total = sum(self.required)
        progress = (state['a']+state['b']+state['c'])/total
        return progress+(.5 if self.complete(state) else (-.5 if state['time_left'] <= 0 else 0))

    def legal(self, key):
        if key == 'WAIT':
            return True
        if self.done(self.state):
            return False
        if key == 'REST':
            return True
        if self.state['energy'] < .16:
            return False
        if key == 'HELP':
            return self.state['helped'] < self.requested_help
        field = key[-1].lower()
        index = ('a','b','c').index(field)
        return self.state[field] < self.required[index] and (field != 'c' or (self.state['a'] == self.required[0] and self.state['b'] == self.required[1]))

    def observation(self):
        return dict(self.state,required=self.required,ally_remaining_requests=self.requested_help-self.state['helped'])

    def branches(self, key):
        result = copy.deepcopy(self.state)
        if self.legal(key) and not self.done(result):
            if key == 'REST':
                result['energy'] = min(1,result['energy']+.4)
            elif key == 'HELP':
                result['energy'] -= .16; result['helped'] += 1
            elif key.startswith('TASK_'):
                result[key[-1].lower()] += 1; result['energy'] -= .16
            result['time_left'] -= 1
        return [(1.,result)]

    def aligned(self, before, after, key):
        progress = max(0,self.progress(after)-self.progress(before))
        help_delta = after['helped']-before['helped']
        return dict(achievement=progress,benevolence=help_delta*.3,
                    security=max(0,(1-before['energy'])-.7)*max(0,after['energy']-before['energy']))


GAMES = (Navigation,Duel,Project)
CORE_FILES = ('core.py','runtime.py')


def core_hashes():
    root = Path(__file__).parent
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in CORE_FILES}


def run(seed=0, variant=0, turns=30, engine='mixed', neutral=False, order=None):
    if engine not in ('mixed','separate'):
        raise ValueError('unknown engine')
    roster = profiles(neutral)
    worlds = [cls(p,seed,variant) for cls in GAMES for p in roster]
    if order is not None:
        worlds = [worlds[i] for i in order]
    original = [copy.deepcopy(w.profile) for w in worlds]
    shared = Policy()
    owners = [Population([w.observe() for w in worlds],shared)] if engine == 'mixed' else [Population([w.observe()],shared) for w in worlds]
    traces = []
    for tick in range(turns):
        snapshots = [w.observe() for w in worlds]
        records = []
        groups = [snapshots] if engine == 'mixed' else [[c] for c in snapshots]
        for population, contexts in zip(owners,groups):
            b = compile_batch(contexts)
            # Stable candidate slots; only game-side numeric arrays change.
            updates = {key:getattr(b,key) for key in Population.UPDATES}
            result = population.step(**updates)
            records.extend(result.records(b))
        for world,c,record in zip(worlds,snapshots,records):
            chosen = next(a for a in c['actions'] if a['id'] == record['action_id'])
            if not chosen['legal'] or chosen['known_failure']:
                raise AssertionError('infeasible choice')
            before = world.observation()
            terminal = world.done(world.state)
            world.advance(record['action_id'])
            world.memory = record['next_state']
            traces.append(dict(game=world.name,npc=world.profile['id'],tick=tick,action=record['action_id'],
                before=before,after=world.observation(),context=c,decision=record,terminal_before=terminal))
    assert [w.profile for w in worlds] == original
    return dict(traces=traces,summary=[w.summary() for w in worlds])


def signature(result):
    return digest(sorted((dict(game=r['game'],npc=r['npc'],tick=r['tick'],action=r['action'],
                               after=r['after'],state=r['decision']['next_state']) for r in result['traces']),
                         key=lambda r:(r['tick'],r['game'],r['npc'])))


def experiment(root, seeds=8, turns=30):
    if type(seeds) is not int or not 1 <= seeds <= 64 or type(turns) is not int or not 1 <= turns <= 200:
        raise ValueError('bounded seeds/turns required')
    root = Path(root); root.mkdir(parents=True,exist_ok=True)
    hashes = core_hashes(); start = time.perf_counter()
    summaries = []; mismatches = 0; capture = None; snapshots = []; active = 0
    for variant in (0,1):
        for seed in range(seeds):
            for neutral in (False,True):
                mixed = run(seed,variant,turns,'mixed',neutral)
                separate = run(seed,variant,turns,'separate',neutral)
                active += sum(not row['terminal_before'] for row in mixed['traces'])
                mismatches += signature(mixed) != signature(separate)
                summaries.append(dict(variant=variant,seed=seed,neutral=neutral,actors=mixed['summary']))
                if seed == 0 and variant == 0 and not neutral:
                    capture = mixed
                    for game in (c.name for c in GAMES):
                        for npc in (p['id'] for p in profiles()):
                            rows = [r for r in mixed['traces'] if r['game'] == game and r['npc'] == npc]
                            for row in (rows[0], rows[min(8,len(rows)-1)], rows[-1]):
                                c = row['context']
                                snapshots.append(dict(id=f"{game}-{npc}-{row['tick']}",family=game,source='cross_game_snapshot',
                                    quality='awaiting_generation',context=c,context_hash=digest(c),prompt=prompt(c),
                                    reference_action=row['action'],observed_after=row['after'],
                                    warning='reference action and realized outcome are not teacher ground truth'))
    assert hashes == core_hashes()
    snapshots = list({r['context_hash']:r for r in snapshots}.values())
    write_json(root/'trajectory.json',capture)
    (root/'teacher_requests.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False,allow_nan=False)+'\n' for c in snapshots),encoding='utf-8')
    results = dict(seeds=seeds,turns=turns,actor_episodes=len(summaries)*12,decisions=len(summaries)*12*turns,
        execution_decisions=len(summaries)*24*turns,active_mixed_decisions=active,core_hashes=hashes,core_unchanged=True,
        mixed_separate_mismatches=mismatches,illegal_choices=0,profile_changes=0,
        teacher_snapshots=len(snapshots),runs=summaries,runtime_seconds=time.perf_counter()-start,
        scope='hand-authored adapters and unchanged policy; not learned game transfer or proof of universal intelligence')
    write_json(root/'evaluation.json',results)
    report(root,results)
    return results


def report(root, results):
    lines = ['# 異なるゲーム性でのコア維持検証','',
        '## 得たもの','',
        '- 障害物と疲労のある空間移動、非推移的な同時対戦、前提工程と締切のある作業計画を同じ方策へ接続。各2種類の地形/対戦傾向/工程条件。',
        f"- {results['actor_episodes']}個体エピソード、混合バッチの{results['decisions']}判断。別々のバッチも実行し、総実行{results['execution_decisions']}判断。混合/個別の軌跡差={results['mixed_separate_mismatches']}。",
        f"- 混合バッチのうち終結前の実選択は{results['active_mixed_decisions']}件。残りは終結後の吸収状態の停止選択で、ゲーム成果や賢さの実例には数えない。",
        '- 性格5軸・主義10軸は3ゲームで同一。未対応の欲求は無効化し、疲労や関係性のないゲームに架空の欲求充足を追加しない。',
        f"- 判断コアと永続状態処理は変更0。実行不能選択0。判断前の教師用材料{results['teacher_snapshots']}件、未採用。",'',
        '## 省いたもの・得失','',
        '- 通常の判断にLLMやゲーム名別の分岐を追加していない。比較環境の描画は省いた。',
        '- 万能な観測解読能力は得ていない。地形のBFS、作業の前提条件、過去の相手手からの確率はゲーム側が供給する。接続用変換の実装・効果の尺度設定はゲームごとに必要。',
        '- 相手の当手の選択・未来の行動傾向は入力しない。対戦の予測は観測頻度の平滑化のみで、高度な心理推理ではない。',
        '- 全人格の勝利/納品を必須条件にしない。他者への援助などの自己不利益は残す。効率の悪さと意図したエゴは軌跡で別途確認する。', '',
        '## 結果（到達/対戦勝利/納品）','',
        '|ゲーム|人格あり|中立比較|意味|','|---|---:|---:|---|']
    meanings = {'navigation':'出口到達','simultaneous_duel':'自分の体力を残して勝利。敗北/引分を含まない','prerequisite_project':'締切内のC納品。時間切れは含まない'}
    for cls in GAMES:
        ratios = []
        for neutral in (False,True):
            actors = [a for r in results['runs'] if r['neutral'] == neutral for a in r['actors'] if a['game'] == cls.name]
            success = sum((a['state']['c'] == a['state']['required'][2]) if cls is Project else
                (a['state']['hp'] > 0 and a['state']['opponent_hp'] <= 0) if cls is Duel else a['done'] for a in actors)
            ratios.append(f'{success}/{len(actors)}')
        lines.append(f'|{cls.name}|{ratios[0]}|{ratios[1]}|{meanings[cls.name]}|')
    lines += ['', '## 方向性と限界', '',
        '確認したのは、異なる実行規則と欠けた欲求領域を持つ3環境が、同じ数値契約・固定人格・コアで動き、混合しても各個体の結果が変わらないこと。学習済み方策の未見ゲームへの転移は未確認。ゲーム側が見込みを適切に作る条件に依存する。',
        '対戦の終結は賢さや勝率の証明ではなく、納品失敗もそれだけで人格表現とは決めない。深い探索、連続行動、32候補/8結果を超える候補圧縮、可変候補を頻繁に入れ替える境界は別課題。', '',
        'core.py/runtime.pyのSHA256をevaluation.jsonに保存。trajectory.jsonに人格ありの各手と前提を保存。', '']
    (Path(root)/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',default='reflex_artifacts/cross_games')
    parser.add_argument('--seeds',type=int,default=8)
    parser.add_argument('--turns',type=int,default=30)
    args = parser.parse_args()
    result = experiment(args.output,args.seeds,args.turns)
    print(json.dumps({k:v for k,v in result.items() if k != 'runs'},ensure_ascii=False,indent=2))
