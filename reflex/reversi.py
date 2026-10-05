"""Headless 8x8 Othello rules, independent referee, unchanged persona core.

Rules: https://www.worldothello.org/about/about-othello/othello-rules/official-rules/english
No learned transfer or professional engine strength is claimed. The adapter owns
board evaluation and optional reply predictions; the core only ranks features.
"""
from collections import Counter
import copy
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import time
import numpy as np
from .core import FEATURES, NEEDS, TRAITS, VALUES, MAX_ACTIONS, MAX_OUTCOMES, Policy, compile_batch, digest
from .cross_games import core_hashes
from .examples import action, context, effect
from .laboratory import profiles
from .teachers import prompt, write_json

RULES_URL = 'https://www.worldothello.org/about/about-othello/othello-rules/official-rules/english'
FULL = (1 << 64)-1
A_FILE = sum(1 << (8*y) for y in range(8))
H_FILE = A_FILE << 7
CORNERS = (0,7,56,63)
CORNER_MASK = sum(1 << i for i in CORNERS)
DIRS = tuple((x,y) for x in (-1,0,1) for y in (-1,0,1) if x or y)
PASS = -1


def shift(bits, dx, dy):
    if dx == 1: bits &= FULL ^ H_FILE
    elif dx == -1: bits &= FULL ^ A_FILE
    distance = dx+8*dy
    return ((bits << distance) & FULL) if distance > 0 else bits >> -distance


@lru_cache(maxsize=100000)
def legal_bits(own, rival):
    empty = FULL ^ (own | rival)
    result = 0
    for dx,dy in DIRS:
        line = shift(own,dx,dy) & rival
        for _ in range(5): line |= shift(line,dx,dy) & rival
        result |= shift(line,dx,dy) & empty
    return result


def squares(bits):
    result = []
    while bits:
        bit = bits & -bits
        result.append(bit.bit_length()-1)
        bits ^= bit
    return result


def flips(own, rival, square):
    if type(square) is not int or not 0 <= square < 64 or (own | rival) & (1 << square): return 0
    result = 0
    for dx,dy in DIRS:
        ray = shift(1 << square,dx,dy); captured = 0
        while ray & rival:
            captured |= ray
            ray = shift(ray,dx,dy)
        if ray & own: result |= captured
    return result


def move_id(square):
    return 'PASS' if square == PASS else chr(97+square%8)+str(1+square//8)


def parse_move(name):
    if name == 'PASS': return PASS
    if not isinstance(name,str) or len(name) != 2 or name[0] not in 'abcdefgh' or name[1] not in '12345678':
        raise ValueError('invalid square')
    return ord(name[0])-97+8*(int(name[1])-1)


@dataclass(frozen=True)
class Position:
    black: int = (1 << 28) | (1 << 35)
    white: int = (1 << 27) | (1 << 36)
    side: int = 1

    def __post_init__(self):
        if self.side not in (-1,1) or min(self.black,self.white) < 0 or (self.black | self.white) > FULL or self.black & self.white:
            raise ValueError('invalid position')

    def discs(self, side):
        return (self.black,self.white) if side == 1 else (self.white,self.black)

    def legal(self):
        return squares(legal_bits(*self.discs(self.side)))

    def terminal(self):
        return legal_bits(self.black,self.white) == 0 and legal_bits(self.white,self.black) == 0

    def play(self, square):
        if self.terminal(): raise ValueError('game already over')
        own,rival = self.discs(self.side)
        if square == PASS:
            if legal_bits(own,rival): raise ValueError('voluntary pass is illegal')
            return Position(self.black,self.white,-self.side)
        captured = flips(own,rival,square)
        if not captured: raise ValueError('move must flip at least one opposing disc')
        own = own | captured | (1 << square); rival ^= captured
        return Position(own,rival,-self.side) if self.side == 1 else Position(rival,own,-self.side)

    def rows(self):
        return [''.join('B' if self.black & (1 << (8*y+x)) else 'W' if self.white & (1 << (8*y+x)) else '.' for x in range(8)) for y in range(8)]


def referee_flips(position, square):
    """Independent coordinate scanning; no shift, legal_bits or flips calls."""
    if square == PASS or not 0 <= square < 64: return []
    cells = [1 if position.black & (1 << i) else -1 if position.white & (1 << i) else 0 for i in range(64)]
    if cells[square]: return []
    out = []; sx,sy = square%8,square//8
    for dx,dy in DIRS:
        x,y = sx+dx,sy+dy; captured = []
        while 0 <= x < 8 and 0 <= y < 8 and cells[8*y+x] == -position.side:
            captured.append(8*y+x); x += dx; y += dy
        if captured and 0 <= x < 8 and 0 <= y < 8 and cells[8*y+x] == position.side: out.extend(captured)
    return sorted(out)


def referee_legal(position):
    return [square for square in range(64) if referee_flips(position,square)]


def audit_move(before, square, after):
    available = referee_legal(before)
    if available != before.legal(): raise AssertionError('fast legal move set differs from independent referee')
    if square == PASS:
        if available or after.black != before.black or after.white != before.white: raise AssertionError('invalid pass')
    else:
        captured = referee_flips(before,square)
        if not captured: raise AssertionError('illegal square')
        expected_black,expected_white = before.black,before.white
        for i in captured+[square]:
            if before.side == 1:
                expected_black |= 1 << i; expected_white &= FULL ^ (1 << i)
            else:
                expected_white |= 1 << i; expected_black &= FULL ^ (1 << i)
        if (after.black,after.white) != (expected_black,expected_white): raise AssertionError('incorrect flip update')
        if (after.black | after.white).bit_count() != (before.black | before.white).bit_count()+1:
            raise AssertionError('occupied squares must increase by exactly one')
    if after.side != -before.side: raise AssertionError('wrong turn')


def ratio(a,b):
    return (a-b)/max(1,a+b)


@lru_cache(maxsize=100000)
def board_features(black,white,side):
    own,rival = (black,white) if side == 1 else (white,black)
    occupied = (own | rival).bit_count()
    corners = ratio((own & CORNER_MASK).bit_count(),(rival & CORNER_MASK).bit_count())
    mobility = ratio(legal_bits(own,rival).bit_count(),legal_bits(rival,own).bit_count())
    frontier = 0
    for dx,dy in DIRS: frontier |= shift(FULL ^ (own | rival),dx,dy)
    protection = -ratio((own & frontier).bit_count(),(rival & frontier).bit_count())
    discs = ratio(own.bit_count(),rival.bit_count())
    if legal_bits(own,rival) == 0 and legal_bits(rival,own) == 0:
        utility = float((own.bit_count() > rival.bit_count())-(own.bit_count() < rival.bit_count()))
    else:
        # Explicit provisional game heuristic, not learned or a published engine.
        utility = .45*corners+.30*mobility+.15*protection+.10*(occupied/64)**3*discs
    return utility,corners,mobility,protection,discs


def utility(position, side):
    return board_features(position.black,position.white,side)[0]


def needs(position, side):
    score,corners,mobility,protection,discs = board_features(position.black,position.white,side)
    return dict(safety=float(np.clip(.5-.35*corners-.15*protection,0,1)),esteem=float(np.clip(.5-.5*score,0,1)))


def consequences(before, after, side):
    u,c,m,f,d = board_features(after.black,after.white,side)
    old = needs(before,side); new = needs(after,side)
    # All profiles see the SAME consequences. Trait/value importance is core-side.
    return effect(u,{key:old[key]-new[key] for key in old},
        values=dict(achievement=u,security=(c+f)/2,power=d),
        style=dict(openness=m,conscientiousness=(c+f)/2,neuroticism=(c+f)/2))


def vector(outcome):
    return np.array([outcome['objective']]+[outcome['needs'].get(k,0) for k in NEEDS]+
        [outcome['values'].get(k,0) for k in VALUES]+[outcome['style'].get(k,0) for k in TRAITS]+[outcome['cost']],dtype=float)


def from_vector(v, p):
    return effect(float(v[0]),dict(zip(NEEDS,map(float,v[1:6]))),dict(zip(VALUES,map(float,v[6:16]))),
                  float(v[-1]),float(p),dict(zip(TRAITS,map(float,v[16:21]))))


def compress_outcomes(outcomes):
    """Preserve probability/first moments, retain worst reply as its own branch.

The remaining branches are at most seven ordered buckets. Aggregation can reduce
within-bucket downside variance; that loss is reported, never called exact risk.
"""
    if len(outcomes) <= MAX_OUTCOMES: return copy.deepcopy(outcomes)
    ordered = sorted(outcomes,key=lambda r:r['objective'])
    result = [copy.deepcopy(ordered[0])]
    for group in np.array_split(np.arange(1,len(ordered)),MAX_OUTCOMES-1):
        p = sum(ordered[int(i)]['p'] for i in group)
        v = sum((ordered[int(i)]['p']*vector(ordered[int(i)]) for i in group),np.zeros(len(FEATURES)))/p
        result.append(from_vector(v,p))
    return result


def forecasts(position, square, depth):
    immediate = position.play(square)
    if depth == 1 or immediate.terminal():
        return [consequences(position,immediate,position.side)],dict(nodes=1,replies=0,compressed=False)
    replies = immediate.legal() or [PASS]
    leaves = [immediate.play(reply) for reply in replies]
    outcomes = [consequences(position,leaf,position.side) for leaf in leaves]
    # Assumed rational opponent response; no actual future opponent move is read.
    scores = np.array([r['objective'] for r in outcomes])
    weights = np.exp(np.clip(-(scores-scores.min())/.12,-80,0)); weights /= weights.sum()
    for row,p in zip(outcomes,weights): row['p'] = float(p)
    return compress_outcomes(outcomes),dict(nodes=1+len(replies),replies=len(replies),compressed=len(outcomes)>MAX_OUTCOMES)


def require_capacity(moves):
    if len(moves) > MAX_ACTIONS:
        raise ValueError('all legal moves exceed the core capacity; refusing to silently prune board moves')


def observe(position, profile, seed, episode, tick, state=None, depth=1):
    if depth not in (1,2): raise ValueError('depth must be one or two plies')
    if position.terminal(): raise ValueError('no decision after game termination')
    moves = position.legal() or [PASS]
    require_capacity(moves)
    acts = []; stats = dict(nodes=0,replies=0,compressed_actions=0,legal_moves=len(position.legal()))
    for square in moves:
        outcomes,detail = forecasts(position,square,depth)
        acts.append(action(move_id(square),*outcomes,confidence=1 if depth == 1 else .85))
        stats['nodes'] += detail['nodes']; stats['replies'] += detail['replies']; stats['compressed_actions'] += detail['compressed']
    c = context('othello',acts,needs(position,position.side),profile['values'],dict(zip(TRAITS,profile['traits'])),mode=None)
    c['scope'].update(game='othello_8x8',episode=episode,npc=profile['id'])
    c['seed'],c['tick'] = seed,tick
    c['objective'] = '終局に自分の色の石を相手より多くする。最強の主義や慎重さなどの判断軸も維持する。'
    c['facts'] = {f'row_{i+1}':row for i,row in enumerate(position.rows())}
    c['facts'].update(side='BLACK' if position.side == 1 else 'WHITE',
        rules='8×8、黒先手。挟んだ相手石を全て返す。合法手がある時のパスは禁止。両者に合法手がなくなれば終局。',
        forecast='現在の盤面だけを使用。深さ1は直後、深さ2は全合法返答を評価して仮定の応答確率を付ける。最悪返答を保持し残り最大7群へ集約。',
        needs='safetyは盤面の守り、esteemは優勢への動機のゲーム用代理。身体/所属/能力成長の欲求は対応しない。')
    for n in NEEDS:
        active = n in ('safety','esteem')
        c['needs'][n].update(supported=active,enabled=active,deficit=needs(position,position.side)[n] if active else None)
    if state is not None: c['state'] = copy.deepcopy(state)
    return c,stats


def neutral_profile():
    result = profiles(True)[0]
    result['id'],result['name'] = 'neutral','中立比較'
    return result


def baseline(position, name, seed, ply):
    moves = position.legal()
    if not moves: return PASS
    if name == 'random':
        return moves[int(digest([seed,ply,position.black,position.white,'baseline'])[:12],16)%len(moves)]
    if name == 'greedy':
        scores = [flips(*position.discs(position.side),move).bit_count() for move in moves]
    elif name == 'minimax2':
        scores = []
        for move in moves:
            after = position.play(move)
            if after.terminal(): score = utility(after,position.side)
            else: score = min(utility(after.play(reply),position.side) for reply in (after.legal() or [PASS]))
            scores.append(score)
    else: raise ValueError('unknown baseline')
    maximum = max(scores)
    ties = [move for move,score in zip(moves,scores) if abs(score-maximum)<1e-12]
    return ties[int(digest([seed,ply,'baseline-tie'])[:12],16)%len(ties)]


def opening(seed, plies=6):
    position = Position(); moves = []
    for ply in range(plies):
        if position.terminal(): break
        move = baseline(position,'random',seed,ply)
        after = position.play(move); audit_move(position,move,after)
        moves.append(move_id(move)); position = after
    return position,moves


def match(profile, opponent='random', seed=0, side=1, depth=1, capture=False):
    position,opening_moves = opening(seed)
    # Depth is excluded from RNG scope for a paired lookahead comparison.
    episode = 'match-'+digest([profile['id'],opponent,seed,side])[:24]
    policy = Policy(); memory = None; traces = []; nodes = 0; compressed = 0; max_legal = 0; max_nodes = 0; times = []; decisions = 0
    for ply in range(128):
        if position.terminal(): break
        c = None; detail = {}
        if position.side == side:
            start = time.perf_counter()
            c,detail = observe(position,profile,seed,episode,ply,memory,depth)
            decision = policy.choose(c)
            times.append((time.perf_counter()-start)*1000)
            memory = decision['next_state']; move = parse_move(decision['action_id']); decisions += 1
            nodes += detail['nodes']; compressed += detail['compressed_actions']; max_legal = max(max_legal,detail['legal_moves'])
            max_nodes = max(max_nodes,detail['nodes'])
        else:
            move = baseline(position,opponent,seed,ply)
        after = position.play(move); audit_move(position,move,after)
        row = dict(ply=ply,side=position.side,move=move_id(move),black=position.black,white=position.white,
                   after_black=after.black,after_white=after.white,persona_turn=position.side == side,forecast=detail)
        if c is not None:
            row['next_state'] = decision['next_state']
            if capture: row['context'] = c
        traces.append(row); position = after
    if not position.terminal(): raise AssertionError('match did not terminate within the rule bound')
    own,rival = position.discs(side)
    result = (own.bit_count()>rival.bit_count())-(own.bit_count()<rival.bit_count())
    return dict(profile=profile['id'],opponent=opponent,seed=seed,side=side,depth=depth,result=result,
        own_discs=own.bit_count(),opponent_discs=rival.bit_count(),empty_squares=64-(own | rival).bit_count(),
        opening=opening_moves,decisions=decisions,forecast_nodes=nodes,compressed_actions=compressed,max_legal_moves=max_legal,
        max_forecast_nodes=max_nodes,decision_ms=times,traces=traces)


def personality_probe():
    """Same public boards, deterministic ranking, held principle mode.

This is a conditional diagnostic, not natural-mode frequency or win strength.
"""
    rows = []
    for seed in (0,1):
        game = match(neutral_profile(),'random',seed,1,1)
        for turn in game['traces'][::5]:
            position = Position(turn['black'],turn['white'],turn['side'])
            for depth in (1,2):
                choices = {}; reference = None
                for p in profiles():
                    c,_ = observe(position,p,seed,'same-board-diagnostic',turn['ply'],depth=depth)
                    primary = max(needs(position,position.side),key=needs(position,position.side).get)
                    c['state'].update(primary_need=primary,mode='principle',mode_urgency=needs(position,position.side)[primary])
                    common = digest([c['actions'],c['facts'],c['needs']])
                    if reference is None: reference = common
                    assert common == reference
                    choices[p['id']] = Policy().choose(c,stochastic=False)['action_id']
                rows.append(dict(seed=seed,ply=turn['ply'],depth=depth,black=position.black,white=position.white,side=position.side,
                    held_mode='principle',choices=choices,different_choices=len(set(choices.values()))>1))
    return dict(positions=len(rows),different_choices=sum(r['different_choices'] for r in rows),rows=rows,
                scope='same board/effects/needs, deterministic ranking in held principle mode; not natural-mode rates')


def experiment(root, seeds=4):
    if type(seeds) is not int or not 1 <= seeds <= 32: raise ValueError('bounded seeds required')
    root = Path(root); root.mkdir(parents=True,exist_ok=True)
    hashes = core_hashes(); roster = profiles()+[neutral_profile()]; original = copy.deepcopy(roster)
    start = time.perf_counter(); results = []; captures = []; candidates = []
    for depth in (1,2):
      for opponent in ('random','greedy','minimax2'):
       for profile in roster:
        for seed in range(seeds):
         for side in (1,-1):
            capture = opponent == 'minimax2' and seed == 0 and side == 1
            game = match(profile,opponent,seed,side,depth,capture)
            if capture:
                captures.append(game)
                rows = [r for r in game['traces'] if 'context' in r]
                for row in (rows[0],rows[len(rows)//2],rows[-1]):
                    c = row['context']
                    candidates.append(dict(id=f"{profile['id']}-d{depth}-p{row['ply']}",family='othello_8x8',
                        context=c,context_hash=digest(c),prompt=prompt(c),quality='awaiting_generation',
                        source='existing_board_game_snapshot',reference_action=row['move'],
                        warning='reference choice and final outcome are not teacher ground truth'))
            # Full contexts belong to selected captures, not every tournament row.
            results.append({k:v for k,v in game.items() if k != 'traces'})
    assert hashes == core_hashes() and roster == original
    candidates = list({r['context_hash']:r for r in candidates}.values())
    probe = personality_probe()
    write_json(root/'trajectories.json',captures)
    (root/'teacher_requests.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False,allow_nan=False)+'\n' for c in candidates),encoding='utf-8')
    evaluation = dict(matches=len(results),seeds=seeds,runs=results,core_hashes=hashes,core_unchanged=True,
        illegal_moves=0,referee_mismatches=0,unterminated_games=0,profile_changes=0,
        decisions=sum(r['decisions'] for r in results),forecast_nodes=sum(r['forecast_nodes'] for r in results),
        compressed_actions=sum(r['compressed_actions'] for r in results),max_legal_moves=max(r['max_legal_moves'] for r in results),
        teacher_snapshots=len(candidates),personality_probe=probe,runtime_seconds=time.perf_counter()-start,rules_url=RULES_URL,
        scope='official 8x8 rules, custom heuristic baselines, small paired openings; not trained behavior or professional engine strength')
    write_json(root/'evaluation.json',evaluation)
    write_report(root,evaluation)
    return evaluation


def write_report(root,evaluation):
    lines = ['# リバーシ/オセロ 8×8・描画なしの検証','',
        '## 得たもの','',
        f"- [WOF公式ルール]({RULES_URL})の8×8・黒先手・全方向の反転・強制パス・両者合法手なしで終局を実装。盤面を描画せず状態/棋譜で検証。",
        f"- {evaluation['matches']}対局・{evaluation['decisions']}人格コアの選択。独立した座標走査の審判が全着手を監査。合法手/反転の不一致0、未終局0。",
        '- 同じ4人格と中立比較、同じコアで、先読みなし/2 ply（自分の一手＋相手の返答）を比較。各開始盤面で黒白を交替。',
        f"- 同じ盤面・候補効果・欲求で主義モードを固定した診断{evaluation['personality_probe']['positions']}条件のうち、人格で選択が分かれたのは{evaluation['personality_probe']['different_choices']}条件。自然なモード選択頻度や勝率とは別の確認。",
        '- 比較相手は自作のランダム合法手、反転数の貪欲法、同じ盤面評価を使う深さ2のミニマックス。公認/強豪エンジンではない。', '',
        '## 削ったもの・残した負担','',
        '- 描画・会話・身体/所属/成長の架空の欲求を省いた。safety/esteemは盤面の守りと優勢への動機のゲーム用代理。心理モデルの実証ではない。',
        '- コアの変更0。盤面の合法性、評価、相手返答の仮定はゲーム側。先読みが人格コア自身から獲得された知能とは扱わない。',
        f"- 2 plyは各候補の全合法返答を評価。8結果枠を超えた{evaluation['compressed_actions']}候補では最悪返答を単独保持、残りを最大7群へ集約。確率質量と期待特徴は保持、群内の損失のばらつきは失う。",
        f"- 合法候補を黙って削らない。観測最大{evaluation['max_legal_moves']}手。{MAX_ACTIONS}手超の局面は現在のコア接続容量不足として明示的に停止する。全盤面対応を証明していない。", '',
        '## 対局結果（勝/分/負）','',
        '|人格|先読み|ランダム|貪欲法|ミニマックス2|','|---|---|---:|---:|---:|']
    for profile in profiles()+[neutral_profile()]:
        for depth in (1,2):
            values = []
            for opponent in ('random','greedy','minimax2'):
                rows = [r for r in evaluation['runs'] if r['profile'] == profile['id'] and r['depth'] == depth and r['opponent'] == opponent]
                counts = Counter(r['result'] for r in rows)
                values.append(f"{counts[1]}/{counts[0]}/{counts[-1]}")
            lines.append(f"|{profile['name']}|{depth} ply|"+'|'.join(values)+'|')
    lines += ['', '## 計算量（盤面変換・先読み・コアを含むローカルCPU）','',
        '|先読み|選択時間の中央値|95百分位|','|---|---:|---:|']
    for depth in (1,2):
        times = [t for r in evaluation['runs'] if r['depth'] == depth for t in r['decision_ms']]
        lines.append(f"|{depth} ply|{np.median(times):.2f} ms|{np.percentile(times,95):.2f} ms|")
    lines += ['', '## 方向性・未確認', '',
        '共有の数値方策へ盤面由来の候補・見込みを渡し、人格/主義は維持する。変わる候補枠は毎手の境界で検査しており、これは固定枠Populationの本番速度測定ではない。計時はウォーム状態のキャッシュと通常デスクトップ負荷を含み、保証値ではない。',
        '同じシード開始盤面を黒白交替で使用する小標本比較。ゲーム開始からの自力の序盤評価、未知の強い対戦相手への一般化、深い終盤探索、対戦からの学習は未検証。先読みの比較は予測確率/確信度の違いも含む。',
        f"教師入力{evaluation['teacher_snapshots']}件は未採用。棋譜/前提はtrajectories.json、集計の正本はevaluation.json。", '']
    (Path(root)/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',default='reflex_artifacts/reversi')
    parser.add_argument('--seeds',type=int,default=4)
    args = parser.parse_args()
    result = experiment(args.output,args.seeds)
    summary = {k:v for k,v in result.items() if k != 'runs'}
    summary['personality_probe'] = {k:v for k,v in result['personality_probe'].items() if k != 'rows'}
    print(json.dumps(summary,ensure_ascii=False,indent=2))
