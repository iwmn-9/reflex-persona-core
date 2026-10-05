"""Two headless tabletop adapters for the optional planning contract.

Connect Four: public deterministic spatial competition.
No Thanks!: basic game only, public card collections/counter transaction ledger;
unseen nine removals and real future deck are never supplied to a search model.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
import copy
import numpy as np
from .core import NEEDS, TRAITS, Policy, compile_batch, digest
from .examples import action, context, effect
from .planning import compress_outcomes, refine

CONNECT_RULES='https://instructions.hasbro.com/en-ca/instruction/connect-4-game'
THANKS_RULES='https://blog.amigo-spiele.de/content/ap/rule/02455-GB-AmigoRule.pdf'


def neutral(): return dict(id='neutral',traits=(.5,)*5,values={})


def make_context(game,state,profile,seed,tick,episode,actions,needs,facts,policy_state=None):
    c=context(episode,actions,needs,profile['values'],dict(zip(TRAITS,profile['traits'])),mode=None)
    for k in NEEDS:
        if k not in needs: c['needs'][k]=dict(supported=False,enabled=False,deficit=None)
    c['scope']=dict(game=game,episode=episode,npc=f'player-{state.turn}')
    c.update(seed=seed,tick=tick,facts=facts)
    if policy_state is not None: c['state']=copy.deepcopy(policy_state)
    return c


def projected_actions(adapter,base,leaves_by_action,viewer):
    actions=[]
    for name,leaves in leaves_by_action.items():
        outcomes=[]
        for p,leaf in leaves:
            e=adapter.consequence(base,leaf,viewer); e['p']=float(p); outcomes.append(e)
        a=action(name,*compress_outcomes(outcomes)); a['target']=adapter.target(base,name)
        actions.append(a)
    return actions


class PersonaModel:
    """Root actor knows its own persona; all other players use a stated assumption.

    It deliberately receives one persona, not the real roster. No hidden opponent
    traits are available for forecasting. Callbacks use deterministic Policy choice;
    live decisions keep the normal isolated near-tie randomness.
    """
    def __init__(self,adapter,viewer,profile,seed,tick,episode,policy_state):
        self.adapter=adapter; self.viewer=viewer; self.profile=copy.deepcopy(profile)
        self.seed=seed; self.tick=tick; self.episode=episode; self.policy_state=copy.deepcopy(policy_state)
        self.policy=Policy(); self.compressed_choices=0

    def __getattr__(self,name): return getattr(self.adapter,name)

    def select(self,state,options):
        profile=self.profile if state.turn==self.viewer else neutral()
        memory=self.policy_state if state.turn==self.viewer else None
        self.compressed_choices+=sum(len(leaves)>8 for leaves in options.values())
        actions=projected_actions(self.adapter,state,options,state.turn)
        c=self.adapter.context(state,profile,self.seed,self.tick,self.episode,actions,memory)
        return self.policy.choose(c,stochastic=False)['action_id']


def observe(adapter,state,profile,budget,seed,tick,episode,policy_state=None):
    names=adapter.legal(state)
    if not names: raise ValueError('no decision in terminal/chance state')
    roots={name:adapter.step(state,name) for name in names}
    model=PersonaModel(adapter,state.turn,profile,seed,tick,episode,policy_state)
    leaves,stats=refine(roots,model,budget)
    actions=projected_actions(adapter,state,leaves,state.turn)
    c=adapter.context(state,profile,seed,tick,episode,actions,policy_state)
    c['facts']['forecast_horizon']=str(stats['reached_depth'])+'手までの仮説。即時の規則効果と将来の推定を区別する'
    stats.update(baseline_transitions=len(roots),compressed_root_actions=sum(len(v)>8 for v in leaves.values()),
                 compressed_continuation_choices=model.compressed_choices,
                 opponent_assumption='neutral game objective, not the actual hidden personality')
    # Planning configuration is provenance outside input/RNG scope, so changes in
    # depth/budget do not change near-tie random draws by themselves.
    return c,stats


@dataclass(frozen=True)
class ConnectPosition:
    discs: tuple = (0,0)
    heights: tuple = (0,)*7
    turn: int = 0

    def legal(self):
        return () if self.winner() is not None or sum(self.heights)==42 else tuple(f'DROP:{c}' for c,h in enumerate(self.heights) if h<6)

    def winner(self):
        for side,bits in enumerate(self.discs):
            for shift in (1,7,6,8):
                pair=bits & (bits>>shift)
                if pair & (pair>>(2*shift)): return side
        return None

    def play(self,name):
        if name not in self.legal(): raise ValueError('illegal drop')
        col=int(name.split(':')[1]); heights=list(self.heights); discs=list(self.discs)
        discs[self.turn]|=1<<(7*col+heights[col]); heights[col]+=1
        return ConnectPosition(tuple(discs),tuple(heights),1-self.turn)


def connect_referee(position):
    """Independent coordinate grid/line scan, no bit-shift win detector."""
    grid=[[next((s+1 for s,b in enumerate(position.discs) if b & (1<<(7*x+y))),0) for x in range(7)] for y in range(6)]
    winners=set()
    for y in range(6):
        for x in range(7):
            for dx,dy in ((1,0),(0,1),(1,1),(1,-1)):
                cells=[(x+i*dx,y+i*dy) for i in range(4)]
                if all(0<=a<7 and 0<=b<6 for a,b in cells):
                    values=[grid[b][a] for a,b in cells]
                    if values[0] and len(set(values))==1: winners.add(values[0]-1)
    if len(winners)>1: raise AssertionError('both sides cannot win in legal play')
    winner=next(iter(winners),None)
    legal=() if winner is not None else tuple(f'DROP:{x}' for x in range(7) if not grid[5][x])
    return winner,legal,grid


@lru_cache(maxsize=20000)
def connect_features(position,side):
    winner=position.winner()
    if winner is not None: return (1. if winner==side else -1.),0.,0.,0.
    own=position.discs[side]; rival=position.discs[1-side]; score=0.; threats=0.; rival_threats=0.
    for cells in CONNECT_LINES:
        a=sum(bool(own & bit) for bit in cells); b=sum(bool(rival & bit) for bit in cells)
        if not b: score+=(0,.01,.06,.22,1)[a]; threats+=int(a==3)
        if not a: score-=(0,.01,.06,.22,1)[b]; rival_threats+=int(b==3)
    central=sum(bool(own & (1<<(21+y))) for y in range(6))/6
    return float(np.clip(score/2,-.85,.85)),min(1,threats/4),min(1,rival_threats/4),central


CONNECT_LINES=[]
for _y in range(6):
    for _x in range(7):
        for _dx,_dy in ((1,0),(0,1),(1,1),(1,-1)):
            _cells=[(_x+i*_dx,_y+i*_dy) for i in range(4)]
            if all(0<=x<7 and 0<=y<6 for x,y in _cells): CONNECT_LINES.append(tuple(1<<(7*x+y) for x,y in _cells))
CONNECT_LINES=tuple(CONNECT_LINES)


class ConnectAdapter:
    game='connect_four'
    def legal(self,s): return s.legal()
    def step(self,s,name): return s.play(name)
    def terminal(self,s): return s.winner() is not None or sum(s.heights)==42
    def chance(self,s): return False
    def target(self,s,name): return f'player-{1-s.turn}'

    def order(self,s,moves):
        # Cheap public tactical predicates/order, no hidden opponent information.
        def wins(col,side):
            bits=s.discs[side] | (1<<(7*col+s.heights[col]))
            return any((p:=(bits & (bits>>d))) & (p>>(2*d)) for d in (1,7,6,8))
        return sorted(moves,key=lambda name:(not wins(int(name.split(':')[1]),s.turn),
                                            not wins(int(name.split(':')[1]),1-s.turn),abs(int(name.split(':')[1])-3),name))

    def consequence(self,before,after,side):
        u,t,r,c=connect_features(after,side); old=connect_features(before,side)
        return effect(u,dict(safety=(old[2]-r)*.5,esteem=(u-old[0])*.5),
                      dict(achievement=u,power=u,security=u*.7+(1-r)*.15,self_direction=u*.5),
                      style=dict(conscientiousness=(1-r)*.2,neuroticism=(1-r)*.2))

    def context(self,s,profile,seed,tick,episode,actions,policy_state=None):
        u,t,r,c=connect_features(s,s.turn)
        facts=dict(own_discs=str(s.discs[s.turn]),other_discs=str(s.discs[1-s.turn]),heights=str(s.heights),
                   board_encoding='7 columns × 6 rows; each column occupies 7 bit slots with an unused sentinel',
                   estimate='公開盤面の連結/脅威の評価。勝率ではない。予測相手の人格は中立の仮定')
        result=make_context(self.game,s,profile,seed,tick,episode,actions,dict(safety=.3+.4*r,esteem=.5),facts,policy_state)
        result['objective']='合法な落下で4連結を目指し、相手の連結を阻む'; return result


def card_points(cards):
    numbers=set(cards)
    return sum(c for c in numbers if c-1 not in numbers)


def thanks_referee_score(cards,chips):
    total=0; previous=None
    for c in sorted(cards):
        if previous is None or c!=previous+1: total+=c
        previous=c
    return total-chips


@dataclass(frozen=True)
class ThanksPosition:
    cards: tuple
    chips: tuple
    turn: int
    card: int | None
    pot: int = 0
    seen: tuple = ()
    remaining: int = 23
    payments: tuple = ()

    def legal(self):
        if self.card is None: return ()
        return ('TAKE','PASS') if self.chips[self.turn] else ('TAKE',)

    def play(self,name):
        if name not in self.legal(): raise ValueError('illegal take/pass')
        chips=list(self.chips); cards=list(self.cards); paid=list(self.payments or (0,)*len(chips))
        if name=='PASS':
            chips[self.turn]-=1; paid[self.turn]+=1
            return replace(self,chips=tuple(chips),pot=self.pot+1,turn=(self.turn+1)%len(chips),payments=tuple(paid))
        cards[self.turn]=tuple(sorted(cards[self.turn]+(self.card,))); chips[self.turn]+=self.pot
        # The taker starts the next card; drawing is a separate chance event.
        return replace(self,cards=tuple(cards),chips=tuple(chips),card=None,pot=0,payments=tuple(paid))

    def draw(self,card):
        if self.card is not None or not self.remaining or card in self.seen or card not in range(3,36): raise ValueError('invalid reveal')
        return replace(self,card=card,seen=tuple(sorted(self.seen+(card,))),remaining=self.remaining-1)

    def scores(self): return tuple(card_points(cards)-chips for cards,chips in zip(self.cards,self.chips))

    @classmethod
    def start(cls,players,first_card,first=0):
        if type(players) is not int or not 3<=players<=7: raise ValueError('No Thanks basic game supports 3..7 players')
        amount=11 if players<=5 else 9 if players==6 else 7
        return cls(((),)*players,(amount,)*players,first,first_card,seen=(first_card,),payments=(0,)*players)


class ThanksAdapter:
    game='no_thanks_basic'
    def __init__(self,sample_seed=0): self.sample_seed=sample_seed
    def legal(self,s): return s.legal()
    def step(self,s,name): return s.play(name)
    def terminal(self,s): return s.card is None and s.remaining==0
    def chance(self,s): return s.card is None and s.remaining>0
    def order(self,s,moves): return sorted(moves)

    def draws(self,s,k):
        available=[c for c in range(3,36) if c not in s.seen]
        # Public belief only. Prefix sampling keeps hypotheses stable across budgets.
        seed=int(digest([self.sample_seed,s.seen,s.remaining])[:16],16)
        samples=np.random.default_rng(seed).permutation(available)[:k]
        return [(1/len(samples),s.draw(int(card))) for card in samples]

    def target(self,s,name):
        if name=='PASS': return f'player-{(s.turn+1)%len(s.chips)}'
        if s.card is None: return None
        # Taking a linking card denies a visible benefit to a rival.
        gains=[(card_points(cards)-card_points(cards+(s.card,)),i) for i,cards in enumerate(s.cards) if i!=s.turn]
        gain,rival=max(gains,default=(0,-1))
        return f'player-{rival}' if gain>0 else None

    def consequence(self,before,after,side):
        old=before.scores(); new=after.scores(); gain=(old[side]-new[side])/35
        rivals=[i for i in range(len(old)) if i!=side]
        relative=((min(new[i] for i in rivals)-new[side])-(min(old[i] for i in rivals)-old[side]))/35
        security=(min(after.chips[side],8)-min(before.chips[side],8))/8
        blocked=0.
        if before.card is not None and before.card in after.cards[side]:
            blocked=max(0,max((card_points(before.cards[i])-card_points(before.cards[i]+(before.card,)) for i in rivals),default=0))/35
        winner_gain=gain
        if self.terminal(after): winner_gain=1. if new[side]==min(new) else -1.
        cost=((after.payments or (0,)*len(old))[side]-(before.payments or (0,)*len(old))[side])*.04
        return effect(float(np.clip(winner_gain,-1,1)),
                      dict(safety=float(np.clip(security,-1,1)),esteem=float(np.clip(relative,-1,1))),
                      dict(achievement=float(np.clip(winner_gain,-1,1)),security=float(np.clip(security,-1,1)),
                           power=float(np.clip(relative+blocked,-1,1)),benevolence=-max(0,blocked),
                           universalism=-max(0,blocked),self_direction=float(np.clip(gain,-1,1))),
                      cost=min(1,max(0,cost)),style=dict(neuroticism=float(np.clip(security,-1,1))))

    def context(self,s,profile,seed,tick,episode,actions,policy_state=None):
        facts=dict(card=str(s.card),pot=str(s.pot),remaining=str(s.remaining),
                   own_chips=str(s.chips[s.turn]),public_counter_ledger=str(s.chips),
                   ledger_basis='開始枚数と公開の支払い/受取りを記憶した推定。隠した手の中を見ていない',
                   unseen='未公開の9枚除外と本当の山札順は入力しない。予測は未観測集合からの標本',
                   seen=str(s.seen),estimate='小さい点数を目指す。相手の未来の判断は中立人格の仮定')
        for i,cards in enumerate(s.cards): facts[f'public_cards_{i}']=str(cards)
        result=make_context(self.game,s,profile,seed,tick,episode,actions,
                            dict(safety=float(np.clip(1-s.chips[s.turn]/8,.1,.9)),esteem=.5),facts,policy_state)
        result['objective']='カードの連番と拒否用の資源、他者の利害を考慮して最終点数を小さくする'; return result
