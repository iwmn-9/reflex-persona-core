"""Public-observation bridge for an explicitly closed-hand riichi baseline.

The external engine owns legality, yaku, furiten and payments. This adapter
supplies GAME-SPECIFIC hand-progress and public exposure proxies to the
unchanged numerical personality core. Proxies are not win probabilities.
The optional lookahead models one own draw, not a four-player continuation.
"""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import copy
import numpy as np
from .core import Policy, TRAITS, NEEDS, compile_batch, digest
from .examples import context, action, effect

ENGINE_VERSION='0.4.10'
SUPPORTED_TYPES=frozenset(('Discard','Riichi','Ron','Tsumo','Pass'))


def engine():
    try:
        import riichienv
    except ImportError as exc:
        raise RuntimeError('Install optional requirements-mahjong.txt in a separate environment') from exc
    return riichienv


def action_type(a):return str(a.action_type).split('.')[-1].capitalize()


def action_id(a):
    # Chi variants can share a claimed tile but consume different own tiles.
    return action_type(a)+':'+str(a.tile)+':'+','.join(str(t) for t in a.consume_tiles)


@dataclass(frozen=True)
class PublicView:
    player: int
    hand: tuple
    discards: tuple
    dora: tuple
    scores: tuple
    riichi: tuple
    oya: int
    round_wind: int
    honba: int
    sticks: int

    @classmethod
    def from_observation(cls,obs):
        d=obs.to_dict();p=d['player_id']
        if any(d['melds']):raise ValueError('This first protocol supports closed hands only')
        # Deliberately project an allowlist; other hands/events never enter.
        return cls(p,tuple(d['hands'][p]),tuple(tuple(r) for r in d['discards']),
                   tuple(d['dora_indicators']),tuple(d['scores']),tuple(d['riichi_declared']),
                   d['oya'],d['round_wind'],d['honba'],d['riichi_sticks'])

    def unseen(self):
        visible=list(self.hand)+list(self.dora)+[t for r in self.discards for t in r]
        if len(visible)!=len(set(visible)) or any(not 0<=t<136 for t in visible):
            raise ValueError('Visible physical tiles must be unique and valid')
        return tuple(t for t in range(136) if t not in set(visible))

    def exposure(self,tile):
        """Fraction of declared opponents for whom the tile is NOT genbutsu.

        This is a bounded exposure index, NOT P(ron). Unmarked danger from
        non-riichi opponents, suji, kabe and value inference are not modelled.
        """
        threats=[i for i,r in enumerate(self.riichi) if i!=self.player and r]
        if not threats:return 0.
        return sum(tile//4 not in {x//4 for x in self.discards[i]} for i in threats)/len(threats)


@lru_cache(maxsize=65536)
def _shanten(types):
    counts=Counter();physical=[]
    for t in types:
        physical.append(t*4+counts[t]);counts[t]+=1
        if counts[t]>4:raise ValueError('More than four copies of a tile type')
    return engine().calculate_shanten(physical)


def shanten(hand):return _shanten(tuple(sorted(t//4 for t in hand)))


def hand_features(hand,unseen):
    s=shanten(hand);by_type=Counter(t//4 for t in unseen);have=Counter(t//4 for t in hand)
    effective=[]
    for t,n in by_type.items():
        if have[t]<4 and shanten(tuple(hand)+(t*4+have[t],))<s:effective.append((t,n))
    ukeire=sum(n for t,n in effective)
    # Explicit rule-derived progress proxy; no trained or universal evaluator.
    # One shanten step is .15, larger than the FULL .10 ukeire range.
    # Broad draws cannot silently outweigh losing a step of hand progress.
    progress=.90*(6-s)/6+.10*ukeire/max(1,len(unseen))
    return dict(shanten=s,ukeire=ukeire,effective_types=[t for t,n in effective],progress=float(progress))


def remove_tile(hand,tile):
    row=list(hand);row.remove(tile);return tuple(row)


def best_next_progress(hand,draw):
    row=tuple(hand)+(draw,)
    if shanten(row)==-1:return 1. # Complete shape; yaku/furiten NOT assessed here.
    # Tenpai without completion stays at .90, below an actual complete shape.
    return max(.90*(6-shanten(remove_tile(row,t)))/6 for t in set(row))


def candidates(obs,lookahead_samples=0,seed=0,tick=0):
    if type(lookahead_samples) is not int or not 0<=lookahead_samples<=64:
        raise ValueError('lookahead_samples must be 0..64')
    view=PublicView.from_observation(obs);unseen=view.unseen();legal=obs.legal_actions()
    if len(view.hand) not in (13,14):raise ValueError('Closed hand of 13 or 14 tiles required')
    if len({action_id(a) for a in legal})!=len(legal):raise ValueError('Ambiguous legal action IDs')
    rng=np.random.default_rng(int(digest([seed,tick,view.player,'own-draw'])[:16],16))
    draws=tuple(int(t) for t in rng.choice(unseen,size=lookahead_samples,replace=True)) if lookahead_samples else ()
    rows=[];discard_features={}
    for a in legal:
        if action_type(a)=='Discard':
            hand=remove_tile(view.hand,a.tile);f=hand_features(hand,unseen)
            futures=[best_next_progress(hand,t) for t in draws]
            discard_features[action_id(a)]=(f,futures)
    for a in legal:
        typ=action_type(a);name=action_id(a);f=None;future=[];bonus=0.
        if typ=='Discard':
            f,future=discard_features[name];progress=f['progress'];exposure=view.exposure(a.tile)
        elif typ in ('Ron','Tsumo'):
            # These are engine-confirmed legal wins, unlike complete-shape proxies.
            progress=1.;exposure=0.
        elif typ=='Riichi':
            # The engine checks legal riichi. Same best available discard proxy
            # plus a small declared-yaku bonus, NOT an EV estimate of the deposit.
            ready=[(f,futures) for f,futures in discard_features.values() if f['shanten']==0]
            if not ready:raise ValueError('Engine legal riichi needs a tenpai discard')
            def forecast(row):
                q,x=row
                return .7*q['progress']+.3*float(np.mean(x)) if x else q['progress']
            q,future=max(ready,key=forecast)
            progress=q['progress'];bonus=.015;exposure=.1 # commitment cost assumption
        else:progress=0.;exposure=0.
        if future:
            bucket=Counter(future)
            outcomes=[effect(min(.7*progress+.3*p+bonus,1.),values={'achievement':min(.7*progress+.3*p+bonus,1.),'security':-exposure},
                             cost=.02*exposure,p=n/len(future)) for p,n in sorted(bucket.items())]
        else:outcomes=[effect(min(progress+bonus,1.),values={'achievement':min(progress+bonus,1.),'security':-exposure},cost=.02*exposure)]
        rows.append(dict(id=name,type=typ,tile=a.tile,supported=typ in SUPPORTED_TYPES,
                         features=f,exposure=exposure,estimated_progress=float(np.mean([o['objective'] for o in outcomes])) if not future else
                         float(sum(o['p']*o['objective'] for o in outcomes)),outcomes=outcomes))
    if not any(r['supported'] for r in rows):raise ValueError('No action in closed-hand protocol')
    return view,rows


def make_context(view,rows,profile,seed,tick,episode,memory=None):
    c=context(episode,[action(r['id'],*r['outcomes'],legal=r['supported']) for r in rows],
              {},profile['values'],dict(zip(TRAITS,profile['traits'])),mode=None)
    for k in NEEDS:c['needs'][k]=dict(supported=False,enabled=False,deficit=None)
    c.update(scope=dict(game='riichi_closed_baseline',episode=episode,npc=f'player-{view.player}'),seed=seed,tick=tick,
             objective='閉じた手の進行と公開された危険への露出を、本人の価値づけで比較する。終局EVではない',
             facts=dict(evaluation='麻雀側のシャンテン/受入れ代理評価。任意の先読みは自分の次の1ツモのみ',
                        security='宣言相手の現物か否かの露出指数。放銃確率ではない',
                        protocol='鳴き/カン/途中流局の選択は今回対象外。合法な和了/立直/打牌/パスを比較',
                        unsupported='欲求全領域、その他の価値、他者傾向の学習、順位目的、数手の方針保持',
                        public=digest(view.__dict__)))
    if memory is not None:c['state']=copy.deepcopy(memory)
    return c


def choose(obs,profile,seed,tick,episode,memory=None,lookahead_samples=0):
    view,rows=candidates(obs,lookahead_samples,seed,tick)
    c=make_context(view,rows,profile,seed,tick,episode,memory)
    b=compile_batch([c]);s=Policy().decide(b,stochastic=False);d=s.records(b)[0]
    selected=next(r for r in rows if r['id']==d['action_id'])
    a=next(a for a in obs.legal_actions() if action_id(a)==d['action_id'])
    stats=dict(features=rows,selected=selected,lookahead_samples=lookahead_samples,
               selected_shanten=None if selected['features'] is None else selected['features']['shanten'],
               threatened=any(view.riichi[i] for i in range(4) if i!=view.player),
               assumption='uniform unseen own draw; no opponent or four-player continuation',
               public_hash=digest(view.__dict__))
    return a,c,d,stats


def reference(obs,seed,tick,kind='efficiency'):
    """Closed-hand mechanical minimum / shanten reference; no persona core."""
    view,rows=candidates(obs)
    allowed=[r for r in rows if r['supported']]
    # Every comparison controller takes offered wins and prefers available riichi.
    wins=[r for r in allowed if r['type'] in ('Ron','Tsumo')]
    riichi=[r for r in allowed if r['type']=='Riichi']
    discards=[r for r in allowed if r['type']=='Discard']
    options=wins or riichi or discards or allowed
    if not wins and not riichi and discards and kind=='efficiency':
        best=min((r['features']['shanten'],-r['features']['ukeire']) for r in discards)
        options=[r for r in discards if (r['features']['shanten'],-r['features']['ukeire'])==best]
    elif kind not in ('random','efficiency'):raise ValueError('Unknown reference')
    rng=np.random.default_rng(int(digest([seed,tick,view.player,'reference'])[:16],16))
    selected=options[int(rng.integers(len(options)))]['id']
    return next(a for a in obs.legal_actions() if action_id(a)==selected)
