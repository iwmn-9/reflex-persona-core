"""Public-bid Goofspiel bridge: simultaneous choices, no shared-policy changes.

OpenSpiel variant: 2..10 players, public used cards, known ascending/descending
prizes, unique highest bid wins; a tied highest bid discards the prize. Score
and fractional winner credit are separate game objectives. Forecasts assume
uniform independent bids unless explicitly configured otherwise.
"""
from dataclasses import dataclass, replace
from itertools import product
import copy
import numpy as np
from .core import Policy, TRAITS, NEEDS
from .examples import action, context, effect
from .board_models import neutral
from .planning import compress_outcomes
from .monte_carlo import evaluate_actions, TerminalEvaluation

RULES='https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/games/goofspiel/goofspiel.cc'


@dataclass(frozen=True)
class Position:
    hands: tuple
    scores: tuple
    prizes: tuple
    round: int = 0
    discarded: int = 0

    @classmethod
    def start(cls,players=4,cards=13,order='ascending'):
        if type(players) is not int or not 2<=players<=10: raise ValueError('players must be 2..10')
        if type(cards) is not int or not 2<=cards<=13: raise ValueError('cards must be 2..13')
        if order not in ('ascending','descending'): raise ValueError('known prize order required')
        deck=tuple(range(1,cards+1))
        return cls((deck,)*players,(0,)*players,deck if order=='ascending' else deck[::-1])

    @property
    def terminal(self): return self.round==len(self.prizes)

    def play(self,bids):
        if self.terminal: raise ValueError('terminal game')
        if len(bids)!=len(self.hands) or any(type(b) is not int or b not in h for b,h in zip(bids,self.hands)):
            raise ValueError('one unused bid per player required')
        high=max(bids); winners=[i for i,b in enumerate(bids) if b==high]
        points=self.prizes[self.round]; scores=list(self.scores)
        if len(winners)==1: scores[winners[0]]+=points
        return replace(self,hands=tuple(tuple(c for c in h if c!=b) for h,b in zip(self.hands,bids)),
                       scores=tuple(scores),round=self.round+1,
                       discarded=self.discarded+(points if len(winners)>1 else 0))

    def share(self,viewer):
        winners=[i for i,p in enumerate(self.scores) if p==max(self.scores)]
        return (1/len(winners) if viewer in winners else 0.),winners


def referee(before,bids,after):
    """Independent list-based transition/conservation checks, no Position.play."""
    hands=[list(h) for h in before.hands]; scores=list(before.scores)
    for i,b in enumerate(bids): hands[i].remove(b)
    ranking=sorted(enumerate(bids),key=lambda x:x[1],reverse=True)
    unique=ranking[0][1]>ranking[1][1]; prize=before.prizes[before.round]
    if unique: scores[ranking[0][0]]+=prize
    assert after.hands==tuple(tuple(h) for h in hands) and after.scores==tuple(scores)
    assert after.round==before.round+1 and after.prizes==before.prizes
    assert after.discarded==before.discarded+(0 if unique else prize)
    assert sum(after.scores)+after.discarded+sum(after.prizes[after.round:])==sum(after.prizes)
    assert all(len(h)==len(after.prizes)-after.round for h in after.hands)


def reserve_bid(s,actor,rng):
    """Declared baseline: reserve high cards for high prizes; no sealed-bid access."""
    h=s.hands[actor]; prize=s.prizes[s.round]
    distances=[abs(c-prize) for c in h]; best=min(distances)
    options=[c for c,d in zip(h,distances) if d<=best+1]
    return options[int(rng.integers(len(options)))]


def consequence(base,scores,viewer,bid,goal,terminal=False):
    """Factual score goal; other axes are explicit game-authored proxies.

    Commitment is ONLY the root card, not all cards used in the simulated future.
    It represents lost future bidding flexibility, not a fictitious point fee.
    """
    remaining=max(1,sum(base.prizes[base.round:])); delta=scores[viewer]-base.scores[viewer]
    opponents=[p for i,p in enumerate(scores) if i!=viewer]
    old_opp=max(p for i,p in enumerate(base.scores) if i!=viewer)
    margin=(scores[viewer]-max(opponents)-(base.scores[viewer]-old_opp))/remaining
    if goal=='score': objective=delta/remaining
    elif goal=='win_share':
        if terminal:
            winners=[i for i,p in enumerate(scores) if p==max(scores)]
            objective=2*(1/len(winners) if viewer in winners else 0)-1
        else: objective=margin  # a score-margin proxy, never labelled win probability
    else: raise ValueError('unknown objective')
    future=sum(base.prizes[base.round+1:])/remaining
    commitment=bid/max(base.prizes)*future
    other_gain=sum(p-old for i,(p,old) in enumerate(zip(scores,base.scores)) if i!=viewer)/remaining
    return effect(float(objective),needs={'esteem':float(delta/remaining),'safety':float(-commitment*.15)},
        values={'achievement':float(objective),'power':float(margin),'security':float(-commitment),
                'benevolence':float(other_gain)},
        style={'openness':float(.2*commitment),'conscientiousness':float(-.2*commitment)})


def immediate_outcomes(s,viewer,bid,goal,beliefs=None):
    """Exact current-prize distribution conditional on independent bid models."""
    hands=s.hands; others=[i for i in range(len(hands)) if i!=viewer]
    models={i:({c:1/len(hands[i]) for c in hands[i]} if beliefs is None else
               {int(a.split(':')[1]):p for a,p in beliefs.probabilities(s,i).items()}) for i in others}
    probabilities={viewer:float(np.prod([sum(p for c,p in models[i].items() if c<bid) for i in others]))}
    for actor in others:
        probabilities[actor]=sum(models[actor][b]*float(np.prod([
            sum(p for c,p in models[i].items() if c<b) for i in others if i!=actor]))
            for b in hands[actor] if b>bid)
    probabilities[None]=max(0.,1-sum(probabilities.values()))
    rows=[]
    for winner,p in probabilities.items():
        if p<=0: continue
        scores=list(s.scores)
        if winner is not None: scores[winner]+=s.prizes[s.round]
        row=consequence(s,scores,viewer,bid,goal,terminal=s.round==len(s.prizes)-1)
        row['p']=p; rows.append(row)
    return compress_outcomes(rows)


def make_context(s,viewer,profile,goal,seed,tick,episode,memory=None,outcomes=None,beliefs=None):
    if s.terminal or not 0<=viewer<len(s.hands): raise ValueError('live viewer required')
    if goal not in ('score','win_share'): raise ValueError('unknown objective')
    actions=[action(f'BID:{b}',*(outcomes[f'BID:{b}'] if outcomes is not None else immediate_outcomes(s,viewer,b,goal,beliefs)))
             for b in s.hands[viewer]]
    c=context(episode,actions,{'esteem':.3,'safety':.15},profile['values'],dict(zip(TRAITS,profile['traits'])),mode=None)
    for key in NEEDS:
        if key not in ('esteem','safety'): c['needs'][key]=dict(supported=False,enabled=False,deficit=None)
    c.update(scope=dict(game='goofspiel',episode=episode,npc=f'player-{viewer}'),seed=seed,tick=tick,
        objective='自分の得点を稼ぐ' if goal=='score' else '最終優勝の持分を高める',
        facts=dict(scores=str(list(s.scores)),
                   current_prize=str(s.prizes[s.round]),known_future_prizes=str(list(s.prizes[s.round+1:])),
                   tie_rule='最高入札が同額なら得点カードを破棄',goal=goal,
                   forecast='今回の得点。相手は独立一様入札。win_shareの反射評価は点差代理値',
                   commitment='今回使う札の将来の入札余力の喪失。点数罰ではない'))
    c['facts'].update({f'public_hand_{i}':str(list(h)) for i,h in enumerate(s.hands)})
    if beliefs is not None:
        c['facts']['opponent_assumption']=f'過去の公開入札から更新した{len(beliefs.players[viewer].names)}行動仮説の混合。独立仮説で相関/交渉は未推定'
        c['facts']['forecast']='今回の得点。相手は公開履歴の混合分布。win_shareの反射評価は点差代理値'
        for i,p in enumerate(beliefs.players):
            if i==viewer: continue
            c['facts'][f'belief_{i}']=f'n={p.observations}; trust={p.confidence:.3f}; '+','.join(f'{k}:{w:.3f}' for k,w in zip(p.names,p.weights))
    if memory is not None: c['state']=copy.deepcopy(memory)
    return c


@dataclass(frozen=True)
class Pending:
    position: Position
    own_bid: int | None = None


class RolloutModel:
    def __init__(self,base,viewer,profile,goal,policy,seed,tick,episode,memory=None,beliefs=None,common_random=False):
        self.base=base; self.viewer=viewer; self.profile=copy.deepcopy(profile); self.goal=goal
        self.policy=policy; self.seed=seed; self.tick=tick; self.episode=episode; self.memory=copy.deepcopy(memory)
        self.beliefs=beliefs; self.distribution_cache={}; self.common_random=common_random

    def begin_trial(self): self.memories={self.viewer:copy.deepcopy(self.memory)}
    def terminal(self,node): return node.position.terminal
    def chance(self,node): return True

    def joint_bids(self,node,rng):
        s=node.position; bids=[]
        # Every actor receives the same PRE-REVEAL public state. No actor gets
        # the root's fixed bid or another actor's current commitment.
        for actor,h in enumerate(s.hands):
            if actor==self.viewer and node.own_bid is not None: bid=node.own_bid
            elif actor!=self.viewer and self.beliefs is not None and self.beliefs.players[actor].confidence>0:
                key=(actor,h,s.prizes[s.round],s.scores)
                if key not in self.distribution_cache:
                    probs=self.beliefs.probabilities(s,actor); running=0.; cumulative=[]
                    for b in h:
                        running+=probs[f'BID:{b}']; cumulative.append(running)
                    self.distribution_cache[key]=tuple(cumulative)
                u=float(rng.random()); index=min(sum(u>=p for p in self.distribution_cache[key]),len(h)-1); bid=h[index]
            elif self.policy=='random':
                bid=h[min(int(rng.random()*len(h)),len(h)-1)] if self.common_random else h[int(rng.integers(len(h)))]
            elif self.policy=='tactical': bid=reserve_bid(s,actor,rng)
            else:
                profile=self.profile if actor==self.viewer else neutral()
                c=make_context(s,actor,profile,self.goal,self.seed,self.tick+s.round-self.base.round,
                               self.episode,self.memories.get(actor))
                d=Policy().choose(c,stochastic=False); self.memories[actor]=d['next_state']
                bid=int(d['action_id'].split(':')[1])
            bids.append(bid)
        return tuple(bids)

    def evaluate(self,node):
        s=node.position
        if not s.terminal: raise ValueError('terminal reward required')
        share,winners=s.share(self.viewer)
        row=consequence(self.base,s.scores,self.viewer,self.root_bid,self.goal,terminal=True)
        return TerminalEvaluation(row,float(share),self.viewer in winners,len(winners)==len(s.hands),float(s.scores[self.viewer]))

    # The generic engine calls begin_trial before each root; root_bid is set by
    # the first joint-round sample, never passed to opponents' contexts.
    def sample(self,node,rng):
        if node.own_bid is not None: self.root_bid=node.own_bid
        return Pending(node.position.play(self.joint_bids(node,rng)))


def observe(s,viewer,profile,goal,seed,tick,episode,memory=None,budget=None,beliefs=None,common_random=False):
    if beliefs is not None and (beliefs.viewer!=viewer or len(beliefs.players)!=len(s.hands)):
        raise ValueError('belief snapshot belongs to a different public observer')
    if budget is None:
        return make_context(s,viewer,profile,goal,seed,tick,episode,memory,beliefs=beliefs),dict(used=False,completed_samples=0,method='reflex',goal=goal)
    model=RolloutModel(s,viewer,profile,goal,budget.rollout_policy,seed,tick,episode,memory,beliefs,common_random)
    roots={f'BID:{b}':Pending(s,b) for b in s.hands[viewer]}
    packed,stats=evaluate_actions(roots,model,budget,[seed,'goofspiel',episode,f'player-{viewer}',tick])
    c=make_context(s,viewer,profile,goal,seed,tick,episode,memory,packed if stats['used'] else None,beliefs)
    if stats['used']:
        c['facts']['forecast']=f'終局まで各候補{stats["completed_samples"]}標本。未来の自分={budget.rollout_policy}。相手='+('観測仮説の混合' if beliefs else budget.rollout_policy)
    stats.update(goal=goal,transition_unit='one joint simultaneous round',opponent_assumption=budget.rollout_policy,
                 own_continuation_assumption=budget.rollout_policy,method='monte_carlo')
    stats['sampling_scheme']='common inverse-CDF uniforms' if common_random else 'default RNG draws'
    if beliefs is not None:
        stats.update(opponent_assumption='discounted public-bid hypothesis mixture',beliefs=beliefs.record(),
                     virtual_learning='frozen at root; fictional outcomes never update real trackers')
    return c,stats


def exact_two_rounds(s,viewer):
    """Finite conditional value targets, NOT human-approved action labels.

    Enumerate independent uniform opponent bids now; the final single-card
    round is forced for everyone. Both point gain and winner share are factual
    conditional expectations, with no personality/heuristic value weighting.
    """
    if s.terminal or any(len(h)!=2 for h in s.hands): raise ValueError('exactly two rounds required')
    others=[i for i in range(len(s.hands)) if i!=viewer]; targets={}
    for own in s.hands[viewer]:
        scores=[]; shares=[]; leaves=0
        for joint in product(*(s.hands[i] for i in others)):
            bids=[own]*len(s.hands)
            for i,b in zip(others,joint): bids[i]=b
            child=s.play(tuple(bids)); final=child.play(tuple(h[0] for h in child.hands))
            scores.append(final.scores[viewer]); shares.append(final.share(viewer)[0]); leaves+=1
        targets[f'BID:{own}']=dict(mean_score=float(np.mean(scores)),mean_gain=float(np.mean(scores)-s.scores[viewer]),
                                    win_share=float(np.mean(shares)),enumerated_joint_choices=leaves)
    return targets
