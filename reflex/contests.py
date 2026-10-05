"""Headless Hagetaka/finite-budget auction rules and public-only bid forecasts.

Hagetaka: 3..5 players, 1..15 bid cards, -5..-1/+1..10 prizes; duplicate
numbers eliminated, negative pot goes to smallest unique number. All-duplicate
pot carries over; zero pot uses highest unique (declared Mobius convention).
Auction: authored sealed first-price rounds, common item points, no borrowing;
highest positive bid wins, ties use a public rotating priority, only winner pays.
Neither predictor accepts real rival controller names, current bids, or deck.
"""
from collections import Counter
from dataclasses import dataclass, replace, asdict
import copy
import numpy as np
from .core import Policy, NEEDS, TRAITS, compile_batch, digest
from .examples import action, context, effect
from .planning import compress_outcomes
from .opponent_beliefs import HypothesisTracker

RULES = 'https://www.amigo-spiele.de/kartenspiele/hols-der-geier_1943_1210'
ZERO_RULE = 'https://mobius-games.co.jp/25th/rule.html'
PRIZES = tuple(range(-5,0))+tuple(range(1,11))
HYPOTHESES = ('uniform','value','frugal','high','low','pressure','response')


@dataclass(frozen=True)
class Public:
    game: str
    hands: tuple
    scores: tuple
    budgets: tuple
    prize: int
    category: str
    remaining: tuple
    carry: int = 0
    round: int = 0
    episode: int = 0
    last: tuple = ()
    last_prize: int = 0

    def __post_init__(self):
        players=len(self.scores)
        if self.game not in ('hagetaka','auction') or not 3<=players<=5:
            raise ValueError('known contest and 3..5 players required')
        if len(self.hands)!=players or len(self.budgets)!=players:
            raise ValueError('complete public roster required')
        if any(type(b) is not int or not 0<=b<=18 for b in self.budgets):
            raise ValueError('bounded nonnegative budget required')
        if any(type(v) is not int for v in self.scores+(self.prize,self.carry,self.round,self.episode)):
            raise ValueError('integer public quantities required')
        if self.round<0 or self.episode<0 or abs(self.pot)>70:
            raise ValueError('bounded round/pot required')
        if self.game=='hagetaka' and (self.prize not in PRIZES or any(
                len(set(h))!=len(h) or any(type(b) is not int or not 1<=b<=15 for b in h) for h in self.hands)):
            raise ValueError('distinct legal bid cards and prize required')
        if self.game=='auction' and (not 1<=self.prize<=30 or self.category not in ('science','culture','power')):
            raise ValueError('bounded auction item required')

    @property
    def pot(self): return self.prize+self.carry

    def legal(self,actor):
        return self.hands[actor] if self.game=='hagetaka' else tuple(range(min(8,self.budgets[actor])+1))


def public_from_record(record):
    return Public(**{**record,'hands':tuple(tuple(h) for h in record['hands']),
        'scores':tuple(record['scores']),'budgets':tuple(record['budgets']),
        'remaining':tuple(record['remaining']),'last':tuple(record['last'])})


def awarded(s,bids):
    if len(bids)!=len(s.scores) or any(type(b) is not int or b not in s.legal(i) for i,b in enumerate(bids)):
        raise ValueError('one legal simultaneous bid per player required')
    if s.game=='hagetaka':
        counts=Counter(bids); unique=[i for i,b in enumerate(bids) if counts[b]==1]
        if not unique: return None
        return (max if s.pot>=0 else min)(unique,key=lambda i:bids[i])
    if s.game!='auction': raise ValueError('unknown contest')
    top=max(bids)
    if top==0: return None
    candidates=[i for i,b in enumerate(bids) if b==top]
    priority=s.round%len(bids)
    return min(candidates,key=lambda i:(i-priority)%len(bids))


def settle(s,bids):
    winner=awarded(s,bids); points=list(s.scores); money=list(s.budgets)
    carry=s.pot if winner is None and s.game=='hagetaka' else 0
    if winner is not None:
        points[winner]+=s.pot if s.game=='hagetaka' else s.prize
        if s.game=='auction': money[winner]-=bids[winner]
    hands=tuple(tuple(x for x in h if x!=b) for h,b in zip(s.hands,bids)) if s.game=='hagetaka' else s.hands
    return replace(s,hands=hands,scores=tuple(points),budgets=tuple(money),carry=carry,
                   last=tuple(bids),last_prize=s.pot,round=s.round+1),winner


def target(s,actor,name):
    h=s.legal(actor)
    if name=='high': return max(h)
    if name=='low': return min(h)
    value=s.pot if s.game=='hagetaka' else s.prize
    nominal=round(value*1.4) if value>=0 else 15-abs(value)
    if s.game=='auction': nominal=value
    if name=='frugal': nominal=nominal-2 if s.game=='hagetaka' and value>=0 else nominal/2
    elif name=='pressure':
        if max(s.scores)-s.scores[actor]>=5:
            nominal+=2
    elif name=='response' and s.last:
        # A known public last-round bid, never the pending simultaneous choice.
        others=[b for i,b in enumerate(s.last) if i!=actor]
        expected=float(np.median(others))
        if s.game=='hagetaka' and value<0: nominal=max(nominal,expected+1)
        else: nominal=expected+1
    return min(h,key=lambda b:(abs(b-nominal),b))


def hypotheses(s,actor):
    h=s.legal(actor)
    return {name:{str(b):(1/len(h) if name=='uniform' else float(b==target(s,actor,name))) for b in h}
            for name in HYPOTHESES}


class PublicBeliefs:
    """Isolated campaign observer; reveal only after EVERY player has chosen."""
    def __init__(self,players,viewer):
        if not 3<=players<=5 or not 0<=viewer<players: raise ValueError('valid public observer required')
        self.viewer=viewer; self.trackers=tuple(HypothesisTracker(HYPOTHESES) for _ in range(players))
        self.last=(-1,-1); self.game=None

    def _check(self,s):
        if len(s.scores)!=len(self.trackers) or (self.game is not None and self.game!=s.game):
            raise ValueError('observer belongs to one game and roster')
        self.game=s.game

    def forecast(self,s,actor):
        self._check(s)
        return {int(k):v for k,v in self.trackers[actor].snapshot().predict(hypotheses(s,actor)).items()}

    def trust(self):
        return max(t.snapshot().confidence for i,t in enumerate(self.trackers) if i!=self.viewer)

    def reveal(self,s,bids):
        self._check(s)
        awarded(s,bids)
        stamp=(s.episode,s.round)
        if stamp<=self.last: raise ValueError('public observations must advance')
        rows=[]
        for actor,b in enumerate(bids):
            if actor!=self.viewer:
                rows.append(self.trackers[actor].observe(hypotheses(s,actor),str(b),f'{s.game}:{s.episode}:{s.round}'))
        self.last=stamp
        return rows


def winners_vector(s,matrix):
    """Vectorized current-round resolution, independent scalar referee below."""
    if s.game=='hagetaka':
        unique=(matrix[:,:,None]==matrix[:,None,:]).sum(axis=2)==1
        if s.pot>=0: result=np.where(unique,matrix,-1).argmax(axis=1)
        else: result=np.where(unique,matrix,99).argmin(axis=1)
        return np.where(unique.any(axis=1),result,-1)
    priority=(np.arange(len(s.scores))-s.round%len(s.scores))%len(s.scores)
    top=matrix.max(axis=1)
    choice=np.where(matrix==top[:,None],priority[None,:],99).argmin(axis=1)
    return np.where(top>0,choice,-1)


def capital_price(s,actor):
    """Authored opportunity-cost proxy, NOT a future bidding equilibrium.

    Unseen items are a known multiset, never the true future order. Use equal
    shares of remaining value divided by cash, with bounded price and terminal
    salvage floor. Its fairness/share assumption is deliberately explicit.
    """
    return min(2.,max(.25,sum(s.remaining)/(len(s.scores)*max(1,s.budgets[actor]))))


def evaluate(s,viewer,bid,joint,capital_pricing=True):
    matrix=joint.copy(); matrix[:,viewer]=bid
    winning=winners_vector(s,matrix)
    gain=np.where(winning==viewer,s.pot if s.game=='hagetaka' else s.prize,0.)
    rival_gain=np.where((winning>=0)&(winning!=viewer),s.pot if s.game=='hagetaka' else s.prize,0.)
    paid=np.where(winning==viewer,bid if s.game=='auction' else 0,0.)
    # Rival payment improves their own terminal wealth cost; price is observed
    # within each hypothesis. It is not a reward for attacking them.
    rival_paid=np.where((winning>=0)&(winning!=viewer),matrix[np.arange(len(matrix)),np.maximum(winning,0)] if s.game=='auction' else 0,0.)
    scale=70 if s.game=='hagetaka' else 30
    own_price=(capital_price(s,viewer) if capital_pricing else .25) if s.game=='auction' else 0.
    prices=np.array([capital_price(s,i) if capital_pricing else .25 for i in range(len(s.scores))]) if s.game=='auction' else np.zeros(len(s.scores))
    rival_price=prices[np.maximum(winning,0)]
    objective=(gain-rival_gain-own_price*paid+rival_price*rival_paid)/scale
    if s.game=='hagetaka':
        commitment=bid/15*len(s.remaining)/15
        safety=-np.maximum(-gain,0)/70-.12*commitment
        financial_cost=np.zeros(len(matrix))
    else:
        commitment=bid/max(1,s.budgets[viewer])
        safety=-paid/18
        financial_cost=paid/48
    rows=[]
    for i in range(len(matrix)):
        good=float((gain[i]-(own_price*paid[i] if capital_pricing else 0))/scale)
        acquisition=float(gain[i]/scale)
        safe=float(np.clip(safety[i],-1,1))
        values={'achievement':good,'power':float((gain[i]-rival_gain[i])/scale),
                'security':safe}
        if s.game=='auction':
            # Power-category alignment describes own acquisition instead of the
            # general relative score proxy for this particular authored item.
            values[{'science':'self_direction','culture':'tradition','power':'power'}[s.category]]=acquisition
        rows.append(effect(float(np.clip(objective[i],-1,1)),
            needs={'esteem':good,'safety':safe},values=values,
            style={'conscientiousness':float(-.05*commitment),'openness':float(.05*commitment)},
            cost=float(financial_cost[i]),p=1/len(matrix)))
    return compress_outcomes(rows), dict(objective_mean=float(np.mean(objective)),
        objective_se=float(np.std(objective,ddof=1)/np.sqrt(len(matrix))),
        own_gain_mean=float(np.mean(gain)),payment_mean=float(np.mean(paid)))


def make_context(s,viewer,profile,seed,tick,memory=None,beliefs=None,samples=32,capital_pricing=True):
    if type(samples) is not int or not 8<=samples<=128: raise ValueError('8..128 samples required')
    names=s.legal(viewer)
    if len(names)*samples>2048: raise ValueError('current-round prediction budget exceeded')
    # Common random variates across all roots AND uniform/learned models.
    rng=np.random.default_rng(int(digest([s.game,seed,s.episode,s.round,viewer,'forecast'])[:16],16))
    u=rng.random((samples,len(s.scores))); joint=np.zeros_like(u,dtype=int)
    for actor in range(len(s.scores)):
        if actor==viewer: continue
        dist=({b:1/len(s.legal(actor)) for b in s.legal(actor)} if beliefs is None else beliefs.forecast(s,actor))
        h=tuple(sorted(dist)); cumulative=np.cumsum([dist[b] for b in h])
        joint[:,actor]=np.asarray(h)[np.minimum(np.searchsorted(cumulative,u[:,actor],side='right'),len(h)-1)]
    choices=[]; estimates={}
    for bid in names:
        outcomes,summary=evaluate(s,viewer,bid,joint,capital_pricing)
        choices.append(action('BID:'+str(bid),*outcomes)); estimates['BID:'+str(bid)]=summary
    needs={'esteem':.3,'safety':(.15 if s.game=='hagetaka' else max(.1,1-s.budgets[viewer]/18))}
    c=context(f'campaign-{seed}-game-{s.episode}',choices,needs,profile['values'],dict(zip(TRAITS,profile['traits'])),mode=None)
    for n in NEEDS:
        if n not in needs: c['needs'][n]=dict(supported=False,enabled=False,deficit=None)
    c.update(scope=dict(game=s.game+'-contest-v1',episode=f'{seed}-{s.episode}',npc=f'player-{viewer}'),
        seed=seed,tick=tick,objective='最終得点で先着する。auctionは残資金の0.25を最終得点に加える',
        facts=dict(prize=str(s.prize),category=s.category,carry=str(s.carry),scores=str(s.scores),
            budgets=str(s.budgets),unseen_remaining_multiset=str(s.remaining),last_public_bids=str(s.last),
            forecast='current-round independent bid hypotheses; future deck/current rival bids are absent',
            valuation='current margin, equal-share capital price and commitment are proxies, not terminal win EV',
            future_capital_pricing=str(bool(capital_pricing))))
    c['facts'].update({f'public_unused_cards_{i}':str(h) for i,h in enumerate(s.hands)})
    if memory is not None: c['state']=copy.deepcopy(memory)
    return c,dict(samples=samples,root_candidates=len(names),joint_resolutions=len(names)*samples,estimates=estimates)


def decide(s,viewer,profile,seed,tick,memory=None,beliefs=None,mode='gated',samples=32,capital_pricing=True):
    if mode not in ('uniform','learned','gated'): raise ValueError('unknown reading mode')
    base,base_stats=make_context(s,viewer,profile,seed,tick,memory,samples=samples,capital_pricing=capital_pricing)
    baseline=Policy().choose(base)
    stats=dict(reason='uniform',reading_applied=False,action_changed=False,
        baseline_action=baseline['action_id'],joint_resolutions=base_stats['joint_resolutions'],
        baseline=base_stats,trust=0 if beliefs is None else beliefs.trust())
    if mode=='uniform' or beliefs is None or (mode=='gated' and beliefs.trust()<.15):
        stats['reason']='no_reliable_public_evidence' if mode=='gated' else 'uniform'
        return base,baseline,stats
    learned,learned_stats=make_context(s,viewer,profile,seed,tick,memory,beliefs,samples,capital_pricing)
    batch=compile_batch([learned]); numeric=Policy().decide(batch); d=numeric.records(batch)[0]
    old=batch.ids[0].index(baseline['action_id']); new=batch.ids[0].index(d['action_id'])
    gain=float(numeric.scores[0,new]-numeric.scores[0,old])
    old_est=learned_stats['estimates'][baseline['action_id']]
    new_est=learned_stats['estimates'][d['action_id']]
    margin=.025+.65*(old_est['objective_se']+new_est['objective_se'])
    totals=np.array(s.scores,dtype=float)+(np.array(s.budgets)*.25 if s.game=='auction' else 0)
    ahead=totals[viewer]>max(x for i,x in enumerate(totals) if i!=viewer)
    maintained=ahead and old_est['objective_mean']-1.65*old_est['objective_se']>=0
    same=d['action_id']==baseline['action_id']
    applied=mode=='learned' or same or (not maintained and gain>margin)
    reason=('always_use_learned' if mode=='learned' else 'same_action' if same else
        'one_round_advantage_maintained_under_model' if maintained else
        'material_persona_gain' if applied else 'gain_below_uncertainty_margin')
    stats.update(reason=reason,reading_applied=applied,action_changed=applied and not same,
        learned_action=d['action_id'],persona_gain=gain,required_gain=margin,
        currently_ahead=bool(ahead),joint_resolutions=stats['joint_resolutions']+learned_stats['joint_resolutions'],
        learned=learned_stats)
    return (learned,d,stats) if applied else (base,baseline,stats)
