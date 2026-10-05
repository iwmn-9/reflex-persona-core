"""Terminal-only shared baseline: legal rules + declared reward semantics.

No positional features, action-name strategies or game-authored personality
effects are called. Rule hooks still require a simulator; this is not learning
an arbitrary rulebook. Uniform continuations approximate unmodelled players.
"""
from dataclasses import dataclass
import copy
import numpy as np
from .core import Policy,TRAITS,NEEDS,compile_batch
from .examples import context,action,effect
from .monte_carlo import TerminalEvaluation,evaluate_actions,RolloutBudget
from .goofspiel import Pending

MODES=('rules_only','rules_persona','reward_persona')


@dataclass(frozen=True)
class Rewards:
    credits: tuple
    returns: tuple
    scores: tuple

    def __post_init__(self):
        if len(self.credits)<2 or len(self.credits)!=len(self.returns) or len(self.scores)!=len(self.credits):
            raise ValueError('matching per-player terminal rewards required')
        if any(not np.isfinite(x) or not 0<=x<=1 for x in self.credits) or abs(sum(self.credits)-1)>1e-10:
            raise ValueError('fractional winner credits must sum to one')
        if any(not np.isfinite(x) or not -1<=x<=1 for x in self.returns):
            raise ValueError('declared returns must be bounded [-1,1]')


def winner_credits(scores,lower=False):
    best=min(scores) if lower else max(scores)
    winners=[i for i,x in enumerate(scores) if x==best]
    return tuple(1/len(winners) if i in winners else 0. for i in range(len(scores)))


class ConnectRules:
    game='connect_four'; players=2; simultaneous=False
    def terminal(self,s):return not s.legal()
    def chance(self,s):return False
    def actor(self,s):return s.turn
    def legal(self,s,viewer=None):return s.legal()
    def root(self,s,a,viewer):return s.play(a)
    def step(self,s,a):return s.play(a)
    def rewards(self,s):
        if not self.terminal(s):raise ValueError('terminal rewards only')
        w=s.winner(); credits=(.5,.5) if w is None else tuple(float(i==w) for i in range(2))
        return Rewards(credits,tuple(2*c-1 for c in credits),credits)
    def observation(self,s):return dict(discs=list(s.discs),heights=list(s.heights),turn=s.turn)


class GoofRules:
    game='goofspiel'; simultaneous=True
    def __init__(self,players=4):self.players=players
    def terminal(self,s):return s.position.terminal
    def chance(self,s):return True
    def actor(self,s):raise ValueError('simultaneous game has no next single actor')
    def legal(self,s,viewer=None):
        if viewer is None:raise ValueError('simultaneous viewer required')
        return tuple(f'BID:{b}' for b in s.position.hands[viewer])
    def root(self,s,a,viewer):return Pending(s.position,int(a.split(':')[1]))
    def sample(self,s,rng,viewer):
        bids=[]
        for i,h in enumerate(s.position.hands):
            u=float(rng.random()) # same number of draws, including the fixed root
            bids.append(s.own_bid if i==viewer and s.own_bid is not None else h[min(int(u*len(h)),len(h)-1)])
        return Pending(s.position.play(tuple(bids)))
    def rewards(self,s):
        p=s.position
        if not p.terminal:raise ValueError('terminal rewards only')
        total=sum(p.prizes)
        return Rewards(winner_credits(p.scores),tuple(2*x/total-1 for x in p.scores),p.scores)
    def observation(self,s):
        p=s.position
        return dict(hands=[list(h) for h in p.hands],scores=list(p.scores),prizes=list(p.prizes),round=p.round,discarded=p.discarded)


class ThanksRules:
    game='no_thanks_basic'; simultaneous=False
    def __init__(self,players=3):
        self.players=players
        self.total_chips=players*(11 if players<=5 else 9 if players==6 else 7)
    def terminal(self,s):return s.card is None and s.remaining==0
    def chance(self,s):return s.card is None and s.remaining>0
    def actor(self,s):return s.turn
    def legal(self,s,viewer=None):return s.legal()
    def root(self,s,a,viewer):return s.play(a)
    def step(self,s,a):return s.play(a)
    def sample(self,s,rng,viewer):
        available=tuple(c for c in range(3,36) if c not in s.seen)
        return s.draw(available[min(int(rng.random()*len(available)),len(available)-1)])
    def rewards(self,s):
        if not self.terminal(s):raise ValueError('terminal rewards only')
        scores=s.scores(); bound=sum(range(3,36))+self.total_chips
        # Rules-derived global range [-total_chips, sum(all cards)]. A fixed
        # normalization declaration, not a tuned positional evaluation.
        returns=tuple(1-2*(x+self.total_chips)/bound for x in scores)
        return Rewards(winner_credits(scores,True),returns,scores)
    def observation(self,s):
        return dict(cards=[list(h) for h in s.cards],chips=list(s.chips),turn=s.turn,card=s.card,pot=s.pot,
                    seen=list(s.seen),remaining=s.remaining,payments=list(s.payments))


def reward_effect(rewards,viewer,mode):
    """One mapping for ALL games; its social interpretation is an assumption.

Competition payoffs are not safety, friendship or fairness measurements.
reward_persona explicitly interprets higher own return as achievement, lower
own downside as security, relative return as power and others' mean return as
benevolence. It does not invent physiology/growth/style/relationship effects.
"""
    objective=2*rewards.credits[viewer]-1
    if mode!='reward_persona':return effect(objective,values={'achievement':objective})
    own=rewards.returns[viewer]
    others=sum(x for i,x in enumerate(rewards.returns) if i!=viewer)/(len(rewards.returns)-1)
    return effect(objective,values=dict(achievement=own,security=min(own,0.),power=(own-others)/2,benevolence=others))


class RuleModel:
    def __init__(self,rules,viewer,mode):self.rules=rules;self.viewer=viewer;self.mode=mode
    def begin_trial(self):pass
    def terminal(self,s):return self.rules.terminal(s)
    def chance(self,s):return self.rules.chance(s)
    def sample(self,s,rng):return self.rules.sample(s,rng,self.viewer)
    def legal(self,s):return self.rules.legal(s)
    def step(self,s,a):return self.rules.step(s,a)
    def choose(self,s,rng,policy):
        names=self.rules.legal(s)
        return names[min(int(rng.random()*len(names)),len(names)-1)]
    def evaluate(self,s):
        r=self.rules.rewards(s); credit=r.credits[self.viewer]
        return TerminalEvaluation(reward_effect(r,self.viewer,self.mode),float(credit),credit>0,
                                  all(c==r.credits[0] for c in r.credits),float(r.scores[self.viewer]))


def observe(rules,state,viewer,profile,mode,budget,seed,tick,episode,memory=None):
    """No intermediate evaluation; incomplete trials produce zero-information.

Rules-only fallback keeps every legal action tied. It cannot secretly call a
game heuristic. Planning/adapter cost is separate from ordinary reflex speed.
"""
    if mode not in MODES:raise ValueError('unknown baseline mode')
    if not 0<=viewer<rules.players or rules.terminal(state) or rules.chance(state) and not rules.simultaneous:
        raise ValueError('live observed decision required')
    if not rules.simultaneous and rules.actor(state)!=viewer:
        raise ValueError('sequential decision belongs to the current actor')
    if budget.rollout_policy!='random':raise ValueError('baseline requires uniform continuation')
    names=rules.legal(state,viewer)
    roots={a:rules.root(state,a,viewer) for a in names}
    packed,stats=evaluate_actions(roots,RuleModel(rules,viewer,mode),budget,[seed,rules.game,episode,f'player-{viewer}',tick])
    acts=[action(a,*(packed[a] if stats['used'] else [effect()])) for a in names]
    c=make_context(rules,state,viewer,profile,mode,seed,tick,episode,acts,memory)
    stats.update(mode=mode,intermediate_evaluation='none',goal='win_share',continuation='uniform',
                 supported_values=['achievement'] if mode!='reward_persona' else ['achievement','security','power','benevolence'])
    return c,stats


def make_context(rules,state,viewer,profile,mode,seed,tick,episode,acts,memory=None):
    """Shared personality binding; planners supply effects, never rewrite persona."""
    actual=profile if mode!='rules_only' else dict(traits=(.5,)*5,values={})
    c=context(episode,acts,{},actual['values'],dict(zip(TRAITS,actual['traits'])),mode=None)
    for k in NEEDS:c['needs'][k]=dict(supported=False,enabled=False,deficit=None)
    c.update(scope=dict(game=rules.game,episode=episode,npc=f'player-{viewer}'),seed=seed,tick=tick,
             objective='終局の優勝持分を基準に本人の価値づけで選ぶ',
             facts=dict(evaluation='合法ルールと終局報酬のみ。未来は全員独立一様。実勝率ではない',
                        semantics='勝敗のみ' if mode!='reward_persona' else '共通の終局利益ベクトル→達成/安全/権力/慈善。意味づけの仮定',
                        unsupported='欲求全領域、様式、関係性、成長、自律、刺激、伝統、同調、普遍性',
                        fallback='標本不足なら候補は無情報の同点。専用評価へ戻さない'))
    for key,value in rules.observation(state).items():
        if isinstance(value,list) and value and isinstance(value[0],list):
            for i,row in enumerate(value):c['facts'][f'observation_{key}_{i}']=str(row)
        else:c['facts'][f'observation_{key}']=str(value)
    if memory is not None:c['state']=copy.deepcopy(memory)
    return c


def decide(rules,state,viewer,profile,mode,budget,seed,tick,episode,memory=None):
    c,stats=observe(rules,state,viewer,profile,mode,budget,seed,tick,episode,memory)
    b=compile_batch([c]); scored=Policy().decide(b,stochastic=False); d=scored.records(b)[0]
    stats['persona_scores']={a:float(scored.scores[0,i]) for i,a in enumerate(b.ids[0])}
    stats['eligible']=[a for i,a in enumerate(b.ids[0]) if scored.eligible[0,i]]
    best=max((x['win_share'] or 0.) for x in stats['actions'].values())
    chosen=stats['actions'][d['action_id']]['win_share']
    stats['estimated_goal_regret']=None if chosen is None else float(best-chosen)
    return c,d,stats
