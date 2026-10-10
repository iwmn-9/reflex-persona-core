"""Second adapter: closed-loop incumbent planning on a public spatial board.

Uses the same paired-policy evaluator as the hidden-future card game. The
declared rival hypothesis is minimax2; it is not inferred from a secret roster.
"""
import copy
from dataclasses import dataclass, replace
import numpy as np
from .board_models import ConnectAdapter, observe, connect_referee
from .board_planning import SHORT, REFLEX
from .core import Policy, digest
from .examples import action
from .monte_carlo import TerminalEvaluation, RolloutBudget
from .monte_carlo_comparison import minimax2
from .policy_rollout import evaluate_policy
from .goal_guard import choose_with_goal


@dataclass(frozen=True)
class Branch:
    position: object
    root: str
    steps: int = 0


def incumbent(s,p,seed,tick,episode,state,budget=SHORT):
    c,stats=observe(ConnectAdapter(),s,p,budget,seed,tick,episode,state)
    return c,Policy(principle_priority='finite').choose(c,stochastic=False),stats


class OwnerPolicyModel:
    def __init__(self,s,p,seed,tick,episode,state,d,budget=SHORT,owner_policy='incumbent'):
        self.initial=s;self.viewer=s.turn;self.profile=copy.deepcopy(p)
        self.seed=seed;self.tick=tick;self.episode=episode;self.state=copy.deepcopy(state)
        if owner_policy not in ('incumbent','reflex'):raise ValueError('declared owner continuation required')
        self.owner_policy=owner_policy
        self.decision=copy.deepcopy(d);self.budget=budget if owner_policy=='incumbent' else REFLEX;self.adapter=ConnectAdapter()
        self.calls=0;self.requests=0;self.cache={};self.committed=False
    def begin_trial(self):self.memory=copy.deepcopy(self.decision['next_state']);self.committed=False
    def terminal(self,b):return self.adapter.terminal(b.position)
    def chance(self,b):return False
    def legal(self,b):return self.adapter.legal(b.position)
    def choose(self,b,rng,policy):
        if not self.committed:
            self.memory['intent_action']=b.root
            self.memory['age']=min(self.state['age']+1,1000000) if self.state and self.state['intent_action']==b.root else 0
            self.committed=True
        s=b.position
        if s.turn!=self.viewer:return minimax2(s,rng)
        self.requests+=1;key=(s,self.tick+b.steps+1,digest(self.memory))
        if key not in self.cache:
            _,d,_=incumbent(s,self.profile,self.seed,self.tick+b.steps+1,self.episode,self.memory,self.budget)
            if len(self.cache)>=4096:self.cache.pop(next(iter(self.cache)))
            self.cache[key]=copy.deepcopy(d);self.calls+=1
        d=self.cache[key]
        self.memory=copy.deepcopy(d['next_state'])
        return d['action_id']
    def step(self,b,a):
        before=b.position;after=before.play(a)
        w,legal,_=connect_referee(after)
        assert w==after.winner() and legal==after.legal()
        return replace(b,position=after,steps=b.steps+1)
    def evaluate(self,b):
        winner=b.position.winner();draw=winner is None;won=winner==self.viewer
        credit=.5 if draw else float(won)
        row=self.adapter.consequence(self.initial,b.position,self.viewer)
        row['objective']=2*credit-1;row['values']['achievement']=row['objective']
        return TerminalEvaluation(row,credit,won,draw,None)


def decide(s,p,seed,tick,episode,state=None,budget=SHORT,rollout=None,owner_policy='incumbent'):
    c,d,st=incumbent(s,p,seed,tick,episode,state,budget)
    rb=RolloutBudget(samples=8,min_samples=8,max_nodes=100000,max_steps=42,rollout_policy='persona') if rollout is None else rollout
    model=OwnerPolicyModel(s,p,seed,tick,episode,state,d,budget,owner_policy)
    roots={a:Branch(s.play(a),a) for a in s.legal()}
    packed,samples,rs=evaluate_policy(roots,model,rb,[seed,'connect-four',episode,tick,'actual-base-policy'])
    rs.update(owner_searches=model.calls,owner_requests=model.requests,cache_hits=model.requests-model.calls,
        incumbent_action=d['action_id'],owner_policy=owner_policy,
        continuation=('actual SHORT own controller' if owner_policy=='incumbent' else 'reflex own controller approximation')+' after each hypothetical public move; minimax2 rival hypothesis')
    st=dict(st,policy_rollout=rs)
    if not rs['used']:return c,d,st
    acts=[action(a,*packed[a]) for a in s.legal()]
    for a in acts:a['target']=ConnectAdapter().target(s,a['id'])
    rc=ConnectAdapter().context(s,p,seed,tick,episode,acts,state)
    rc['facts']['continuation']=rs['continuation']
    names=s.legal();shares=np.array([[v.win_share for v in samples[a]] for a in names])
    rd,guard=choose_with_goal(rc,names,shares)
    rc['actions']=[a for a in rc['actions'] if a['id'] in guard['allowed']]
    rs.update(changed=rd['action_id']!=d['action_id'],guard=guard,action=rd['action_id'])
    return rc,rd,st
