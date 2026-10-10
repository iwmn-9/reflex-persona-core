"""Experimental public-belief rollout of the actual incumbent NPC controller.

No true rival persona, benchmark seat, rival private state or future world deck
is accepted. Future owner searches use the SAME incumbent budget and policy;
the improved root selector itself is not recursively called in its own model.
"""
import copy
from dataclasses import dataclass, replace
import numpy as np
from .board_models import ThanksAdapter, thanks_referee_score
from .core import digest
from .monte_carlo import RolloutBudget, TerminalEvaluation
from .policy_rollout import evaluate_policy
from .strong_search import NAMES, PERSONA, persona_context
from .goal_progress import relative_progress, choose_with_progress, omit_expired_proxies


@dataclass(frozen=True)
class Branch:
    position: object
    root: str
    steps: int = 0


class OwnerPolicyModel:
    def __init__(self, position, viewer, profile, memory, owner_state, root_decision,
                 seed, encounter, tick, mode, budget=PERSONA, base=None):
        from .strong_table import decide
        self.initial=position;self.viewer=viewer;self.profile=copy.deepcopy(profile)
        self.initial_memory=copy.deepcopy(memory);self.initial_state=copy.deepcopy(owner_state)
        self.root_decision=copy.deepcopy(root_decision)
        self.seed=seed;self.encounter=encounter;self.tick=tick;self.mode=mode
        self.budget=budget;self.base=decide if base is None else base
        self.policy_calls=0;self.transitions=0;self.samples=[];self.starts=0

    def begin_trial(self):
        self.trial=self.starts//len(self.initial.legal());self.starts+=1
        self.memory=copy.deepcopy(self.initial_memory)
        self.owner_state=copy.deepcopy(self.root_decision['next_state'])
        self.deck=None;self.kinds=None;self.nonce=None;self.root_committed=False

    def terminal(self,b): return b.position.card is None and b.position.remaining==0
    def chance(self,b): return b.position.card is None and b.position.remaining>0
    def legal(self,b): return b.position.legal()

    def _initialize(self,rng):
        if self.deck is not None:return
        # Common trial streams across roots; one latent rival hypothesis persists
        # through the trial. Future learning changes only this detached observer.
        # TAKE may first reach chance and PASS first reach a rival. Never choose
        # the latent world from whichever of those two RNG channels ran first.
        common=np.random.default_rng(int(digest([self.seed,self.encounter,self.tick,
            self.viewer,'base-policy-world',self.trial])[:16],16))
        unseen=np.array([c for c in range(3,36) if c not in self.initial.seen])
        self.deck=list(map(int,common.permutation(unseen)[:self.initial.remaining]))
        self.kinds=[NAMES[int(common.choice(len(NAMES),p=self.initial_memory.weights(a,self.mode=='adaptive')))]
                    for a in range(4)]
        self.nonce=int(common.integers(0,2**62))

    def _commit_root(self,b):
        if self.root_committed:return
        self.owner_state['intent_action']=b.root
        old=self.initial_state
        self.owner_state['age']=min(old['age']+1,1000000) if old and old['intent_action']==b.root else 0
        self.root_committed=True

    def sample(self,b,rng):
        self._initialize(rng);self._commit_root(b)
        self.transitions+=1
        # Actual play advances its decision clock on actions, not card reveals.
        return replace(b,position=b.position.draw(self.deck.pop(0)))

    def choose(self,b,rng,policy):
        self._initialize(rng);self._commit_root(b)
        s=b.position;legal=s.legal()
        if s.turn==self.viewer:
            # The actual incumbent selector receives PUBLIC position, own traits,
            # own sandboxed state/memory, and an independent search nonce only.
            _,d,_=self.base('no_thanks',s,self.viewer,self.profile,self.nonce,
                self.encounter,self.tick+b.steps+1,self.memory,self.owner_state,
                self.mode,self.budget,variant='certified_expiry')
            self.owner_state=copy.deepcopy(d['next_state']);self.policy_calls+=1
            return d['action_id']
        if len(legal)==1:return legal[0]
        probs=self.memory.models(s,s.turn,self.mode=='adaptive')[self.kinds[s.turn]]
        probs=np.array([.92*probs[a]+.08/len(legal) for a in legal])
        return legal[int(rng.choice(len(legal),p=probs))]

    def step(self,b,a):
        s=b.position
        if s.turn!=self.viewer and self.mode=='adaptive':
            self.memory.observe(s,s.turn,a,f'virtual-{self.nonce}-{b.steps}-{s.turn}')
        self.transitions+=1
        return replace(b,position=s.play(a),steps=b.steps+1)

    def evaluate(self,b):
        self._commit_root(b)
        s=b.position;scores=s.scores()
        assert scores==tuple(thanks_referee_score(c,h) for c,h in zip(s.cards,s.chips))
        assert sum(s.chips)==44 and s.pot==0
        winners=[a for a,x in enumerate(scores) if x==min(scores)]
        share=1/len(winners) if self.viewer in winners else 0.
        row=ThanksAdapter().consequence(self.initial,self.initial.play(b.root),self.viewer)
        row['objective']=2*share-1;row['values']['achievement']=row['objective']
        margin=(min(x for a,x in enumerate(scores) if a!=self.viewer)-scores[self.viewer])/35
        row['values']['power']=float(np.clip(margin,-1,1));row['needs']['esteem']=row['values']['power']
        self.samples.append((b.root,scores))
        return TerminalEvaluation(row,share,self.viewer in winners,len(winners)>1,float(scores[self.viewer]))


def decide(game,s,viewer,p,seed,encounter,tick,memory,state,mode,budget=PERSONA,
           *,variant='certified_expiry',rollout=None,selection='direct'):
    from .strong_table import decide as incumbent
    c,d,stats=incumbent(game,s,viewer,p,seed,encounter,tick,memory,state,mode,budget,variant=variant)
    if selection not in ('direct','paired_guard'):raise ValueError('registered rollout selection required')
    if game!='no_thanks' or mode=='reflex' or len(s.legal())<2:return c,d,stats
    rollout=RolloutBudget(samples=8,min_samples=8,max_nodes=100000,max_steps=2048,rollout_policy='persona') if rollout is None else rollout
    model=OwnerPolicyModel(s,viewer,p,memory,state,d,seed,encounter,tick,mode,budget)
    roots={a:Branch(s.play(a),a) for a in s.legal()}
    _,samples,rs=evaluate_policy(roots,model,rollout,[seed,game,encounter,tick,viewer,'actual-base-policy'])
    rs.update(owner_searches=model.policy_calls,owner_budget=vars(budget),
        continuation='actual certified_expiry base controller re-searches after every future owner turn; public rival hypotheses; NOT recursively improved controller')
    stats=dict(stats,policy_rollout=rs,incumbent_action=d['action_id'])
    if not rs['used']:return c,d,stats
    names=s.legal();n=rs['completed_samples']
    terminal={a:[sc for root,sc in model.samples if root==a][:n] for a in names}
    scores=np.array([terminal[a] for a in names],float)
    shares=np.array([[v.win_share for v in samples[a]] for a in names])
    rc=persona_context(game,s,viewer,p,seed,tick,f'series-{seed}-encounter-{encounter}',state,names,scores,shares)
    if s.remaining==0 and s.chips[(viewer+1)%4]==0:
        rc=omit_expired_proxies(rc,needs=('safety',),values=('security',),style=('neuroticism',))
    progress=relative_progress(scores,viewer,direction=-1,scale=35)
    rc,rd,guard=choose_with_progress(rc,names,shares,progress)
    rs['suggested_action']=rd['action_id'];rs['guard']=guard
    metric=progress if np.all(shares==0) else shares
    i=names.index(d['action_id']);j=names.index(rd['action_id']);delta=metric[j]-metric[i]
    gain=float(delta.mean());se=float(delta.std(ddof=1)/np.sqrt(n)) if n>1 else 0.
    rs.update(paired_gain=gain,paired_se=se,signal='progress' if np.all(shares==0) else 'winner_credit')
    changed=rd['action_id']!=d['action_id']
    accepted=not changed or selection=='direct' or gain>.12+1.96*se
    rs.update(selection=selection,changed=changed and accepted,accepted=accepted)
    if not accepted:return c,d,stats
    rc['facts']['forecast']=rs['continuation']
    rc['facts']['continuation']=rs['continuation']
    stats.update(action=rd['action_id'],guard=guard)
    return rc,rd,stats
