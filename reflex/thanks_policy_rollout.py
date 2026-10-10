"""Experimental public-belief rollout of the actual incumbent NPC controller.

No true rival persona, benchmark seat, rival private state or future world deck
is accepted. Future owner searches use the SAME incumbent budget and policy;
the improved root selector itself is not recursively called in its own model.
"""
import copy
import pickle
from dataclasses import dataclass, replace,asdict
import numpy as np
from .board_models import ThanksAdapter, thanks_referee_score
from .core import digest
from .monte_carlo import RolloutBudget, TerminalEvaluation
from .policy_rollout import evaluate_policy
from .strong_search import NAMES, PERSONA, persona_context,PublicMemory,search,objective_choice,random_stream
from .laboratory import PROFILES
from .goal_progress import relative_progress, choose_with_progress, omit_expired_proxies


@dataclass(frozen=True)
class Branch:
    position: object
    root: str
    steps: int = 0


class OwnerPolicyModel:
    def __init__(self, position, viewer, profile, memory, owner_state, root_decision,
                 seed, encounter, tick, mode, budget=PERSONA, base=None,future_seed='same_owner',rival_policy='reactive'):
        from .strong_table import decide
        self.initial=position;self.viewer=viewer;self.profile=copy.deepcopy(profile)
        self.initial_memory=copy.deepcopy(memory);self.initial_state=copy.deepcopy(owner_state)
        self.root_decision=copy.deepcopy(root_decision)
        self.seed=seed;self.encounter=encounter;self.tick=tick;self.mode=mode
        self.budget=budget;self.base=decide if base is None else base
        self.cache={} if base is None else None;self.policy_requests=0
        if future_seed not in ('same_owner','resampled'):raise ValueError('declared future owner seed required')
        if viewer!=position.turn or len(position.chips)!=4 or sum(position.chips)+position.pot!=44:
            raise ValueError('active owner in the four-player public chip ledger required')
        self.future_seed=future_seed
        if rival_policy not in ('reactive','searched'):raise ValueError('declared rival continuation required')
        self.rival_policy=rival_policy;self.rival_cache={};self.rival_requests=0;self.rival_searches=0
        self.policy_calls=0;self.transitions=0;self.samples=[];self.starts=0

    def begin_trial(self):
        self.trial=self.starts//len(self.initial.legal());self.starts+=1
        self.memory=copy.deepcopy(self.initial_memory)
        self.owner_state=copy.deepcopy(self.root_decision['next_state'])
        self.deck=None;self.kinds=None;self.nonce=None;self.root_committed=False
        self.rival_memories=[PublicMemory('no_thanks',a) for a in range(4)] if self.rival_policy=='searched' else None
        self.rival_states=[None]*4

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
        if self.rival_policy=='reactive':
            self.kinds=[NAMES[int(common.choice(len(NAMES),p=self.initial_memory.weights(a,self.mode=='adaptive')))] for a in range(4)]
        else:
            # A new unlearned public prior, NOT weights borrowed from the old
            # reactive hypotheses and NOT identities from the real roster.
            menu=('objective',*(p['id'] for p in PROFILES))
            self.kinds=[menu[int(common.choice(len(menu),p=np.full(len(menu),1/len(menu))))] for a in range(4)]
        self.nonce=int(common.integers(0,2**62))

    def _commit_root(self,b):
        if self.root_committed:return
        self.owner_state['intent_action']=b.root
        old=self.initial_state
        self.owner_state['age']=min(old['age']+1,1000000) if old and old['intent_action']==b.root else 0
        if self.rival_memories is not None:
            for a in range(4):
                if a!=self.viewer:self.rival_memories[a].observe(self.initial,self.viewer,b.root,f'virtual-root-{self.nonce}')
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
            # own sandboxed state/memory. Retaining its known decision seed also
            # retains the actor/episode identity and future search/tie streams.
            # Resampled future streams are an explicit diagnostic alternative.
            owner_seed=self.seed if self.future_seed=='same_owner' else self.nonce
            self.policy_requests+=1
            key=(s,owner_seed,self.tick+b.steps+1,pickle.dumps((self.memory,self.owner_state),protocol=5))
            if self.cache is None or key not in self.cache:
                _,d,_=self.base('no_thanks',s,self.viewer,self.profile,owner_seed,
                    self.encounter,self.tick+b.steps+1,self.memory,self.owner_state,
                    self.mode,self.budget,variant='certified_expiry')
                self.policy_calls+=1
                if self.cache is not None:
                    if len(self.cache)>=4096:self.cache.pop(next(iter(self.cache)))
                    self.cache[key]=copy.deepcopy(d)
            else:d=self.cache[key]
            self.owner_state=copy.deepcopy(d['next_state'])
            return d['action_id']
        if self.rival_policy=='searched':
            return self._searched_rival(s,b.steps)
        if len(legal)==1:return legal[0]
        probs=self.memory.models(s,s.turn,self.mode=='adaptive')[self.kinds[s.turn]]
        probs=np.array([.92*probs[a]+.08/len(legal) for a in legal])
        return legal[int(rng.choice(len(legal),p=probs))]

    def step(self,b,a):
        s=b.position
        if s.turn!=self.viewer and self.mode=='adaptive':
            self.memory.observe(s,s.turn,a,f'virtual-{self.nonce}-{b.steps}-{s.turn}')
        if self.rival_memories is not None:
            for viewer in range(4):
                if viewer!=self.viewer and viewer!=s.turn:
                    self.rival_memories[viewer].observe(s,s.turn,a,f'virtual-{self.nonce}-{b.steps}-{s.turn}')
        self.transitions+=1
        return replace(b,position=s.play(a),steps=b.steps+1)

    def _searched_rival(self,s,steps):
        from .strong_table import decide as rival_decide
        actor=s.turn;kind=self.kinds[actor];memory=self.rival_memories[actor];state=self.rival_states[actor]
        tick=self.tick+steps+1;key=(s,actor,kind,self.nonce,tick,pickle.dumps((memory,state),protocol=5))
        self.rival_requests+=1
        if key not in self.rival_cache:
            if kind=='objective':
                names,scores,shares,_=search('no_thanks',s,actor,memory,True,self.budget,
                    random_stream(self.nonce,'no_thanks',self.encounter,tick,actor,'public-objective-hypothesis'))
                result=(names[objective_choice('no_thanks',names,scores,shares,actor)],None)
            else:
                profile=next(p for p in PROFILES if p['id']==kind)
                _,d,_=rival_decide('no_thanks',s,actor,profile,self.nonce,self.encounter,tick,memory,state,
                    'adaptive',self.budget,variant='certified_expiry')
                result=(d['action_id'],copy.deepcopy(d['next_state']))
            if len(self.rival_cache)>=4096:self.rival_cache.pop(next(iter(self.rival_cache)))
            self.rival_cache[key]=result;self.rival_searches+=1
        action,next_state=self.rival_cache[key];self.rival_states[actor]=copy.deepcopy(next_state)
        return action

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
           *,variant='certified_expiry',rollout=None,selection='direct',future_seed='same_owner',rival_policy='reactive'):
    from .strong_table import decide as incumbent
    c,d,stats=incumbent(game,s,viewer,p,seed,encounter,tick,memory,state,mode,budget,variant=variant)
    if selection not in ('direct','paired_guard'):raise ValueError('registered rollout selection required')
    if game!='no_thanks' or mode=='reflex' or variant!='certified_expiry' or len(s.legal())<2:return c,d,stats
    rollout=RolloutBudget(samples=8,min_samples=8,max_nodes=100000,max_steps=2048,rollout_policy='persona') if rollout is None else rollout
    model=OwnerPolicyModel(s,viewer,p,memory,state,d,seed,encounter,tick,mode,budget,future_seed=future_seed,rival_policy=rival_policy)
    roots={a:Branch(s.play(a),a) for a in s.legal()}
    _,samples,rs=evaluate_policy(roots,model,rollout,[seed,game,encounter,tick,viewer,'actual-base-policy'])
    rs.update(owner_searches=model.policy_calls,owner_requests=model.policy_requests,
        cache_hits=model.policy_requests-model.policy_calls,owner_budget=asdict(budget),future_seed=future_seed,
        rival_policy=rival_policy,rival_requests=model.rival_requests,rival_searches=model.rival_searches,rival_budget=asdict(budget),
        rival_initial_state='unknown private rival state/memory represented by a fresh fictional public-history observer; no real rival state copied' if rival_policy=='searched' else 'owner-observed reactive model mixture',
        continuation='Incumbent owner replans each public turn; '+
            ('five unlearned searched rival priors' if rival_policy=='searched' else 'owner-learned reactive rival mixture')+'; no recursive improved controller')
    stats=dict(stats,policy_rollout=rs,incumbent_action=d['action_id'])
    if not rs['used']:return c,d,stats
    names=s.legal();n=rs['completed_samples']
    terminal={a:[sc for root,sc in model.samples if root==a][:n] for a in names}
    scores=np.array([terminal[a] for a in names],float)
    shares=np.array([[v.win_share for v in samples[a]] for a in names])
    rc=persona_context(game,s,viewer,p,seed,tick,f'series-{seed}-encounter-{encounter}',state,names,scores,shares)
    if s.remaining==0 and s.chips[(viewer+1)%4]==0:
        rc=omit_expired_proxies(rc,needs=('safety',),values=('security',),style=('neuroticism',))
        rc['facts']['expiry_certificate']=c['facts']['expiry_certificate']
    rc['facts']['forecast']=rs['continuation'];rc['facts']['continuation']=rs['continuation']
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
    stats.update(action=rd['action_id'],guard=guard,incumbent_search={k:v for k,v in stats.items() if k not in ('policy_rollout','incumbent_action')},
        sample_count=n,actions={a:dict(win_share=float(shares[i].mean()),
            standard_error=float(shares[i].std(ddof=1)/np.sqrt(n)) if n>1 else 0.,
            mean_score=float(scores[i,:,viewer].mean()),goal_progress=float(progress[i].mean())) for i,a in enumerate(names)})
    return rc,rd,stats
