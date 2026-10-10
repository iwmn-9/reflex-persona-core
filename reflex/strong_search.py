"""Objective-led public-information search, separate from personality choice.

Goofspiel optimizes one shared open-loop card allocation across sampled rival
plans, then validates each root's plan on fresh scenarios. No determinization
gets its own clairvoyant response. No Thanks evaluates root/continuation pairs
against public unseen-card samples. Both replan after each real observation.
These are bounded search opponents, not solved-game or maximal-intelligence CPUs.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
import copy
import math
import numpy as np
from .core import TRAITS, digest, compile_batch, Policy, FEATURES
from .laboratory import PROFILES
from .goofspiel import make_context, consequence
from .board_models import ThanksAdapter, ThanksPosition, card_points
from .examples import action, effect
from .tabletop_trials import competitive, score, thanks_observe
from .planning import compress_outcomes
from .opponent_beliefs import HypothesisTracker
from .goal_guard import choose_with_goal

NAMES=('uniform','reserve','high','low','accept','cheap','growth','steady','care','ego','recent','conditional')


def distribution(names,choice=None):
    return {a:(1/len(names) if choice is None else float(a==choice)) for a in names}


def nearest(hand,target):
    return min(hand,key=lambda b:(abs(b-target),b))


@lru_cache(maxsize=4096)
def _public_models(game,s,actor):
    """Shared pure hypotheses; no observer memory or actual roster in cache."""
    result={}
    if game=='goofspiel':
        h=s.hands[actor];names=tuple(f'BID:{b}' for b in h);prize=s.prizes[s.round]
        choices=dict(reserve=nearest(h,prize),high=max(h),low=min(h),accept=nearest(h,prize+2),cheap=nearest(h,prize-2))
        result.update({k:distribution(names,f'BID:{b}') for k,b in choices.items()})
        for p in PROFILES:
            c=competitive(make_context(s,actor,p,'win_share',0,s.round,'hypothesis'))
            result[p['id']]=distribution(names,score(c)[0]['action_id'])
    else:
        names=s.legal();added=card_points(s.cards[actor]+(s.card,))-card_points(s.cards[actor])
        def pick(margin,reserve=0):return 'TAKE' if len(names)==1 or s.chips[actor]<=reserve or added<=s.pot+margin else 'PASS'
        choices=dict(reserve=pick(0,2),high=pick(9),low=pick(-2),accept=pick(4),cheap=pick(0))
        result.update({k:distribution(names,v) for k,v in choices.items()})
        for p in PROFILES:
            c,_=thanks_observe(s,p,0,0,'hypothesis',None)
            result[p['id']]=distribution(names,score(c)[0]['action_id'])
    result['uniform']=distribution(names)
    return result


class PublicMemory:
    """One observer's bounded, persistent predictions; only revealed acts learn.

    Actual profiles/controllers are never inputs. Candidate persona policies are
    hypotheses, not identities. IDs include encounters to avoid round-ID reuse.
    Recent observations are retained per rival, capped at 16, not full histories.
    """
    def __init__(self,game,viewer,players=4):
        if game not in ('goofspiel','no_thanks') or players!=4 or type(viewer) is not int or not 0<=viewer<players:
            raise ValueError('this four-player experiment requires a valid game/observer')
        self.game=game;self.viewer=viewer;self.players=players
        self.trackers=[HypothesisTracker(NAMES,retention=.94,window=12) for _ in range(players)]
        self.recent=[[] for _ in range(players)]
        self.initial_coefficients=np.array([.5,.2,0.,0.,0.,0.]) if game=='goofspiel' else np.array([0.,-4.,-.5,0.,0.,0.])
        self.coefficients=[self.initial_coefficients.copy() for _ in range(players)]

    def features(self,s,actor):
        if self.game=='goofspiel':
            rank=s.prizes[s.round]/max(s.prizes)-.5
            gap=(max(s.scores)-s.scores[actor])/sum(s.prizes)
            return np.array([1.,rank,gap,s.round/len(s.prizes),max(s.hands[actor])/max(s.prizes)-.5,0.])
        added=card_points(s.cards[actor]+(s.card,))-card_points(s.cards[actor]);scores=s.scores()
        gap=(scores[actor]-min(v for a,v in enumerate(scores) if a!=actor))/60
        return np.array([1.,(added-s.pot)/20,s.chips[actor]/11,gap,s.remaining/24,s.pot/10])

    def recent_threshold(self,actor,use_history=True):
        past=self.recent[actor] if use_history else []
        accepted=[gap for take,gap in past if take]
        rejected=[gap for take,gap in past if not take]
        threshold=(max(accepted)+min(rejected))/2 if accepted and rejected else (max(accepted) if accepted else 0.)
        return float(np.clip(threshold,-4,12))

    def models(self,s,actor,use_history=True):
        result=dict(_public_models(self.game,s,actor))
        if self.game=='goofspiel':
            h=s.hands[actor];names=tuple(f'BID:{b}' for b in h);prize=s.prizes[s.round]
            past=self.recent[actor] if use_history else []
            # Learn contextual bid rank, rather than memorizing a disappearing card.
            target=(sum(v[1] for v in past)/len(past)) if past else .5
            result['recent']=distribution(names,f'BID:{h[min(len(h)-1,round(target*(len(h)-1)))]}')
            coefficients=self.coefficients[actor] if use_history else self.initial_coefficients
            center=float(np.clip(coefficients@self.features(s,actor),0,1))
            ranks=np.linspace(0,1,len(h));prob=np.exp(-((ranks-center)/.16)**2);prob/=prob.sum()
            result['conditional']=dict(zip(names,map(float,prob)))
        else:
            names=s.legal();added=card_points(s.cards[actor]+(s.card,))-card_points(s.cards[actor])
            def pick(margin,reserve=0):
                return 'TAKE' if len(names)==1 or s.chips[actor]<=reserve or added<=s.pot+margin else 'PASS'
            result['recent']=distribution(names,pick(self.recent_threshold(actor,use_history)))
            coefficients=self.coefficients[actor] if use_history else self.initial_coefficients
            z=float(np.clip(coefficients@self.features(s,actor),-12,12));take=1/(1+math.exp(-z))
            result['conditional']=({'TAKE':1.} if len(names)==1 else {'TAKE':take,'PASS':1-take})
        return {n:result[n] for n in NAMES}

    def weights(self,actor,adaptive=True):
        if not adaptive:return np.full(len(NAMES),1/len(NAMES))
        snap=self.trackers[actor].snapshot()
        # Model averaging itself, with a fixed broad floor. The existing trust
        # diagnostic stays recorded; entropy is not treated as action accuracy.
        return .9*np.asarray(snap.weights)+.1/len(NAMES)

    def predict(self,s,actor,adaptive=True,models=None):
        models=self.models(s,actor,adaptive) if models is None else models
        weights=self.weights(actor,adaptive);names=tuple(next(iter(models.values())))
        return {a:float(sum(w*((1-.08)*models[n][a]+.08/len(names)) for n,w in zip(NAMES,weights))) for a in names}

    def observe(self,s,actor,revealed,observation_id):
        if actor==self.viewer:return None
        models=self.models(s,actor);prediction=self.predict(s,actor,models=models)
        row=self.trackers[actor].observe(models,revealed,observation_id,forecast=prediction)
        row['mixture_probability']=prediction[revealed]
        row['mixture_log_loss']=-math.log(prediction[revealed])
        if len(prediction)>1:
            if self.game=='goofspiel':
                h=s.hands[actor];bid=int(revealed.split(':')[1]);entry=(s.prizes[s.round],h.index(bid)/(len(h)-1))
            else:entry=(revealed=='TAKE',card_points(s.cards[actor]+(s.card,))-card_points(s.cards[actor])-s.pot)
            self.recent[actor]=(self.recent[actor]+[entry])[-16:]
            features=self.features(s,actor);w=self.coefficients[actor]
            if self.game=='goofspiel':pred=float(np.clip(w@features,0,1));error=entry[1]-pred;rate=.18
            else:pred=1/(1+math.exp(-float(np.clip(w@features,-12,12))));error=float(entry[0])-pred;rate=.25
            self.coefficients[actor]=np.clip(.998*w+rate*error*features,-8,8)
        return row

    def record(self):return [dict(t.snapshot().record(),conditional_coefficients=list(map(float,self.coefficients[i]))) for i,t in enumerate(self.trackers)]

    def forecast_weights(self,s,actor,adaptive=True):return self.weights(actor,adaptive)

    def forecast_coefficients(self,s,actor,adaptive=True):
        return self.coefficients[actor] if adaptive else self.initial_coefficients

    def rollout_banks(self,adaptive=True):
        return [dict(weights=np.array([self.weights(a,adaptive) for a in range(self.players)]),
                     thresholds=np.array([self.recent_threshold(a,adaptive) for a in range(self.players)]),
                     coefficients=np.array([self.forecast_coefficients(None,a,adaptive) for a in range(self.players)]))]

    def rollout_bank_indices(self,remaining):return np.zeros_like(remaining,dtype=int)


@dataclass(frozen=True)
class SearchBudget:
    train: int = 64
    validate: int = 128
    plans: int = 96
    generations: int = 4

    def __post_init__(self):
        if any(type(v) is not int or v<1 for v in (self.train,self.validate,self.plans,self.generations)):
            raise ValueError('positive integer search budget required')
        if self.train<8 or self.validate<16 or self.plans<16:raise ValueError('insufficient independent search budget')


STRONG=SearchBudget(128,256,192,5)
PERSONA=SearchBudget(32,96,64,3)


def random_stream(seed,game,encounter,tick,viewer,phase):
    return np.random.default_rng(int(digest([seed,game,encounter,tick,viewer,phase])[:16],16))


def _goof_scenarios(s,viewer,memory,adaptive,count,rng):
    remaining=len(s.hands[viewer]);result=np.zeros((count,len(s.hands),remaining),int)
    for actor,h in enumerate(s.hands):
        if actor==viewer:continue
        models=memory.models(s,actor,adaptive);weights=memory.forecast_weights(s,actor,adaptive)
        kinds=rng.choice(len(NAMES),count,p=weights)
        values=np.asarray(h);rows=np.arange(count);available=np.ones((count,len(h)),bool)
        root_probs=np.array([[.92*models[name][f'BID:{b}']+.08/len(h) for b in h] for name in NAMES])
        first=(rng.random(count)[:,None]>np.cumsum(root_probs[kinds],axis=1)).sum(1)
        first=np.minimum(first,len(h)-1);result[:,actor,0]=values[first];available[rows,first]=False
        ranks=root_probs.argmax(1)/max(1,len(h)-1)
        for r in range(1,remaining):
            left=remaining-r;prize=s.prizes[s.round+r]
            target=prize+np.array([0,0,0,0,2,-2,0,0,0,0,0,0])[kinds]
            chosen=np.where(available,abs(values[None,:]-target[:,None]),1000).argmin(1)
            desired=np.clip(np.rint(ranks[kinds]*(left-1)),0,left-1).astype(int)
            features=memory.features(s,actor).copy();features[1]=prize/max(s.prizes)-.5;features[3]=(s.round+r)/len(s.prizes)
            coefficients=memory.forecast_coefficients(s,actor,adaptive)
            conditional_rank=float(np.clip(coefficients@features,0,1))
            desired=np.where(kinds==11,round(conditional_rank*(left-1)),desired)
            desired=np.where(kinds==0,rng.integers(left,size=count),desired)
            desired=np.where(kinds==2,left-1,np.where(kinds==3,0,desired))
            ranked=(available.cumsum(1)>desired[:,None]).argmax(1)
            use_rank=(kinds==0)|(kinds==2)|(kinds==3)|(kinds>=6)
            chosen=np.where(use_rank,ranked,chosen)
            result[:,actor,r]=values[chosen];available[rows,chosen]=False
    return result


def _goof_returns(s,viewer,plans,scenarios):
    # Each own plan is shared across ALL scenarios. Current sealed rival bids
    # never enter candidate generation, nor does each sample get its own plan.
    scores=np.broadcast_to(np.asarray(s.scores),(len(plans),len(scenarios),len(s.hands))).copy()
    for r,prize in enumerate(s.prizes[s.round:]):
        bids=np.broadcast_to(scenarios[:,:,r],scores.shape).copy();bids[:,:,viewer]=plans[:,r,None]
        high=bids.max(2);unique=(bids==high[:,:,None]).sum(2)==1
        winner=bids.argmax(2)
        for actor in range(len(s.hands)):scores[:,:,actor]+=prize*unique*(winner==actor)
    wins=scores==scores.max(2,keepdims=True);shares=wins[:,:,viewer]/wins.sum(2)
    fitness=shares+.05*scores[:,:,viewer]/sum(s.prizes)
    return scores,shares,fitness


def goof_search(s,viewer,memory,adaptive,budget,rng):
    h=s.hands[viewer];n=len(h)
    train=_goof_scenarios(s,viewer,memory,adaptive,budget.train,rng)
    prize_order=np.argsort(s.prizes[s.round:]);reserve=np.empty(n,int);reserve[prize_order]=h
    plans=[reserve.copy(),reserve[::-1].copy()]
    for first in h:
        tail=[b for b in reserve if b!=first];plans.append(np.array([first]+tail))
    while len(plans)<budget.plans:plans.append(rng.permutation(h))
    plans=np.asarray(plans);best_by_root={}
    for generation in range(budget.generations):
        _,_,fitness=_goof_returns(s,viewer,plans,train);fit=fitness.mean(1)
        for b in h:
            ids=np.flatnonzero(plans[:,0]==b)
            if len(ids):
                i=int(ids[np.argmax(fit[ids])])
                if b not in best_by_root or fit[i]>best_by_root[b][0]:best_by_root[b]=(float(fit[i]),plans[i].copy())
        if n==1:break
        elites=plans[np.argsort(fit)[-max(4,budget.plans//8):]]
        next_plans=[v[1] for v in best_by_root.values()]
        next_plans.extend(elites)
        while len(next_plans)<budget.plans:
            child=elites[int(rng.integers(len(elites)))].copy()
            for _ in range(1+int(rng.integers(3))):
                a,b=rng.choice(n,2,replace=False);child[a],child[b]=child[b],child[a]
            next_plans.append(child)
        plans=np.asarray(next_plans[:budget.plans])
    validated=np.array([best_by_root[b][1] for b in h])
    fresh=_goof_scenarios(s,viewer,memory,adaptive,budget.validate,rng)
    scores,shares,fitness=_goof_returns(s,viewer,validated,fresh)
    return tuple(f'BID:{b}' for b in h),scores,shares,dict(
        candidate_plans=budget.plans*budget.generations,training_scenarios=budget.train,
        validation_scenarios=budget.validate,plans={f'BID:{b}':list(map(int,p)) for b,p in zip(h,validated)},
        continuation='shared own allocation; frozen rival hypotheses; closed-loop replanning after public reveal')


@lru_cache(maxsize=1)
def _thanks_templates():
    rows=[]
    for actor in range(4):
        s=ThanksPosition.start(4,20,actor)
        for p in PROFILES:
            c,_=thanks_observe(s,p,0,0,'hypothesis',None);rows.append(c)
    return compile_batch(rows)


def _thanks_persona_take_reference(active,turn,card,pot,chips,points,cards,kinds,remaining,model_states):
    """Same finite Policy in virtual persona hypotheses, using exact immediate
    rule effects. No real rival identity or memory is supplied to this model.
    """
    a=turn[active];typ=kinds[active,a];selected=active[(typ>=6)&(typ<=9)]
    if not len(selected):return selected,np.array([],bool)
    a=turn[selected];c=card[selected];stock=chips[selected,a];kind=kinds[selected,a]
    batch=_thanks_templates().take(a*4+kind-6);n=len(selected);rows=np.arange(n)
    added=np.where(cards[selected,a,c-1],0,c)-np.where(cards[selected,a,c+1],c+1,0)
    gain=np.stack((-np.ones(n),(pot[selected]-added)),axis=1)/35
    security=np.stack(((np.minimum(stock-1,8)-np.minimum(stock,8))/8,
                       (np.minimum(stock+pot[selected],8)-np.minimum(stock,8))/8),axis=1)
    other_added=np.where(cards[selected,:,c-1],0,c[:,None])-np.where(cards[selected,:,c+1],c[:,None]+1,0)
    other_added[rows,a]=1000;blocked=np.maximum(0,-other_added.min(1))/35
    effects=np.zeros((n,2,1,len(FEATURES)));idx={f:i for i,f in enumerate(FEATURES)}
    objective=gain.copy();last=remaining[selected]==0
    if last.any():
        final=points[selected]-chips[selected];final[rows,a]+=added-pot[selected]
        objective[last,1]=np.where(final[rows,a][last]==final.min(1)[last],1.,-1.)
    def set_effect(key,value):effects[:,:,0,idx[key]]=np.clip(value,-1,1)
    set_effect('objective',objective);set_effect('achievement',objective)
    set_effect('safety',security);set_effect('security',security);set_effect('style_neuroticism',security)
    set_effect('esteem',gain);set_effect('power',gain+np.stack((np.zeros(n),blocked),axis=1))
    set_effect('self_direction',gain);effects[:,0,0,idx['cost']]=.04
    needs=batch.needs.copy();needs[:,1]=np.clip(1-stock/8,.1,.9);needs[:,3]=.5
    legal=np.ones((n,2),bool);legal[:,0]=stock>0
    previous=model_states[selected,a]
    batch=replace(batch,effects=effects,needs=needs,legal=legal,mode=previous[:,0].astype(int),
                  primary=previous[:,1].astype(int),mode_urgency=previous[:,2])
    decisions=Policy(principle_priority='finite').decide(batch,False)
    model_states[selected,a,0]=decisions.mode;model_states[selected,a,1]=decisions.primary
    model_states[selected,a,2]=decisions.mode_urgency
    return selected,decisions.action==1


@lru_cache(maxsize=1)
def _persona_kernel():
    """Compile this experiment's deterministic one-outcome responses once.

    Both the state-transition table and numerical weights are obtained from the
    actual common Policy. No second handwritten personality formula. For two
    deterministic outcomes the shared downside transform is monotone, so its
    weighted-utility ordering is identical. This shortcut is only used inside
    the virtual finite-persona hypothesis, not the real stochastic NPC policy.
    """
    urgency=np.array([0.,.1,.125,.25,.375,.5,.625,.75,.875,.9])
    actors,profiles,stock,modes,primary,oldurg=np.indices((4,4,10,3,3,10)).reshape(6,-1)
    batch=_thanks_templates().take(actors*4+profiles);n=len(stock)
    needs=batch.needs.copy();needs[:,1]=np.clip(1-stock/8,.1,.9);needs[:,3]=.5
    legal=np.ones((n,2),bool);legal[:,0]=stock>0
    batch=replace(batch,needs=needs,effects=np.zeros_like(batch.effects),legal=legal,
        mode=modes-1,primary=np.array([-1,1,3])[primary],mode_urgency=urgency[oldurg])
    result=Policy(principle_priority='finite').decide(batch,False)
    transitions=np.stack((result.mode,result.primary,result.mode_urgency),axis=1).reshape(4,4,10,3,3,10,3)
    # Positive/negative unit effects expose the Policy's own coefficients. Cost
    # stays nonnegative; its negative score is unscaled using the objective's
    # positive/negative probes. All probes obey the public effect contract.
    contexts=[]
    for p in PROFILES:
        for amount in range(10):
            for mode in range(2):
                for primary_need in ('safety','esteem'):
                    c,_=thanks_observe(replace(ThanksPosition.start(4,20),chips=(amount,11,11,11)),p,0,0,'hypothesis',None)
                    c['state'].update(primary_need=primary_need,mode='principle' if mode else 'need',
                        mode_urgency=c['needs'][primary_need]['deficit'])
                    acts=[]
                    for j,feature in enumerate(FEATURES):
                        for sign in ((1,) if feature=='cost' else (1,-1)):
                            row=effect()
                            if feature in ('objective','cost'):row[feature]=float(sign)
                            elif feature.startswith('style_'):row['style'][feature[6:]]=float(sign)
                            elif feature in c['needs']:row['needs'][feature]=float(sign)
                            else:row['values'][feature]=float(sign)
                            acts.append(action(('p' if sign>0 else 'n')+f'{j:02}',row))
                    c['actions']=acts;contexts.append(c)
    batch=compile_batch(contexts);dec=Policy(principle_priority='finite').decide(batch,False)
    ids=batch.ids[0];plus=np.array([ids.index(f'p{j:02}') for j in range(len(FEATURES))])
    minus=np.array([ids.index(f'n{j:02}') for j in range(len(FEATURES)-1)])
    weights=dec.scores[:,plus].copy();neg=dec.scores[:,minus]
    weights[:,:-1]=np.where(weights[:,:-1]>=0,weights[:,:-1],-neg)
    downside=-neg[:,0]/weights[:,0]
    weights[:,-1]/=downside
    return transitions,weights.reshape(4,10,2,2,len(FEATURES)),urgency


def _thanks_persona_take(active,turn,card,pot,chips,points,cards,kinds,remaining,model_states):
    a=turn[active];typ=kinds[active,a];selected=active[(typ>=6)&(typ<=9)]
    if not len(selected):return selected,np.array([],bool)
    a=turn[selected];p=kinds[selected,a]-6;c=card[selected];stock=chips[selected,a]
    transitions,weights,urgencies=_persona_kernel();old=model_states[selected,a]
    stock_class=np.minimum(stock,9);primary=np.where(old[:,1]<0,0,np.where(old[:,1]==1,1,2))
    u=np.abs(urgencies[None,:]-old[:,2,None]).argmin(1)
    state=transitions[a,p,stock_class,(old[:,0]+1).astype(int),primary,u]
    model_states[selected,a]=state
    w=weights[p,stock_class,state[:,0].astype(int),(state[:,1]==3).astype(int)]
    n=len(selected);rows=np.arange(n);idx={f:i for i,f in enumerate(FEATURES)}
    added=np.where(cards[selected,a,c-1],0,c)-np.where(cards[selected,a,c+1],c+1,0)
    gain=np.clip(np.stack((-np.ones(n),pot[selected]-added),axis=1)/35,-1,1)
    security=np.stack(((np.minimum(stock-1,8)-np.minimum(stock,8))/8,
        (np.minimum(stock+pot[selected],8)-np.minimum(stock,8))/8),axis=1)
    other_added=np.where(cards[selected,:,c-1],0,c[:,None])-np.where(cards[selected,:,c+1],c[:,None]+1,0)
    other_added[rows,a]=1000;blocked=np.maximum(0,-other_added.min(1))/35
    objective=gain.copy();last=remaining[selected]==0
    if last.any():
        final=points[selected]-chips[selected];final[rows,a]+=added-pot[selected]
        objective[last,1]=np.where(final[rows,a][last]==final.min(1)[last],1.,-1.)
    utility=objective*(w[:,idx['objective']]+w[:,idx['achievement']])[:,None]
    utility+=gain*(w[:,idx['esteem']]+w[:,idx['self_direction']])[:,None]
    utility+=security*(w[:,idx['safety']]+w[:,idx['security']]+w[:,idx['style_neuroticism']])[:,None]
    power=np.clip(gain+np.stack((np.zeros(n),blocked),axis=1),-1,1)
    utility+=power*w[:,idx['power'],None];utility[:,0]+=.04*w[:,idx['cost']]
    return selected,(utility[:,1]>utility[:,0])|(stock==0)


def _thanks_simulate(s,viewer,memory,adaptive,roots,styles,count,rng,*,own_profile=None,own_state=None):
    """Vectorized exact transitions; unknown deck samples only PUBLIC unseen set."""
    if own_profile is not None and own_profile not in PROFILES:
        raise ValueError('the fixed continuation kernel supports exactly the four declared experiment profiles')
    n=len(roots)*count;players=len(s.chips);rows=np.arange(n)
    chips=np.tile(s.chips,(n,1));points=np.tile([card_points(c) for c in s.cards],(n,1))
    cards=np.zeros((n,players,37),bool)
    for actor,hand in enumerate(s.cards):cards[:,actor,list(hand)]=True
    turn=np.full(n,s.turn,int);card=np.full(n,s.card,int);pot=np.full(n,s.pot,int)
    unseen=np.array([c for c in range(3,36) if c not in s.seen])
    decks=np.array([rng.permutation(unseen)[:s.remaining] for _ in range(count)])
    decks=np.tile(decks,(len(roots),1));drawn=np.zeros(n,int);done=np.zeros(n,bool)
    banks=memory.rollout_banks(adaptive)
    bank_kinds=np.empty((len(banks),n,players),int)
    for bank_index,bank in enumerate(banks):
        for actor in range(players):
            bank_kinds[bank_index,:,actor]=np.tile(rng.choice(len(NAMES),count,p=bank['weights'][actor]),len(roots))
    own_style=np.repeat(styles,count);root=np.repeat(roots,count)
    model_states=np.zeros((n,players,3));model_states[:,:,:2]=-1
    if own_profile is not None:
        bank_kinds[:,:,viewer]=6+next(i for i,p in enumerate(PROFILES) if p['id']==own_profile['id'])
        if own_state is not None:
            # Same supported state representation used by compile_batch; the
            # actual owner's state is allowed, rival private states never are.
            c,_=thanks_observe(s,own_profile,0,0,'rollout',own_state)
            b=compile_batch([c]);model_states[:,viewer]=[b.mode[0],b.primary[0],b.mode_urgency[0]]
    # Root rival action distributions are not consulted: decision belongs to viewer.
    for step in range(2048):
        active=np.flatnonzero(~done)
        if not len(active):break
        bank_indices=memory.rollout_bank_indices(s.remaining-drawn)
        kinds=bank_kinds[bank_indices,rows]
        a=turn[active];c=card[active]
        added=np.where(cards[active,a,c-1],0,c)-np.where(cards[active,a,c+1],c+1,0)
        typ=kinds[active,a]
        margins=np.array([0,0,9,-2,4,0,4,0,1,7,0,0])[typ]
        thresholds=np.array([bank['thresholds'] for bank in banks])
        margins=np.where(typ==10,thresholds[bank_indices[active],a],margins)
        reserves=np.array([0,2,0,0,0,0,0,3,1,0,0,0])[typ]
        # Own continuation is one shared threshold policy per candidate, chosen
        # on training returns then evaluated on fresh decks, never per deck.
        own=a==viewer
        margins=np.where(own,np.array([-2,0,2,5,9,0,4,2])[own_style[active]],margins)
        reserves=np.where(own,np.array([0,0,0,0,0,2,2,4])[own_style[active]],reserves)
        take=(added<=pot[active]+margins)|(chips[active,a]<=reserves)
        uniform=(typ==0)&~own
        take=np.where(uniform,rng.random(len(active))<.5,take)
        persona_rows,persona_take=_thanks_persona_take(active,turn,card,pot,chips,points,cards,kinds,s.remaining-drawn,model_states)
        lookup=np.searchsorted(active,persona_rows);use=(turn[persona_rows]!=viewer)|(own_profile is not None)
        take[lookup[use]]=persona_take[use]
        conditional=(typ==11)&~own
        if conditional.any():
            current=points[active]-chips[active];other=current.astype(float);other[np.arange(len(active)),a]=np.inf
            feature=np.stack((np.ones(len(active)),(added-pot[active])/20,chips[active,a]/11,
                (current[np.arange(len(active)),a]-other.min(1))/60,(s.remaining-drawn[active])/24,pot[active]/10),axis=1)
            coefficients=np.array([bank['coefficients'] for bank in banks])
            w=coefficients[bank_indices[active],a]
            prob=1/(1+np.exp(-np.clip((feature*w).sum(1),-12,12)))
            take=np.where(conditional,rng.random(len(active))<prob,take)
        if step==0:take=root[active]=='TAKE'
        take|=chips[active,a]==0
        accepted=active[take];rejected=active[~take]
        if len(rejected):
            ar=turn[rejected];chips[rejected,ar]-=1;pot[rejected]+=1;turn[rejected]=(ar+1)%players
        if len(accepted):
            aa=turn[accepted];cc=card[accepted]
            ad=np.where(cards[accepted,aa,cc-1],0,cc)-np.where(cards[accepted,aa,cc+1],cc+1,0)
            points[accepted,aa]+=ad;cards[accepted,aa,cc]=True;chips[accepted,aa]+=pot[accepted];pot[accepted]=0
            finished=drawn[accepted]==s.remaining;done[accepted[finished]]=True
            continuing=accepted[~finished]
            if len(continuing):card[continuing]=decks[continuing,drawn[continuing]];drawn[continuing]+=1
    else:raise RuntimeError('No Thanks rollout failed to terminate')
    scores=points-chips;win=scores==scores.min(1,keepdims=True);shares=win[:,viewer]/win.sum(1)
    return scores.reshape(len(roots),count,players),shares.reshape(len(roots),count)


def thanks_search(s,viewer,memory,adaptive,budget,rng):
    names=s.legal();roots=tuple(a for a in names for _ in range(8));styles=tuple(range(8))*len(names)
    scores,shares=_thanks_simulate(s,viewer,memory,adaptive,roots,styles,budget.train,rng)
    fit=shares.mean(1)-.0005*scores[:,:,viewer].mean(1)
    chosen=[i*8+int(np.argmax(fit[i*8:(i+1)*8])) for i in range(len(names))]
    policies=tuple(styles[i] for i in chosen)
    scores,shares=_thanks_simulate(s,viewer,memory,adaptive,names,policies,budget.validate,rng)
    return names,scores,shares,dict(training_scenarios=budget.train,validation_scenarios=budget.validate,
        continuation_styles=dict(zip(names,policies)),continuation='one threshold policy per root; sampled public unseen cards; public replanning')


def search(game,s,viewer,memory,adaptive,budget,rng):
    names,scores,shares,stats=(goof_search if game=='goofspiel' else thanks_search)(s,viewer,memory,adaptive,budget,rng)
    means=shares.mean(1);paired=shares-means[:,None]
    stats['actions']={a:dict(win_share=float(means[i]),standard_error=float(shares[i].std(ddof=1)/np.sqrt(budget.validate)),
        mean_score=float(scores[i,:,viewer].mean())) for i,a in enumerate(names)}
    stats['sample_count']=budget.validate
    return names,scores,shares,stats


def objective_choice(game,names,scores,shares,viewer):
    # Lexicographic goal; raw score breaks exactly tied sampled winner credit.
    means=shares.mean(1);raw=scores[:,:,viewer].mean(1)
    sign=1 if game=='goofspiel' else -1
    return max(range(len(names)),key=lambda i:(means[i],sign*raw[i],-i))


def persona_context(game,s,viewer,p,seed,tick,episode,state,names,scores,shares,*,sample_weights=None):
    weights=np.full(shares.shape,1/shares.shape[1]) if sample_weights is None else np.asarray(sample_weights,dtype=float)
    if weights.shape!=shares.shape or not np.isfinite(weights).all() or np.any(weights<0) or not np.allclose(weights.sum(1),1):
        raise ValueError('one normalized finite nonnegative outcome mass per root required')
    if sample_weights is not None:weights=weights/weights.sum(1,keepdims=True)
    packed={}
    for i,name in enumerate(names):
        rows=[]
        # Group terminal shares (at most five for four players), preserving first
        # moments inside each winner group. Within-group subjective downside is
        # approximate, as in the existing eight-outcome compression contract.
        for share in np.unique(shares[i]):
            selected=shares[i]==share;mass=weights[i,selected]
            if mass.sum()==0:continue
            finals=scores[i,selected];others=[a for a in range(len(s.hands if game=='goofspiel' else s.chips)) if a!=viewer]
            average=lambda values:float(values.mean()) if sample_weights is None else float(np.average(values,weights=mass))
            if game=='goofspiel':
                remaining=max(1,sum(s.prizes[s.round:]));bid=int(name.split(':')[1])
                commitment=bid/max(s.prizes)*sum(s.prizes[s.round+1:])/remaining
                margin=((finals[:,viewer]-finals[:,others].max(1))-(s.scores[viewer]-max(s.scores[a] for a in others)))/remaining
                delta=(finals[:,viewer]-s.scores[viewer])/remaining
                row=effect(2*float(share)-1,needs={'esteem':average(delta),'safety':-.15*commitment},
                    values={'achievement':2*float(share)-1,'power':average(margin),'security':-commitment},
                    style={'openness':.2*commitment,'conscientiousness':-.2*commitment})
            else:
                # Separate root commitment from whole-game payments. Counting
                # all future passes as this action's cost punished good plans.
                root=s.play(name);adapter=ThanksAdapter()
                row=adapter.consequence(s,root,viewer)
                row['objective']=2*float(share)-1
                row['values']['achievement']=row['objective']
                margin=(finals[:,others].min(1)-finals[:,viewer])/35
                row['values']['power']=average(np.clip(margin,-1,1))
                row['needs']['esteem']=average(np.clip(margin,-1,1))
            row['p']=float(selected.mean()) if sample_weights is None else float(np.clip(mass.sum(),0,1));rows.append(row)
        packed[name]=compress_outcomes(rows)
    if game=='goofspiel':c=make_context(s,viewer,p,'win_share',seed,tick,episode,state,packed)
    else:c=ThanksAdapter().context(s,p,seed,tick,episode,[action(a,*packed[a]) for a in names],state)
    c['facts']['forecast']=f'終局までのモデル予測。各候補{scores.shape[1]}検証標本。相手は宣言した公開情報の仮説。実際の未来の再計画と同一ではない'
    c['facts']['continuation']='根ごとに全標本へ共通の本人の計画/方策。実際の他者の内部状態と未知山札は入力しない'
    if game=='no_thanks':c['facts']['estimate']=c['facts']['forecast']
    return competitive(c)


def reasonable_persona(c,names,shares,max_regret=.12):
    """Personality chooses among statistically unresolved/near-goal candidates.

    This is a purpose contract, not a personality change or an optimality claim.
    Paired SE is sampling error only, not uncertainty about model correctness.
    """
    return choose_with_goal(c,names,shares,max_regret=max_regret)
