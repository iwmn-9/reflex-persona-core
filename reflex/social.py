"""Optional recipient-specific appraisal, separate from facts and predictions.

Finite decaying attitude memory + existing personality Policy + a purpose
corridor. Coefficients are engineering choices, not validated psychology.
Games identify observed attributable benefits and the beneficiaries of actions.
Neither a lost lottery nor an opponent's hidden intent is such evidence.
"""
from collections import OrderedDict
import copy
from dataclasses import dataclass, replace
import numpy as np
from .core import VALUES, compile_batch, digest, identifier, number
from .runtime import Population


@dataclass(frozen=True)
class SocialBinding:
    target: str
    support: float
    distress: float=0.

    def __post_init__(self):
        identifier(self.target);number(self.support,-1,1);number(self.distress,0,1)


@dataclass(frozen=True)
class SocialEvent:
    target: str
    event_id: str
    tick: int
    benefit: float
    agency: bool=True
    confidence: float=1.

    def __post_init__(self):
        identifier(self.target);identifier(self.event_id);number(self.benefit,-1,1);number(self.confidence,0,1)
        if type(self.tick) is not int or not 0<=self.tick<2**63-1 or type(self.agency) is not bool:
            raise ValueError('observed tick and explicit agency required')


class SocialMemory:
    """Own attitude, NOT calibrated expectations of another person's next action."""
    def __init__(self, context, capacity=64, half_life=24.):
        compile_batch([context])
        if type(capacity) is not int or not 1<=capacity<=1024:raise ValueError('bounded social capacity required')
        number(half_life,1,4096)
        self.owner=digest(context['scope']);self.personality=copy.deepcopy(context['personality'])
        self.values=copy.deepcopy(context['values']);self.capacity=capacity;self.half_life=float(half_life)
        self.entries=OrderedDict();self.events=OrderedDict();self.last_tick=-1

    def observe(self,event):
        if not isinstance(event,SocialEvent) or event.tick<self.last_tick:raise ValueError('ordered observed social event required')
        if event.event_id in self.events:raise ValueError('social evidence cannot be counted twice')
        # Validation finishes before mutation. Missing agency contributes no
        # grievance or confidence; one zero effect is not a broken promise.
        e=self.entries.get(event.target,dict(attitude=0.,evidence=0.,tick=event.tick))
        attitude=self.status(event.target,event.tick)['attitude']
        weight=event.confidence if event.agency and event.benefit!=0 else 0.
        agreeable=self.personality['agreeableness'];sensitive=self.personality['neuroticism']
        rate=(.18+.3*agreeable) if event.benefit>0 else (.18+.3*sensitive+.15*(1-agreeable))
        attitude+=weight*rate*(event.benefit-attitude)
        if weight:
            self.entries.pop(event.target,None)
            self.entries[event.target]=dict(attitude=attitude,evidence=min(64,e['evidence']+weight),tick=event.tick)
            while len(self.entries)>self.capacity:self.entries.popitem(last=False)
        self.events[event.event_id]=event.tick
        while len(self.events)>4*self.capacity:self.events.popitem(last=False)
        self.last_tick=event.tick
        return self.status(event.target,event.tick)

    def status(self,target,tick):
        identifier(target)
        if type(tick) is not int or tick<0 or tick<self.last_tick:raise ValueError('current observed tick required')
        e=self.entries.get(target)
        if e is None:return dict(attitude=0.,confidence=0.,evidence=0.)
        decay=2**(-(tick-e['tick'])/self.half_life)
        evidence=e['evidence']*decay
        return dict(attitude=float(e['attitude']*decay),confidence=float(evidence/(evidence+3)),evidence=float(evidence))

    def relationship_adjustment(self,status,weight):
        return weight*status['attitude']*status['confidence']

    def scores(self,ids,targets,bindings,tick):
        if not isinstance(bindings,dict) or set(bindings)-set(ids):raise ValueError('bindings must belong to available roots')
        score=np.zeros(len(ids));audit={}
        agreeable=self.personality['agreeableness'];v=self.values
        weight=1.2*(v['benevolence']+.3*agreeable+.5*v['power'])
        for j,key in enumerate(ids):
            b=bindings.get(key)
            if b is None:continue
            if not isinstance(b,SocialBinding) or targets[j]!=b.target:raise ValueError('same action recipient required')
            e=self.status(b.target,tick)
            # Universal concern reduces the effect of a grudge when its recipient
            # is in distress. It does not make all help costless or obligatory.
            concern=1-b.distress*v['universalism'] if e['attitude']<0 and b.support>0 else 1.
            score[j]=np.clip(self.relationship_adjustment(e,weight)*b.support*concern,-1.5,1.5)
            audit[key]=dict(target=b.target,**e,adjustment=float(score[j]))
        return score,audit

    def record(self):
        return dict(version='social-memory-v1',owner=self.owner,personality=self.personality.copy(),values=self.values.copy(),
            capacity=self.capacity,half_life=self.half_life,last_tick=self.last_tick,
            entries=[dict(target=k,**v) for k,v in self.entries.items()],events=list(self.events.items()))

    @classmethod
    def from_record(cls,context,record):
        m=cls(context,record['capacity'],record['half_life'])
        if record['version']!='social-memory-v1' or any(record[k]!=getattr(m,k) for k in ('owner','personality','values')):
            raise ValueError('social checkpoint owner/personality mismatch')
        if type(record['last_tick']) is not int or not -1<=record['last_tick']<2**63-1:raise ValueError('invalid social clock')
        m.last_tick=record['last_tick']
        if len(record['entries'])>m.capacity or len(record['events'])>4*m.capacity:raise ValueError('social checkpoint capacity exceeded')
        for row in record['entries']:
            identifier(row['target']);number(row['attitude'],-1,1);number(row['evidence'],0,64)
            if row['target'] in m.entries or type(row['tick']) is not int or not 0<=row['tick']<=m.last_tick:
                raise ValueError('invalid social checkpoint entry')
            m.entries[row['target']]={k:row[k] for k in ('attitude','evidence','tick')}
        for key,tick in record['events']:
            identifier(key)
            if key in m.events or type(tick) is not int or not 0<=tick<=m.last_tick:raise ValueError('invalid social event checkpoint')
            m.events[key]=tick
        return m


def appraised(batch,base,appraisal,purpose,max_regret,stochastic):
    """Change preferences, not effects; prevent socially induced purpose collapse.

    The baseline's deliberate sacrifice remains permitted. Social pressure can
    improve on it or spend a small game-selected regret, but cannot worsen an
    already large baseline sacrifice. This is not a global win-first override.
    """
    bias=np.asarray(appraisal,dtype=float);number(max_regret,0,2)
    if bias.shape!=base.scores.shape or not np.isfinite(bias).all() or (abs(bias)>1.5).any():
        raise ValueError('bounded per-root social appraisal required')
    purpose=np.einsum('nak,nak->na',batch.probability,batch.effects[:,:,:,0]) if purpose is None else np.asarray(purpose,dtype=float)
    if purpose.shape!=bias.shape or not np.isfinite(purpose).all() or (abs(purpose)>1).any():
        raise ValueError('bounded same-target purpose evidence required')
    if not bias.any():return base
    rows=np.arange(len(bias));best=np.max(np.where(base.eligible,purpose,-np.inf),1)
    floor=np.minimum(purpose[rows,base.action],best-max_regret)
    eligible=base.eligible & (purpose>=floor[:,None]-1e-9)
    scores=base.scores+bias
    maximum=np.max(np.where(eligible,scores,-np.inf),1)
    near=eligible & (scores>=maximum[:,None]-.025)
    weights=np.where(near,np.exp(np.clip((scores-maximum[:,None])/.008,-80,0)),0)
    weights/=weights.sum(1,keepdims=True)
    chosen=np.minimum((np.cumsum(weights,axis=1)<batch.rng[:,1,None]).sum(1),bias.shape[1]-1) if stochastic else np.argmax(np.where(eligible,scores,-np.inf),1)
    return replace(base,action=chosen,scores=scores,eligible=eligible)


class SocialPopulation(Population):
    """Connected batch-reflex path; games reveal feedback after resolution."""
    def __init__(self,contexts,policy=None,capacity=64,half_life=24.):
        super().__init__(contexts,policy)
        self.social=[SocialMemory(c,capacity,half_life) for c in contexts]
        self.social_audit=None

    def step(self,stochastic=True,*,bindings,purpose=None,max_social_regret=.15,**updates):
        if len(bindings)!=len(self.social):raise ValueError('one binding set per actor required')
        scores=np.zeros_like(self.batch.legal,dtype=float);audit=[]
        for i,m in enumerate(self.social):
            score,row=m.scores(self.batch.ids[i],self.batch.targets[i],bindings[i],int(self.ticks[i]))
            scores[i,:len(score)]=score;audit.append(row)
        d=super().step(stochastic,appraisal=scores,purpose=purpose,max_social_regret=max_social_regret,**updates)
        self.social_audit=audit
        return d

    def observe_batch(self,events):
        """Validate a complete feedback batch before any relationship changes."""
        if len(events)!=len(self.social):raise ValueError('one observed event list per actor required')
        memories=copy.deepcopy(self.social);result=[]
        for i,(m,rows) in enumerate(zip(memories,events)):
            updates=[]
            for event in rows:
                if not isinstance(event,SocialEvent) or event.tick>=int(self.ticks[i]):raise ValueError('future social feedback forbidden')
                updates.append(m.observe(event))
            result.append(updates)
        self.social=memories
        return result
