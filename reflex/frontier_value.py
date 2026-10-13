"""Optional endpoint estimates for a finite search, with explicit provenance.

These are estimates under a named continuation, not new factual experiences or
proof that a replanning controller will obtain the predicted result. Outcome
spread describes possible results; it is never used as model confidence.
"""
from dataclasses import dataclass
from collections import OrderedDict
import copy
import math
import numpy as np
from .core import digest
from .purpose_plan import Goal


@dataclass(frozen=True)
class EndpointEstimate:
    outcomes: tuple
    source: str
    settled_fraction: float

    def __post_init__(self):
        if not isinstance(self.outcomes,tuple) or not 1<=len(self.outcomes)<=256:
            raise ValueError('bounded endpoint distribution required')
        for row in self.outcomes:
            if not isinstance(row,tuple) or len(row)!=2:raise ValueError('probability/value pairs required')
            p,v=row
            if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in row) or not 0<p<=1 or not -1<=v<=1:
                raise ValueError('positive probabilities and bounded endpoint values required')
        if abs(sum(p for p,_ in self.outcomes)-1)>1e-8:raise ValueError('normalized endpoint distribution required')
        if not isinstance(self.source,str) or not self.source:raise ValueError('estimate provenance required')
        if isinstance(self.settled_fraction,bool) or not isinstance(self.settled_fraction,(int,float)) or not math.isfinite(self.settled_fraction) or not 0<=self.settled_fraction<=1:
            raise ValueError('explicit modeled settlement coverage required')


class PolicyTail:
    """Sample a resolved clone's reflex continuation beyond the tree frontier.

    No tree is grown in the tail. The entire actual owner lifecycle still runs.
    Unsettled samples retain the adapter's running proxy and are counted as such.
    Tail flows do NOT enter the tree's personality return: this is purpose-value
    completion only. The reference policy is not the future replanning policy.
    """
    def __init__(self,*,observe,advance,terminal,assess,request_for,feedback,model_id,target,seeds=(701,702,703,704),steps=16,record=None,cache_entries=16384):
        if any(not callable(x) for x in (observe,advance,terminal,assess,request_for,feedback)):
            raise ValueError('explicit public continuation callbacks required')
        if not isinstance(model_id,str) or not model_id or not isinstance(target,str) or not target:
            raise ValueError('model and purpose identities required')
        if type(steps) is not int or not 1<=steps<=256 or not isinstance(seeds,tuple) or not 1<=len(seeds)<=32 or any(type(s) is not int for s in seeds) or len(set(seeds))!=len(seeds):
            raise ValueError('finite tail and independent declared model seeds required')
        if (record is not None and not callable(record)) or type(cache_entries) is not int or not 1<=cache_entries<=65536:
            raise ValueError('complete public state record and bounded memoization required')
        self.observe=observe;self.advance=advance;self.terminal=terminal;self.assess=assess
        self.request_for=request_for;self.feedback=feedback;self.target=target;self.seeds=seeds;self.steps=steps
        self.identity=digest(['owned-reflex-purpose-tail-v1',model_id,target,seeds,steps])
        self.record=record;self.cache_entries=cache_entries;self.cache=OrderedDict();self.cache_hits=0

    def evaluate(self,world,actor):
        from .closed_loop import reveal
        from .decision_loop import DecisionLoop
        if actor.pending is not None:raise ValueError('resolved frontier owner required')
        values={};settled=0
        for seed in self.seeds:
            w=copy.deepcopy(world);own=copy.deepcopy(actor);visited=[];cached=None
            for step in range(self.steps):
                if self.terminal(w):break
                if self.record is not None:
                    stamp=digest([self.identity,self.record(w),own.record(),seed,self.steps-step])
                    if stamp in self.cache:
                        cached=self.cache[stamp];self.cache.move_to_end(stamp);self.cache_hits+=1;break
                    visited.append(stamp)
                c=self.observe(copy.deepcopy(w),copy.deepcopy(own.state));req=self.request_for(c)
                if req.reader is not None or req.value_proposal is not None or own.predictor is not None:
                    raise ValueError('tail requires explicit continuation without advice overrides')
                effective=own._prepare(req)['effective']
                result=DecisionLoop.decide_batch([(own,req)],False)[0];key=result['decision']['action_id']
                after,flow=self.advance(copy.deepcopy(w),key,seed)
                reveal(own,result['ticket'],self.feedback(w,after,key,effective,flow));w=after
            g,ended=cached if cached is not None else (Goal(**self.assess(w)),bool(self.terminal(w)))
            if g.target!=self.target or ended!=(g.status!='running'):raise ValueError('same-target coherent tail endpoint required')
            for stamp in reversed(visited):
                self.cache[stamp]=(g,ended);self.cache.move_to_end(stamp)
                if len(self.cache)>self.cache_entries:self.cache.popitem(last=False)
            values[g.value]=values.get(g.value,0)+1;settled+=ended
        return EndpointEstimate(tuple((n/len(self.seeds),v) for v,n in sorted(values.items())),self.identity,settled/len(self.seeds))


class TerminalMemory:
    """Nearest distinct-episode terminal returns under a fixed reference policy.

    Inputs are public numerical features supplied by the integration. Fitting
    must use resolved actual training episodes, not forecasts or evaluation
    futures. One nearest row per episode prevents long trajectories from voting
    repeatedly. Support is a declared geometric heuristic, not a confidence
    bound or automatic transfer to another rule set/policy.
    """
    def __init__(self,features,values,episodes,*,reference,neighbors=16,max_distance=2.):
        x=np.asarray(features,float);y=np.asarray(values,float)
        if x.ndim!=2 or not 1<=x.shape[1]<=256 or not len(x) or y.shape!=(len(x),) or len(episodes)!=len(x) or not np.isfinite(x).all() or np.any(np.abs(x)>1e6) or not np.isfinite(y).all() or np.any(np.abs(y)>1):
            raise ValueError('finite public features and resolved terminal labels required')
        if not all(isinstance(e,str) and e for e in episodes) or not isinstance(reference,str) or not reference:
            raise ValueError('episode and continuation provenance required')
        if type(neighbors) is not int or not 1<=neighbors<=256 or isinstance(max_distance,bool) or not math.isfinite(max_distance) or max_distance<=0:
            raise ValueError('bounded neighbors and positive support distance required')
        unique,inverse,count=np.unique(episodes,return_inverse=True,return_counts=True)
        if len(unique)<neighbors:raise ValueError('enough independent training episodes required')
        if any(np.ptp(y[inverse==i])>0 for i in range(len(unique))):
            raise ValueError('one resolved terminal result per episode required')
        weights=1/count[inverse];mean=np.average(x,axis=0,weights=weights)
        scale=np.maximum(np.sqrt(np.average((x-mean)**2,axis=0,weights=weights)),.05)
        self._record=dict(version='episode-terminal-memory-v1',features=x.tolist(),values=y.tolist(),episodes=list(episodes),reference=reference,neighbors=neighbors,max_distance=float(max_distance))
        self.identity=digest(self._record);self.x=(x-mean)/scale;self.y=y.copy();self.mean=mean;self.scale=scale;self.episodes=np.array(episodes)
        self.neighbors=neighbors;self.max_distance=max_distance
        for a in (self.x,self.y,self.mean,self.scale,self.episodes):a.flags.writeable=False

    def record(self):return copy.deepcopy(self._record)

    @classmethod
    def from_record(cls,record):
        if set(record)!={'version','features','values','episodes','reference','neighbors','max_distance'} or record['version']!='episode-terminal-memory-v1':
            raise ValueError('complete terminal memory required')
        return cls(**{k:v for k,v in record.items() if k!='version'})

    def predict(self,features):
        x=np.asarray(features,float)
        if x.shape!=self.mean.shape or not np.isfinite(x).all() or np.any(np.abs(x)>1e6):raise ValueError('same public feature schema required')
        z=(x-self.mean)/self.scale;dist=np.sqrt(np.mean((self.x-z)**2,axis=1))
        seen=set();indices=[]
        for i in np.argsort(dist,kind='stable'):
            if self.episodes[i] in seen:continue
            seen.add(self.episodes[i]);indices.append(i)
            if len(indices)==self.neighbors:break
        support=bool(dist[indices[-1]]<=self.max_distance and np.max(np.abs(z))<=4.)
        if not support:return None
        values,counts=np.unique(self.y[indices],return_counts=True)
        return EndpointEstimate(tuple((int(n)/len(indices),float(v)) for v,n in zip(values,counts)),self.identity,1.)


class LearnedFrontier:
    def __init__(self,memory,features,*,schema_id,reference):
        if not isinstance(memory,TerminalMemory) or not callable(features) or not isinstance(schema_id,str) or not schema_id or not isinstance(reference,str) or not reference:
            raise ValueError('frozen terminal memory and explicit feature schema required')
        self.memory=memory;self.features=features;self.reference=reference
        self.compatible=reference==memory._record['reference']
        self.identity=digest([memory.identity,schema_id,reference])

    def evaluate(self,world,actor):
        # Geometry cannot validate the meaning of features or outcomes. The
        # caller binds public rules, purpose, schema and continuation versions.
        if not self.compatible:return None
        return self.memory.predict(self.features(world,actor))
