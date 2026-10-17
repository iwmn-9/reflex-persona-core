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


@dataclass(frozen=True)
class TailDistribution:
    trajectories: tuple
    purpose_estimates: tuple


class PolicyTail:
    """Continue a resolved owner through its ordinary feedback lifecycle.

    `evaluate` is the legacy scalar-purpose estimate. `trajectories` also returns
    the correlated preference flows and can invoke an explicitly bounded future
    controller. Without that callback, continuation uses the learning reflex
    policy. Unsettled samples retain the adapter's declared running proxy.
    """
    def __init__(self,*,observe,advance,terminal,assess,request_for,feedback,model_id,target,seeds=(701,702,703,704),steps=16,record=None,cache_entries=16384,owner_advance=None,
                 future_planner=None,future_planner_id=None,future_planner_limit=1,
                 purpose_adjustment=None,purpose_adjustment_id=None):
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
        if owner_advance is not None and not callable(owner_advance):raise ValueError('explicit owner-conditioned advance required')
        self.owner_advance=owner_advance
        self.identity=digest([self.identity,'owner-conditioned' if owner_advance else 'fixed-kernel'])
        self.simulated_steps=0
        if future_planner is not None and (not callable(future_planner) or not isinstance(future_planner_id,str) or not future_planner_id):
            raise ValueError('identified finite future controller required')
        if type(future_planner_limit) is not int or not 1<=future_planner_limit<=256:
            raise ValueError('bounded future planning calls required')
        self.future_planner=future_planner;self.future_planner_limit=future_planner_limit
        self.future_calls=0
        self.identity=digest([self.identity,future_planner_id,future_planner_limit])
        if purpose_adjustment is not None and (not callable(purpose_adjustment) or not isinstance(purpose_adjustment_id,str) or not purpose_adjustment_id):
            raise ValueError('identified conditional-mean adjustment required')
        self.purpose_adjustment=purpose_adjustment
        if purpose_adjustment is not None:self.identity=digest([self.identity,purpose_adjustment_id])

    def trajectories(self,world,actor):
        return list(self.distribution(world,actor).trajectories)

    def distribution(self,world,actor):
        """Full sampled returns from the learning owner, including preferences.

        No scalar endpoint cache is used: it cannot reconstruct correlated
        needs/value/cost paths. Hypothetical observations stay in each clone.
        """
        from .decision_loop import DecisionLoop
        from .model_observer import resolve_feedback
        rows=[];estimates=[]
        if actor.pending is not None:raise ValueError('resolved frontier owner required')
        for seed in self.seeds:
            w=copy.deepcopy(world);own=copy.deepcopy(actor);flows=[];actions=[];calls=0;adjustment=0.
            for step in range(self.steps):
                if self.terminal(w):break
                c=self.observe(copy.deepcopy(w),copy.deepcopy(own.state));req=self.request_for(c)
                if req.reader is not None or req.value_proposal is not None or (own.predictor is not None and self.owner_advance is None):
                    raise ValueError('tail requires explicit continuation without advice overrides')
                planner=None
                if self.future_planner is not None and calls<self.future_planner_limit:
                    planner=self.future_planner(copy.deepcopy(own),copy.deepcopy(req),copy.deepcopy(w))
                    if planner is not None:calls+=1;self.future_calls+=1
                # Kernels receive the same resolved pre-choice owner as a tree
                # root, not an actor carrying an unresolved hypothetical ticket.
                model_owner=copy.deepcopy(own) if self.owner_advance is not None else None
                result=DecisionLoop.decide_batch([(own,req)],False,planner)[0];key=result['decision']['action_id']
                after,flow=(self.advance(copy.deepcopy(w),key,seed) if self.owner_advance is None else
                    self.owner_advance(copy.deepcopy(w),key,seed,model_owner,copy.deepcopy(result['context'])))
                if self.purpose_adjustment is not None:
                    delta=self.purpose_adjustment(copy.deepcopy(w),key,model_owner,copy.deepcopy(result['context']),copy.deepcopy(after))
                    if isinstance(delta,bool) or not isinstance(delta,(int,float)) or not math.isfinite(delta):
                        raise ValueError('finite conditional-mean correction required')
                    adjustment+=delta
                resolve_feedback(own,result['ticket'],self.feedback(w,after,key,result['context'],flow))
                self.simulated_steps+=1
                flows.append(copy.deepcopy(flow));actions.append(key);w=after
            g=Goal(**self.assess(w));ended=bool(self.terminal(w))
            if g.target!=self.target or ended!=(g.status!='running'):raise ValueError('same-target coherent tail endpoint required')
            rows.append((1/len(self.seeds),tuple(flows),tuple(actions),g.record(),ended))
            estimates.append(g.value+adjustment)
        return TailDistribution(tuple(rows),tuple(estimates))

    def evaluate(self,world,actor):
        from .closed_loop import reveal
        from .decision_loop import DecisionLoop
        if self.future_planner is not None or self.purpose_adjustment is not None:raise ValueError('future planning or mean correction requires complete preference trajectories')
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
                if req.reader is not None or req.value_proposal is not None or (own.predictor is not None and self.owner_advance is None):
                    raise ValueError('tail requires explicit continuation without advice overrides')
                model_owner=copy.deepcopy(own) if self.owner_advance is not None else None
                result=DecisionLoop.decide_batch([(own,req)],False)[0];key=result['decision']['action_id']
                after,flow=(self.advance(copy.deepcopy(w),key,seed) if self.owner_advance is None else
                    self.owner_advance(copy.deepcopy(w),key,seed,model_owner,copy.deepcopy(result['context'])))
                reveal(own,result['ticket'],self.feedback(w,after,key,result['context'],flow));w=after
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
