"""Bounded Bayesian segment prediction from public categorical observations.

Within a segment, the response is either a supplied conditional hypothesis or
an unknown categorical law with a symmetric Dirichlet prior. Segment lengths,
including a stationary hypothesis, compete by marginal likelihood. No revealed
regime, winning action or personality coefficient is available to this model.
"""
from dataclasses import dataclass
from functools import lru_cache
from collections import OrderedDict
from threading import Lock
from itertools import combinations
import math
import numpy as np
from .core import digest
from .adaptive_dynamics import _labels,_copy_predictive_record
from .belief_planning import _probabilities,UnsupportedObservation


HAZARDS=(0.,1/8,1/32,1/128)
_PREFIXES=OrderedDict()
_PREFIX_LOCK=Lock()


def _segment_posterior(sequence,libraries,alphabet):
    # Imagined futures share long factual prefixes. Cache immutable sufficient
    # statistics, never an actor or a mutable memory. Iteration avoids a stack
    # limit at the maximum retained history; all hazards advance together.
    nlib=len(libraries[0]) if libraries else 0
    end=len(sequence);start=end;cached=None
    with _PREFIX_LOCK:
        while start and (sequence[:start],libraries[:start],alphabet) not in _PREFIXES:start-=1
        if start:
            key=(sequence[:start],libraries[:start],alphabet)
            cached=_PREFIXES[key];_PREFIXES.move_to_end(key)
    if cached is not None:weights,counts,logp=cached
    else:
        weights=np.zeros((len(HAZARDS),0,nlib+1));counts=np.zeros((0,alphabet));logp=np.zeros(len(HAZARDS))
    prior=np.array(([.5/nlib]*nlib+[.5]) if nlib else [1.]);h=np.array(HAZARDS)[:,None,None]
    for i in range(start,end):
        z=sequence[i];likelihood=np.empty((len(counts),nlib+1))
        if nlib:likelihood[:,:nlib]=np.array(libraries[i])[:,z]
        likelihood[:,-1]=(counts[:,z]+.5)/(counts.sum(axis=1)+.5*alphabet)
        reset=prior*np.array([row[z] for row in libraries[i]]+[1/alphabet])
        masses=np.concatenate(((h if i else np.ones_like(h))*reset[None,None,:],(1-h)*weights*likelihood[None,:,:]),axis=1)
        norm=masses.sum(axis=(1,2));logp=logp+np.log(norm);weights=masses/norm[:,None,None]
        counts=np.vstack((np.zeros((1,alphabet)),counts));counts[:,z]+=1.
        for a in (weights,counts,logp):a.flags.writeable=False
        key=(sequence[:i+1],libraries[:i+1],alphabet)
        with _PREFIX_LOCK:
            _PREFIXES[key]=(weights,counts,logp);_PREFIXES.move_to_end(key)
            if len(_PREFIXES)>256:_PREFIXES.popitem(last=False)
    return weights,counts,logp


def _normalize_logs(values):
    a=np.asarray(values,float);p=np.exp(a-np.max(a));return p/p.sum()


@lru_cache(maxsize=8192)
def segment_prediction(sequence,libraries,current,alphabet):
    """Exact finite-history mixture; the caller controls the retained window.

    Returns next probabilities, history log evidence and hazard posterior.
    Unlike fixed-share expert weights, these are probabilities of the declared
    segment models, conditional on their prior and retained observations.
    A fresh symmetric segment may explain events absent from every library.
    """
    nlib=len(current);prior=np.array(([.5/nlib]*nlib+[.5]) if nlib else [1.])
    base=np.array((*current,tuple([1/alphabet]*alphabet)))
    predictions=[]
    weights_all,counts,evidence=_segment_posterior(sequence,libraries,alphabet)
    for hazard,weights in zip(HAZARDS,weights_all):
        if not len(weights):p=prior@base
        else:
            p=hazard*(prior@base)
            if nlib:p+=(1-hazard)*(weights[:,:nlib].sum(axis=0)@np.array(current))
            p+=(1-hazard)*(weights[:,-1]@((counts+.5)/(counts.sum(axis=1)[:,None]+.5*alphabet)))
        predictions.append(p)
    posterior=_normalize_logs(evidence)
    probability=posterior@np.array(predictions)
    log_evidence=max(evidence)+math.log(sum(math.exp(v-max(evidence)) for v in evidence)/len(evidence))
    return tuple(float(v) for v in probability/probability.sum()),log_evidence,tuple(float(v) for v in posterior)


@lru_cache(maxsize=8192)
def persistent_prediction(root,libraries,future,future_libraries,current,alphabet):
    """Integrate uncertainty in the next regime, then hold its law constant.

    Changes in the factual root still use the segment model. Only continuation
    after that root assumes persistence. This separates uncertain parameters
    from a prior on future unobserved regime changes. No true law is supplied.
    """
    nlib=len(current);prior=np.array(([.5/nlib]*nlib+[.5]) if nlib else [1.])
    if root:
        weights,counts,evidence=_segment_posterior(root,libraries,alphabet)
        hp=_normalize_logs(evidence);h=np.array(HAZARDS)[:,None,None]
        weights=np.sum(hp[:,None,None]*np.concatenate((h*prior[None,None,:],(1-h)*weights),axis=1),axis=0)
        counts=np.vstack((np.zeros((1,alphabet)),counts))
        logp=max(evidence)+math.log(sum(math.exp(v-max(evidence)) for v in evidence)/len(evidence))
    else:
        weights=prior[None,:];counts=np.zeros((1,alphabet));logp=0.
    for i,z in enumerate(future):
        likelihood=np.empty_like(weights)
        if nlib:likelihood[:,:nlib]=np.array(future_libraries[i])[:,z]
        likelihood[:,-1]=(counts[:,z]+.5)/(counts.sum(axis=1)+.5*alphabet)
        weights=weights*likelihood;norm=weights.sum();weights/=norm;logp+=math.log(norm)
        counts=counts.copy();counts[:,z]+=1.
    p=weights[:,-1]@((counts+.5)/(counts.sum(axis=1)[:,None]+.5*alphabet))
    if nlib:p+=weights[:,:nlib].sum(axis=0)@np.array(current)
    return tuple(float(v) for v in p/p.sum()),logp


@dataclass(frozen=True)
class SegmentEvidence:
    context: tuple
    legal: tuple
    models: tuple
    outcome: str

    def __post_init__(self):
        safe=(type(self.context) is tuple and type(self.legal) is tuple and all(type(x) is str for x in self.context+self.legal)
            and type(self.outcome) is str and type(self.models) is tuple and
            all(type(row) is tuple and all(type(p) in (int,float) for p in row) for row in self.models))
        object.__setattr__(self,'_shareable',safe)
        valid=(safe and len(self.context)<=4 and all(0<len(v)<=128 for v in self.context) and
            _labels(self.legal,256) and bool(self.legal) and tuple(sorted(self.legal))==self.legal and
            self.outcome in self.legal and len(self.models)<=16 and
            all(len(row)==len(self.legal) and _probabilities(row) for row in self.models))
        object.__setattr__(self,'_valid',valid)

    __deepcopy__=_copy_predictive_record


@dataclass(frozen=True)
class SegmentMemory:
    identity: str
    history: tuple=()
    observations: int=0

    def __post_init__(self):
        object.__setattr__(self,'_shareable',type(self.identity) is str and type(self.observations) is int and
            type(self.history) is tuple and all(type(x) is SegmentEvidence and x._shareable for x in self.history))

    __deepcopy__=_copy_predictive_record


class SegmentedEventLearner:
    """Alternative to rolling-window/fixed-share model competition.

    Observable feature subsets compete by evidence of independent strata.
    All calculations use at most the retained window (default 128 events).
    Forgetting beyond that boundary is a modeling approximation, not a claim
    to retain the posterior of an unlimited history or discover new rules.
    """
    def __init__(self,*,owner,model_id,factors=(),names=(),window=128):
        self.factors,self.names=tuple(factors),tuple(names);self.window=window
        if not _labels(self.factors,4) or not _labels(self.names,16) or type(window) is not int or not 2<=window<=512:
            raise ValueError('bounded feature/library definitions and retained history required')
        if not all(isinstance(x,str) and x for x in (owner,model_id)):raise ValueError('identified public observer required')
        self.masks=tuple(c for n in range(len(self.factors)+1) for c in combinations(range(len(self.factors)),n))
        self.identity=digest(['segmented-event-v1',owner,model_id,self.factors,self.names,window,HAZARDS])

    def start(self):return SegmentMemory(self.identity)

    def _inputs(self,context,legal,models):
        if (not isinstance(context,tuple) or len(context)!=len(self.factors) or
            any(type(v) is not str or not 0<len(v)<=128 for v in context) or
            not _labels(legal,256) or not legal or tuple(sorted(legal))!=legal or
            not isinstance(models,dict) or set(models)!=set(self.names)):
            raise ValueError('bounded public stratum and complete model library required')
        rows=[]
        for name in self.names:
            row=models[name]
            if not isinstance(row,dict) or set(row)!=set(legal) or not _probabilities(list(row.values())):
                raise ValueError('complete normalized conditional model required')
            rows.append(tuple(float(row[k]) for k in legal))
        return tuple(rows)

    def _check(self,memory):
        if (not isinstance(memory,SegmentMemory) or memory.identity!=self.identity or type(memory.observations) is not int or
            memory.observations<0 or not isinstance(memory.history,tuple) or len(memory.history)!=min(self.window,memory.observations)):
            raise ValueError('matching complete segmented observation history required')
        for row in memory.history:
            if (not isinstance(row,SegmentEvidence) or not row._valid or len(row.context)!=len(self.factors)
                or len(row.models)!=len(self.names)):
                raise ValueError('valid actual public evidence required')

    def predict(self,memory,context,legal,models):
        self._check(memory);current=self._inputs(context,legal,models);predictions=[];log_weights=[]
        for mask in self.masks:
            groups={}
            for row in memory.history:
                key=(row.legal,tuple(row.context[i] for i in mask));groups.setdefault(key,[]).append(row)
            target=(legal,tuple(context[i] for i in mask));target_p=None;log_weight=-len(mask)*math.log(2)
            for key,rows in groups.items():
                labels={v:i for i,v in enumerate(key[0])}
                seq=tuple(labels[r.outcome] for r in rows);libs=tuple(r.models for r in rows)
                p,evidence,_=segment_prediction(seq,libs,current if key==target else rows[-1].models,len(key[0]))
                log_weight+=evidence
                if key==target:target_p=p
            if target_p is None:target_p=segment_prediction((),(),current,len(legal))[0]
            predictions.append(target_p);log_weights.append(log_weight)
        p=_normalize_logs(log_weights)@np.array(predictions)
        return dict(zip(legal,(float(v) for v in p/p.sum())))

    def observe(self,memory,context,legal,models,outcome):
        self._check(memory);rows=self._inputs(context,legal,models)
        if outcome not in legal:raise UnsupportedObservation('response outside known rule alphabet')
        history=(*memory.history,SegmentEvidence(context,legal,rows,outcome))[-self.window:]
        return SegmentMemory(self.identity,history,memory.observations+1)

    def persistent_continuation(self,memory,future,context,legal,models):
        self._check(memory);current=self._inputs(context,legal,models);predictions=[];log_weights=[]
        if (not isinstance(future,tuple) or any(not isinstance(r,SegmentEvidence) or not r._valid or
                len(r.context)!=len(self.factors) or len(r.models)!=len(self.names) for r in future)):
            raise ValueError('valid public continuation evidence required')
        for mask in self.masks:
            groups={}
            for channel,history in enumerate((memory.history,future)):
                for row in history:
                    key=(row.legal,tuple(row.context[i] for i in mask))
                    groups.setdefault(key,[[],[]])[channel].append(row)
            target=(legal,tuple(context[i] for i in mask));target_p=None;log_weight=-len(mask)*math.log(2)
            for key,(past,following) in groups.items():
                labels={v:i for i,v in enumerate(key[0])};rows=past+following
                p,evidence=persistent_prediction(tuple(labels[r.outcome] for r in past),tuple(r.models for r in past),
                    tuple(labels[r.outcome] for r in following),tuple(r.models for r in following),
                    current if key==target else rows[-1].models,len(key[0]))
                log_weight+=evidence
                if key==target:target_p=p
            if target_p is None:target_p=persistent_prediction((),(),(),(),current,len(legal))[0]
            predictions.append(target_p);log_weights.append(log_weight)
        p=_normalize_logs(log_weights)@np.array(predictions)
        return dict(zip(legal,(float(v) for v in p/p.sum())))

    def diagnostics(self,memory):
        self._check(memory)
        return dict(observations=memory.observations,retained=len(memory.history),
            interpretation='segmented marginal likelihood on retained public history; no true-world calibration guarantee')

    def record(self,memory):
        self._check(memory)
        return dict(format='segmented-events-v1',identity=memory.identity,observations=memory.observations,
            history=[dict(context=list(r.context),legal=list(r.legal),models=[list(v) for v in r.models],outcome=r.outcome) for r in memory.history])

    def restore(self,record):
        if set(record)!={'format','identity','observations','history'} or record['format']!='segmented-events-v1':
            raise ValueError('complete segmented checkpoint required')
        memory=SegmentMemory(record['identity'],tuple(SegmentEvidence(tuple(r['context']),tuple(r['legal']),tuple(tuple(v) for v in r['models']),r['outcome']) for r in record['history']),record['observations'])
        self._check(memory)
        if self.record(memory)!=record:raise ValueError('lossless segmented checkpoint required')
        return memory
