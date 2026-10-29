"""Research: retain supplied priors and transfer shapes inside unknown models.

One half of the segment prior remains on supplied labeled models, as before.
The other half averages a fresh Dirichlet law and permuted supplied shapes.
An empty library or alphabet above four retains the original model exactly.
This is a prior hypothesis, not evidence of the true unknown response law.
"""
from functools import lru_cache
import math
import numpy as np
from reflex.core import digest
from reflex.segmented_prediction import SegmentedEventLearner, segment_prediction, _normalize_logs, HAZARDS
from tools.symmetric_prediction import complete_templates


@lru_cache(maxsize=2048)
def weighted_posterior(sequence, libraries, alphabet, prior):
    if len(sequence) == 1:
        weights=np.zeros((len(HAZARDS),0,len(prior)))
        counts=np.zeros((0,alphabet));evidence=np.zeros(len(HAZARDS))
    else:
        weights,counts,evidence=weighted_posterior(sequence[:-1],libraries[:-1],alphabet,prior)
    z=sequence[-1];nlib=len(prior)-1
    likelihood=np.empty((len(counts),nlib+1))
    likelihood[:,:nlib]=np.array(libraries[-1])[:,z]
    likelihood[:,-1]=(counts[:,z]+.5)/(counts.sum(axis=1)+.5*alphabet)
    reset=np.array(prior)*np.array([row[z] for row in libraries[-1]]+[1/alphabet])
    h=np.array(HAZARDS)[:,None,None]
    masses=np.concatenate(((h if len(sequence)>1 else np.ones_like(h))*reset[None,None,:],
                            (1-h)*weights*likelihood[None,:,:]),axis=1)
    norm=masses.sum(axis=(1,2));weights=masses/norm[:,None,None]
    evidence=evidence+np.log(norm)
    counts=np.vstack((np.zeros((1,alphabet)),counts));counts[:,z]+=1.
    for a in (weights,counts,evidence):a.flags.writeable=False
    return weights,counts,evidence


@lru_cache(maxsize=4096)
def transfer_prediction(sequence,libraries,current,alphabet):
    if not current or alphabet>4:
        return segment_prediction(sequence,libraries,current,alphabet)
    n=len(current);templates=complete_templates(current,alphabet)
    prior=(.5/n,)*n+(.25/len(templates),)*len(templates)+(.25,)
    augmented=tuple((*row,*complete_templates(row,alphabet)) for row in libraries)
    extended=(*current,*templates)
    base=np.array((*extended,(1/alphabet,)*alphabet))
    initial=np.array(prior)@base
    if not sequence:return tuple(initial),0.,(1/len(HAZARDS),)*len(HAZARDS)
    weights,counts,evidence=weighted_posterior(sequence,augmented,alphabet,prior)
    forecasts=[]
    for hazard,weight in zip(HAZARDS,weights):
        p=hazard*initial
        p+=(1-hazard)*(weight[:,:-1].sum(axis=0)@np.array(extended))
        p+=(1-hazard)*(weight[:,-1]@((counts+.5)/(counts.sum(axis=1)[:,None]+.5*alphabet)))
        forecasts.append(p)
    posterior=_normalize_logs(evidence);p=posterior@np.array(forecasts)
    logp=max(evidence)+math.log(sum(math.exp(v-max(evidence)) for v in evidence)/len(evidence))
    return tuple(float(x) for x in p/p.sum()),logp,tuple(posterior)


class TemplateTransferLearner(SegmentedEventLearner):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.identity=digest(['unknown-shape-transfer-v1',self.identity])

    def predict(self,memory,context,legal,models):
        self._check(memory);current=self._inputs(context,legal,models)
        predictions=[];weights=[]
        for mask in self.masks:
            groups={}
            for row in memory.history:
                key=(row.legal,tuple(row.context[i] for i in mask))
                groups.setdefault(key,[]).append(row)
            target=(legal,tuple(context[i] for i in mask));target_p=None
            log_weight=-len(mask)*math.log(2)
            for key,rows in groups.items():
                labels={v:i for i,v in enumerate(key[0])}
                sequence=tuple(labels[r.outcome] for r in rows)
                libraries=tuple(r.models for r in rows)
                supplied=current if key==target else rows[-1].models
                p,evidence,_=transfer_prediction(sequence,libraries,supplied,len(key[0]))
                log_weight+=evidence
                if key==target:target_p=p
            if target_p is None:target_p=transfer_prediction((),(),current,len(legal))[0]
            predictions.append(target_p);weights.append(log_weight)
        p=_normalize_logs(weights)@np.array(predictions)
        return dict(zip(legal,(float(x) for x in p/p.sum())))

    def persistent_continuation(self,*args,**kwargs):
        raise ValueError('shape transfer requires its declared posterior continuation')
