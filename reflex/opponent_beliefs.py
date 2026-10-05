"""Bounded categorical behavior hypotheses, external to personality/Policy.

The game supplies pre-action hypothetical distributions and a later revealed
action. Discounted log-likelihood weights are a design model, not psychological
type identification. Predictive trust is heuristic, not calibrated probability.
"""
from collections import deque
from dataclasses import dataclass
import math


def distributions(models,names,smoothing):
    if set(models)!=set(names): raise ValueError('one distribution per hypothesis required')
    actions=tuple(sorted(models[names[0]]))
    if not actions: raise ValueError('legal actions required')
    matrix=[]
    for name in names:
        row=models[name]
        if set(row)!=set(actions): raise ValueError('all hypotheses must cover identical legal actions')
        values=[row[a] for a in actions]
        if any(isinstance(p,bool) or not isinstance(p,(float,int)) or not math.isfinite(p) or p<0 for p in values):
            raise ValueError('finite nonnegative probabilities required')
        if abs(sum(values)-1)>1e-8: raise ValueError('probabilities must sum to one')
        matrix.append(tuple((1-smoothing)*p+smoothing/len(actions) for p in values))
    return actions,tuple(matrix)


@dataclass(frozen=True)
class BeliefSnapshot:
    names: tuple
    weights: tuple
    observations: int
    confidence: float
    smoothing: float
    surprise: float = 0.
    responsive: bool = True

    def predict(self,models):
        actions,matrix=distributions(models,self.names,self.smoothing)
        # Broad fallback always remains in the forecast, including when one
        # named hypothesis dominates. Confidence is NOT a posterior type label.
        probs=[(1-self.confidence)/len(actions)+self.confidence*sum(w*r[j] for w,r in zip(self.weights,matrix))
               for j in range(len(actions))]
        total=sum(probs)
        return dict(zip(actions,(p/total for p in probs)))

    def record(self):
        return dict(weights=dict(zip(self.names,self.weights)),observations=self.observations,
                    confidence=self.confidence,confidence_kind='heuristic predictive trust, not calibrated',smoothing=self.smoothing,
                    surprise=self.surprise,responsive=self.responsive)


class HypothesisTracker:
    def __init__(self,names,retention=.9,smoothing=.08,window=6,responsive=True):
        if not 2<=len(names)<=16 or len(set(names))!=len(names) or any(not isinstance(n,str) or not n for n in names):
            raise ValueError('2..16 distinct hypothesis names required')
        if not .5<=retention<=1 or not 0<smoothing<1 or type(window) is not int or not 2<=window<=64:
            raise ValueError('invalid bounded inference configuration')
        if type(responsive) is not bool: raise ValueError('responsive must be boolean')
        self.names=tuple(names); self.retention=retention; self.smoothing=smoothing; self.responsive=responsive
        self.surprise=0.
        self.logs=[0.]*len(names); self.observations=0; self.lifts=deque(maxlen=window)
        self.ids=deque(maxlen=64)

    def snapshot(self):
        top=max(self.logs); unscaled=[math.exp(x-top) for x in self.logs]; total=sum(unscaled)
        weights=tuple(x/total for x in unscaled)
        entropy=-sum(w*math.log(w) for w in weights if w)/math.log(len(weights))
        # At least three informative observations; forced moves supply none.
        evidence=0. if self.observations<3 else self.observations/(self.observations+3)
        quality=min(1.,math.exp(sum(self.lifts)/len(self.lifts))) if self.lifts else 1.
        if self.responsive: quality*=math.exp(-self.surprise)
        confidence=min(.9,evidence*max(0.,1-entropy)*quality)
        return BeliefSnapshot(self.names,weights,self.observations,float(confidence),self.smoothing,self.surprise,self.responsive)

    def observe(self,models,action,observation_id):
        actions,matrix=distributions(models,self.names,self.smoothing)
        if action not in actions: raise ValueError('revealed action must have been legal before reveal')
        if not isinstance(observation_id,str) or not observation_id: raise ValueError('observation ID required')
        if observation_id in self.ids: raise ValueError('duplicate public observation')
        before=self.snapshot(); forecast=before.predict(models)
        predictive=forecast[action]; uniform=1/len(actions)
        self.ids.append(observation_id)
        effective_retention=None
        if len(actions)>1:
            j=actions.index(action)
            lift=math.log(predictive/uniform); shock=max(0.,-lift)
            self.surprise=min(8.,.5*self.surprise+shock) if self.responsive else 0.
            # Unexpected at less than half the uniform probability: weaken old
            # evidence immediately, but do not declare which new model is true.
            effective_retention=self.retention*(.25 if self.responsive and shock>math.log(2) else 1.)
            self.logs=[effective_retention*x+math.log(row[j]) for x,row in zip(self.logs,matrix)]
            # Remove a harmless common log offset to keep long-running state bounded.
            peak=max(self.logs); self.logs=[max(-80.,x-peak) for x in self.logs]
            self.observations+=1; self.lifts.append(lift)
        return dict(observation_id=observation_id,action=action,forced=len(actions)==1,
                    predicted_probability=predictive,uniform_probability=uniform,
                    log_loss=-math.log(predictive),uniform_log_loss=-math.log(uniform),
                    effective_retention=effective_retention,before=before.record(),after=self.snapshot().record())
