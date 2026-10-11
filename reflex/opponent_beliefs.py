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
    evidence_confidence: float | None = None
    assumption_trust: float = 0.
    inertia: float = 0.
    evidence_weights: tuple | None = None
    evidence_alignment: float = 0.

    def predict(self,models):
        actions,matrix=distributions(models,self.names,self.smoothing)
        # Broad fallback always remains in the forecast, including when one
        # named hypothesis dominates. Confidence is NOT a posterior type label.
        probs=[(1-self.confidence)/len(actions)+self.confidence*sum(w*r[j] for w,r in zip(self.weights,matrix))
               for j in range(len(actions))]
        total=sum(probs)
        return dict(zip(actions,(p/total for p in probs)))

    def record(self):
        result=dict(weights=dict(zip(self.names,self.weights)),observations=self.observations,
                    confidence=self.confidence,confidence_kind='heuristic predictive trust, not calibrated',smoothing=self.smoothing,
                    surprise=self.surprise,responsive=self.responsive)
        if self.evidence_confidence is not None:
            result.update(evidence_confidence=self.evidence_confidence,assumption_trust=self.assumption_trust,
                inertia=self.inertia,evidence_weights=dict(zip(self.names,self.evidence_weights)),evidence_alignment=self.evidence_alignment,
                prediction_basis='subjective prior plus independent public-evidence tracker; confidence is not proof')
        return result


class HypothesisTracker:
    def __init__(self,names,retention=.9,smoothing=.08,window=6,responsive=True,*,inertia=0.,prior=None,assumption_trust=0.):
        if not 2<=len(names)<=16 or len(set(names))!=len(names) or any(not isinstance(n,str) or not n for n in names):
            raise ValueError('2..16 distinct hypothesis names required')
        if not .5<=retention<=1 or not 0<smoothing<1 or type(window) is not int or not 2<=window<=64:
            raise ValueError('invalid bounded inference configuration')
        if type(responsive) is not bool: raise ValueError('responsive must be boolean')
        self.names=tuple(names); self.retention=retention; self.smoothing=smoothing; self.responsive=responsive
        for v in (inertia,assumption_trust):
            if isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or not 0<=v<=1:
                raise ValueError('bounded belief disposition required')
        if assumption_trust and prior is None:raise ValueError('explicit prior for assumed trust required')
        if prior is not None:
            if set(prior)!=set(names) or any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or v<=0 for v in prior.values()) or abs(sum(prior.values())-1)>1e-8:
                raise ValueError('positive normalized subjective prior required')
        self.inertia=float(inertia);self.assumption_trust=float(assumption_trust)
        self.prior=None if prior is None else tuple(float(prior[n]) for n in names)
        self.surprise=0.
        self.logs=[0.]*len(names); self.observations=0; self.lifts=deque(maxlen=window)
        if self.prior is not None:
            base=[math.log(p) for p in self.prior];peak=max(base);self.logs=[max(-80.,v-peak) for v in base]
        self.ids=deque(maxlen=64)
        self._evidence=None if self.prior is None and not self.inertia and not self.assumption_trust else HypothesisTracker(names,retention,smoothing,window,responsive)

    def record(self):
        return dict(format='subjective-belief-v2',names=list(self.names),retention=self.retention,
            smoothing=self.smoothing,window=self.lifts.maxlen,responsive=self.responsive,
            inertia=self.inertia,prior=None if self.prior is None else dict(zip(self.names,self.prior)),
            assumption_trust=self.assumption_trust,logs=list(self.logs),observations=self.observations,
            lifts=list(self.lifts),ids=list(self.ids),surprise=self.surprise,
            evidence=None if self._evidence is None else self._evidence.record())

    @classmethod
    def from_record(cls,r):
        if r['format']!='subjective-belief-v2':raise ValueError('known belief checkpoint required')
        t=cls(r['names'],r['retention'],r['smoothing'],r['window'],r['responsive'],
            inertia=r['inertia'],prior=r['prior'],assumption_trust=r['assumption_trust'])
        if len(r['logs'])!=len(t.names) or len(r['lifts'])>t.lifts.maxlen or len(r['ids'])>64 or len(set(r['ids']))!=len(r['ids']):
            raise ValueError('bounded belief checkpoint required')
        if type(r['observations']) is not int or r['observations']<len(r['lifts']) or any(not isinstance(k,str) or not k for k in r['ids']):
            raise ValueError('valid observed belief history required')
        for v in (*r['logs'],*r['lifts'],r['surprise']):
            if isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v):raise ValueError('finite belief history required')
        if any(not -80<=v<=0 for v in r['logs']) or max(r['logs'])!=0 or not 0<=r['surprise']<=8:
            raise ValueError('normalized bounded belief state required')
        t.logs=list(r['logs']);t.observations=r['observations'];t.lifts.extend(r['lifts']);t.ids.extend(r['ids']);t.surprise=r['surprise']
        if t._evidence is None:
            if r['evidence'] is not None:raise ValueError('unexpected subjective evidence state')
        else:
            e=cls.from_record(r['evidence'])
            if e._evidence is not None or e.names!=t.names or e.retention!=t.retention or e.smoothing!=t.smoothing or e.responsive!=t.responsive or e.lifts.maxlen!=t.lifts.maxlen or e.observations!=t.observations or e.ids!=t.ids:
                raise ValueError('matching independent observed evidence required')
            t._evidence=e
        return t

    def snapshot(self):
        top=max(self.logs); unscaled=[math.exp(x-top) for x in self.logs]; total=sum(unscaled)
        weights=tuple(x/total for x in unscaled)
        entropy=-sum(w*math.log(w) for w in weights if w)/math.log(len(weights))
        # At least three informative observations; forced moves supply none.
        evidence=0. if self.observations<3 else self.observations/(self.observations+3)
        quality=min(1.,math.exp(sum(self.lifts)/len(self.lifts))) if self.lifts else 1.
        if self.responsive: quality*=math.exp(-self.surprise)
        confidence=min(.9,evidence*max(0.,1-entropy)*quality)
        if self.prior is None and not self.inertia and not self.assumption_trust:
            return BeliefSnapshot(self.names,weights,self.observations,float(confidence),self.smoothing,self.surprise,self.responsive)
        # Assumed trust is not earned evidence. One unlucky result does not
        # erase a prior, but four recent predictive failures cannot hide behind
        # an old conviction. Even maximum inertia keeps a nonzero update rate.
        assumed_quality=1. if self.observations<4 else quality
        assumed=self.assumption_trust*max(0.,1-entropy)*assumed_quality
        empirical=self._evidence.snapshot()
        alignment=sum(min(a,b) for a,b in zip(weights,empirical.weights))
        supported=empirical.confidence*alignment
        return BeliefSnapshot(self.names,weights,self.observations,float(min(.9,max(supported,assumed))),
            self.smoothing,self.surprise,self.responsive,empirical.confidence,float(assumed),self.inertia,empirical.weights,float(alignment))

    def observe(self,models,action,observation_id,*,forecast=None):
        actions,matrix=distributions(models,self.names,self.smoothing)
        if action not in actions: raise ValueError('revealed action must have been legal before reveal')
        if not isinstance(observation_id,str) or not observation_id: raise ValueError('observation ID required')
        if observation_id in self.ids: raise ValueError('duplicate public observation')
        before=self.snapshot()
        if forecast is None:forecast=before.predict(models)
        else:
            if set(forecast)!=set(actions):raise ValueError('forecast must cover identical legal actions')
            probabilities=list(forecast.values())
            if any(isinstance(p,bool) or not isinstance(p,(float,int)) or not math.isfinite(p) or p<=0 for p in probabilities):
                raise ValueError('forecast requires finite positive probabilities')
            if abs(sum(probabilities)-1)>1e-8:raise ValueError('forecast probabilities must sum to one')
        if self._evidence is not None:self._evidence.observe(models,action,observation_id)
        predictive=forecast[action]; uniform=1/len(actions)
        self.ids.append(observation_id)
        effective_retention=None
        if len(actions)>1:
            j=actions.index(action)
            lift=math.log(predictive/uniform); shock=max(0.,-lift)
            self.surprise=min(8.,.5*self.surprise+shock) if self.responsive else 0.
            # Unexpected at less than half the uniform probability: weaken old
            # evidence immediately, but do not declare which new model is true.
            effective_retention=self.retention*((.25+.75*self.inertia) if self.responsive and shock>math.log(2) else 1.)
            rate=1-.8*self.inertia
            self.logs=[effective_retention*x+rate*math.log(row[j]) for x,row in zip(self.logs,matrix)]
            # Remove a harmless common log offset to keep long-running state bounded.
            peak=max(self.logs); self.logs=[max(-80.,x-peak) for x in self.logs]
            self.observations+=1; self.lifts.append(lift)
        return dict(observation_id=observation_id,action=action,forced=len(actions)==1,
                    predicted_probability=predictive,uniform_probability=uniform,
                    log_loss=-math.log(predictive),uniform_log_loss=-math.log(uniform),
                    effective_retention=effective_retention,before=before.record(),after=self.snapshot().record())
