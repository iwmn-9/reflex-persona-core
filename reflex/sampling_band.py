"""Numerical sampling ambiguity is distinct from the person's risk preference.

These are paired Monte Carlo errors under one declared model, not confidence in
that model, calibrated intervals, or a guarantee of policy improvement. Keeping
an ambiguous option never removes its downside from the ordinary persona score.
"""
from dataclasses import dataclass
import math
from .core import identifier


@dataclass(frozen=True)
class PurposeSampling:
    sample_ids: tuple
    samples: tuple
    source: str
    standard_errors: float = 2.

    def __post_init__(self):
        if (not isinstance(self.sample_ids,tuple) or not 2<=len(self.sample_ids)<=32
            or any(type(s) is not int for s in self.sample_ids) or len(set(self.sample_ids))!=len(self.sample_ids)):
            raise ValueError('two to thirty-two distinct independent scenario ids required')
        if not isinstance(self.source,str) or not self.source:raise ValueError('declared sampling source required')
        if isinstance(self.standard_errors,bool) or not isinstance(self.standard_errors,(int,float)) or not math.isfinite(self.standard_errors) or not 0<=self.standard_errors<=8:
            raise ValueError('bounded empirical sampling margin required')
        if not isinstance(self.samples,tuple) or not 1<=len(self.samples)<=256:
            raise ValueError('bounded immutable plan samples required')
        seen=set()
        for entry in self.samples:
            if not isinstance(entry,tuple) or len(entry)!=2:raise ValueError('plan and aligned samples required')
            key,values=entry;identifier(key)
            if key in seen:raise ValueError('unique sampled plans required')
            seen.add(key)
            if (not isinstance(values,tuple) or len(values)!=len(self.sample_ids)
                or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or abs(v)>1e6 for v in values)):
                raise ValueError('finite aligned complete-path estimates required')

    def mask(self,names,purpose,supported,regret):
        values=dict(self.samples)
        if set(values)!=set(names) or len(purpose)!=len(names) or len(supported)!=len(names):
            raise ValueError('complete matching forecast samples required')
        n=len(self.sample_ids)
        for key,q in zip(names,purpose):
            mean=max(-1.,min(1.,math.fsum(values[key])/n))
            if abs(mean-q)>1e-10:raise ValueError('sample mean does not match declared purpose')
        allowed=list(map(bool,supported));excluded={}
        for i,key in enumerate(names):
            if not supported[i]:continue
            for j,other in enumerate(names):
                if not supported[j] or purpose[j]-purpose[i]<=regret+1e-12:continue
                differences=[b-a for a,b in zip(values[key],values[other])]
                mean=math.fsum(differences)/n
                se=math.sqrt(math.fsum((d-mean)**2 for d in differences)/(n*(n-1)))
                if purpose[j]-purpose[i]>regret+self.standard_errors*se+1e-12:
                    allowed[i]=False
                    excluded[key]=dict(by=other,purpose_gap=float(purpose[j]-purpose[i]),paired_standard_error=se)
                    break
        return allowed,dict(source=self.source,samples=n,standard_errors=self.standard_errors,
            ordinary_purpose_band=regret,excluded=excluded,
            semantics='Empirical paired Monte Carlo margin only; not calibrated confidence or world-model accuracy. Full persona and risk distributions remain active.')
