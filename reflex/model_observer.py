"""Branch-local progress/pressure simulation, never empirical experience.

The caller owns activity and feedback semantics. The root context is already
prepared by DecisionLoop; pressure is only reapplied to later raw observations.
No game rules, action rankings, new coefficients or reward learning live here.
"""
from dataclasses import dataclass
import copy
from .core import digest
from .progress import ProgressWatch,PurposeRequest,PurposeFeedback
from .pressure import NeedPressure


@dataclass(frozen=True)
class ModelFeedback:
    purpose: PurposeFeedback | None
    needs: dict | None=None
    maintained: tuple=()
    completed: tuple=()


class ModelObserver:
    def __init__(self,scope,*,purpose,feedback,progress=None,pressure=None,previous=None):
        self.owner=digest(scope);self.purpose=purpose;self.feedback=feedback
        if progress is not None and (not isinstance(progress,ProgressWatch) or progress.owner!=self.owner):
            raise ValueError('owned model progress checkpoint required')
        if pressure is not None and (not isinstance(pressure,NeedPressure) or pressure.owner!=self.owner):
            raise ValueError('owned model pressure checkpoint required')
        if progress is None and pressure is None:raise ValueError('at least one observed-state component required')
        if not callable(purpose) or not callable(feedback):raise ValueError('game feedback callbacks required')
        self.progress=copy.deepcopy(progress);self.pressure=copy.deepcopy(pressure)
        self.previous=copy.deepcopy(previous)

    def fork(self):
        # Keep callbacks shared; copy only modeled state. Deep-copying a bound
        # game method could silently copy external game resources as well.
        out=copy.copy(self);out.progress=copy.deepcopy(self.progress)
        out.pressure=copy.deepcopy(self.pressure);out.previous=copy.deepcopy(self.previous)
        return out

    def prepare(self,state,context,*,prepared=False):
        if digest(context['scope'])!=self.owner:raise ValueError('model observer owner mismatch')
        c=copy.deepcopy(context)
        if not prepared and self.pressure is not None:c=self.pressure.apply(c)
        req=self.purpose(copy.deepcopy(state),copy.deepcopy(c),copy.deepcopy(self.previous)) if self.progress is not None else None
        if self.progress is not None and not isinstance(req,PurposeRequest):raise ValueError('model purpose request required')
        allowed={a['id'] for a in c['actions'] if a['legal'] and not a['known_failure']}
        audit=None
        if self.progress is not None:allowed,audit=self.progress.mask(c,req)
        return c,req,allowed,audit

    def advance(self,before,after,key,context,request):
        if digest(context['scope'])!=self.owner:raise ValueError('model observer owner mismatch')
        fb=self.feedback(copy.deepcopy(before),copy.deepcopy(after),key)
        if not isinstance(fb,ModelFeedback):raise ValueError('explicit modeled feedback required')
        if self.progress is None and fb.purpose is not None:raise ValueError('unexpected modeled purpose feedback')
        if self.progress is not None and not isinstance(fb.purpose,PurposeFeedback):raise ValueError('modeled purpose feedback required')
        if self.pressure is None and (fb.needs is not None or fb.maintained or fb.completed):raise ValueError('unexpected modeled pressure feedback')
        progress=copy.deepcopy(self.progress);pressure=copy.deepcopy(self.pressure)
        if progress is not None:progress.observe(context['tick'],key,request,fb.purpose)
        if pressure is not None:
            if fb.needs is None:raise ValueError('explicit modeled need feedback required')
            pressure.observe(context['tick'],fb.needs,fb.maintained,fb.completed)
        self.progress=progress;self.pressure=pressure;self.previous=copy.deepcopy(before)

    def record(self):
        return dict(owner=self.owner,progress=None if self.progress is None else self.progress.record(),
            pressure=None if self.pressure is None else self.pressure.record())
