"""Bounded opponent hypotheses connected to DecisionLoop's observed lifecycle."""
import copy
from collections import deque,Counter
from .core import digest
from .opponent_beliefs import HypothesisTracker,distributions
from .decision_loop import Reading


class CategoricalReader:
    """Game callbacks map PUBLIC snapshots to hypotheses and conditional effects.

    Every root/response effect evaluation counts against the supplied cap. The
    loop trains this owner-local tracker only after the actual event is revealed.
    World state/pending opponent moves are never supplied to these callbacks.
    """
    def __init__(self,scope,names,models_for,outcome_for,key='categorical-response',known_conditionals=False):
        self.owner=digest(scope);self.tracker=HypothesisTracker(names)
        if type(known_conditionals) is not bool:raise ValueError('explicit known-rule declaration required')
        self.models_for=models_for;self.outcome_for=outcome_for;self.key=key;self.known_conditionals=known_conditionals
        self.recent=deque(maxlen=4)

    def forecasts(self,context):
        if digest(context['scope'])!=self.owner:raise ValueError('predictor owner mismatch')
        models=self.models_for(copy.deepcopy(context))
        candidate=self.tracker.snapshot().predict(models)
        if any(k not in candidate for k in self.recent):raise ValueError('response space changed without rebuilding predictor')
        counts=Counter(self.recent)
        prior={k:(counts[k]+1)/(len(self.recent)+len(candidate)) for k in candidate} if len(self.recent)>=3 else {k:1/len(candidate) for k in candidate}
        return prior,candidate

    def __call__(self,context,cap):
        if digest(context['scope'])!=self.owner:raise ValueError('predictor owner mismatch')
        if self.known_conditionals and len(self.recent)<3:return None
        prior,candidate=self.forecasts(context)
        forecast=prior if self.known_conditionals else candidate
        required=len(context['actions'])*len(forecast)
        if required>cap:return None
        result=copy.deepcopy(context)
        for a in result['actions']:
            rows=[]
            for response,p in forecast.items():
                row=self.outcome_for(copy.deepcopy(context),a['id'],response)
                row=copy.deepcopy(row);row['p']=p;rows.append(row)
            a['outcomes']=rows
        result['facts']['predictor_model']='owner-local finite hypotheses; trained only from revealed events'
        return Reading(result,required,self.key,responses=forecast,known_conditionals=self.known_conditionals,
            response_candidate=candidate if self.known_conditionals else None)

    def updated(self,context,observed,event,ticket):
        if digest(context['scope'])!=self.owner:raise ValueError('predictor owner mismatch')
        if not isinstance(event,dict) or set(event)!=set(('revealed_action',)):raise ValueError('one actual revealed action required')
        new=copy.deepcopy(self)
        models=self.models_for(copy.deepcopy(context))
        update=new.tracker.observe(models,event['revealed_action'],ticket)
        new.recent.append(event['revealed_action'])
        return new,update

    def record(self):
        t=self.tracker
        return dict(version='loop-predictor-v1',owner=self.owner,key=self.key,known_conditionals=self.known_conditionals,names=list(t.names),
            logs=list(t.logs),observations=t.observations,lifts=list(t.lifts),ids=list(t.ids),surprise=t.surprise,recent=list(self.recent))

    def restored(self,record):
        import math
        from collections import deque
        if record['version']!='loop-predictor-v1' or record['owner']!=self.owner or record['key']!=self.key or record['known_conditionals']!=self.known_conditionals or tuple(record['names'])!=self.tracker.names:
            raise ValueError('predictor checkpoint contract mismatch')
        new=copy.deepcopy(self);t=new.tracker
        if (len(record['logs'])!=len(t.names) or len(record['lifts'])>t.lifts.maxlen or len(record['ids'])>t.ids.maxlen or
            type(record['observations']) is not int or record['observations']<0):raise ValueError('invalid predictor checkpoint dimensions')
        if any(not math.isfinite(v) for v in record['logs']+record['lifts']+[record['surprise']]) or not 0<=record['surprise']<=8:
            raise ValueError('invalid predictor checkpoint numbers')
        if any(not isinstance(v,str) or not v for v in record['ids']) or len(set(record['ids']))!=len(record['ids']):raise ValueError('invalid predictor observation IDs')
        t.logs=record['logs'].copy();t.lifts=deque(record['lifts'],maxlen=t.lifts.maxlen);t.ids=deque(record['ids'],maxlen=t.ids.maxlen)
        if not isinstance(record['recent'],list) or len(record['recent'])>4 or any(not isinstance(k,str) or not k for k in record['recent']):raise ValueError('invalid recent response checkpoint')
        new.recent=deque(record['recent'],maxlen=4)
        t.observations=record['observations'];t.surprise=record['surprise'];return new
