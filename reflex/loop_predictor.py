"""Bounded opponent hypotheses connected to DecisionLoop's observed lifecycle."""
import copy
from collections import deque,Counter
from .core import digest
from .opponent_beliefs import HypothesisTracker,distributions
from .decision_loop import Reading,EvidenceGate
from .prediction_support import SupportBanks,gate_record,restored_gate


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


class SupportedCategoricalReader:
    """DecisionLoop reader with finite public support and separate trust keys.

    ``opportunity_for`` reads the pre-action PUBLIC context. ``support_key`` is
    the adapter's versioned meaning of those classes; checkpoint restore cannot
    silently change it. Branch evaluations retain the ordinary node accounting.
    """
    def __init__(self,scope,names,models_for,outcome_for,*,classes,opportunity_for,support_key,key='categorical-response',known_conditionals=False):
        if not isinstance(support_key,str) or not 0<len(support_key)<=128:raise ValueError('versioned bounded support contract required')
        self.owner=digest(scope);self.key=key;self.support_key=support_key;self.opportunity_for=opportunity_for
        self.support=SupportBanks(classes,lambda:CategoricalReader(scope,names,models_for,outcome_for,key,known_conditionals))
        for opportunity in self.support.classes:
            self.support.select(opportunity).key=key+':'+digest([support_key,opportunity])

    def _class(self,context):
        if digest(context['scope'])!=self.owner:raise ValueError('predictor owner mismatch')
        opportunity=self.opportunity_for(copy.deepcopy(context));self.support.select(opportunity)
        return opportunity

    def forecasts(self,context):return self.support.select(self._class(context)).forecasts(context)

    def __call__(self,context,cap):
        opportunity=self._class(context);reading=self.support.select(opportunity)(context,cap)
        if reading is not None:reading.context['facts']['prediction_support']=self.support_key+':'+opportunity
        return reading

    def updated(self,context,observed,event,ticket):
        opportunity=self._class(context)
        child,update=self.support.select(opportunity).updated(context,observed,event,ticket)
        new=copy.deepcopy(self);new.support._banks[opportunity]=child
        return new,dict(update,opportunity_class=opportunity,support_key=self.support_key)

    def record(self):
        return dict(version='loop-supported-predictor-v1',owner=self.owner,key=self.key,support_key=self.support_key,
            classes=list(self.support.classes),banks={k:self.support.select(k).record() for k in self.support.classes})

    def restored(self,record):
        if record['version']!='loop-supported-predictor-v1' or record['owner']!=self.owner or record['key']!=self.key or record['support_key']!=self.support_key or tuple(record['classes'])!=self.support.classes or set(record['banks'])!=set(self.support.classes):
            raise ValueError('supported predictor checkpoint contract mismatch')
        new=copy.deepcopy(self)
        for k in self.support.classes:new.support._banks[k]=self.support.select(k).restored(record['banks'][k])
        return new


class ValidatedSupportedCategoricalReader(SupportedCategoricalReader):
    """Keep local evidence intact; validate and revoke cross-support borrowing."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.shared=copy.deepcopy(self.support.select(self.support.classes[0]))
        self.shared.key=self.key+':shared-candidate';self.transfer=EvidenceGate()

    def selected(self,context):
        opportunity=self._class(context)
        return self.shared if self.transfer.accepted(opportunity) else self.support.select(opportunity)

    def forecasts(self,context):return self.selected(context).forecasts(context)

    def __call__(self,context,cap):
        opportunity=self._class(context);selected=self.selected(context);reading=selected(context,cap)
        if reading is None:return None
        key=self.support.select(opportunity).key+(':validated-transfer' if selected is self.shared else ':local')
        reading.context['facts']['prediction_support']=self.support_key+':'+opportunity
        return Reading(reading.context,reading.nodes,key,reading.target,reading.responses,reading.known_conditionals,reading.response_candidate)

    def updated(self,context,observed,event,ticket):
        opportunity=self._class(context);local=self.support.select(opportunity)
        if not isinstance(event,dict) or set(event)!={'revealed_action'}:raise ValueError('one actual revealed action required')
        index=0 if local.known_conditionals else 1
        local_forecast=local.forecasts(context)[index];shared_forecast=self.shared.forecasts(context)[index]
        new=copy.deepcopy(self)
        transfer=(new.transfer.categorical(opportunity,local_forecast,shared_forecast,event['revealed_action'])
                  if len(local_forecast)>1 else dict(scored=False,reason='forced public response supplies no transfer evidence'))
        child,update=local.updated(context,observed,event,ticket)
        new.support._banks[opportunity]=child
        new.shared,_=self.shared.updated(context,observed,event,ticket)
        return new,dict(update,opportunity_class=opportunity,support_key=self.support_key,transfer=transfer)

    def record(self):
        record=super().record();record.update(version='loop-validated-supported-predictor-v1',shared=self.shared.record(),transfer=gate_record(self.transfer))
        return record

    def restored(self,record):
        if record['version']!='loop-validated-supported-predictor-v1':raise ValueError('validated support checkpoint mismatch')
        if any(record['transfer'][k]!=getattr(self.transfer,k) for k in ('capacity','window','min_trials','margin')):
            raise ValueError('transfer checkpoint validation controls changed')
        local=copy.deepcopy(record);local['version']='loop-supported-predictor-v1'
        new=super().restored(local)
        new.shared=self.shared.restored(record['shared']);new.transfer=restored_gate(record['transfer'])
        if set(new.transfer.entries)-set(self.support.classes):raise ValueError('undeclared transfer opportunity')
        return new
