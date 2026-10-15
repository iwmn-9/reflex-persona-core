"""Detached execution of the owner's real immediate decision lifecycle.

The game creates Requests from PUBLIC contexts and reveals modeled feedback
after a transition. Every trial owns a fresh actor copy. Its outcome/relationship
and opponent learning stays inside that hypothetical branch. This is not a
recursive model of an outer planner or a calibrated world-model guarantee.
"""
import copy
import numpy as np
from .core import digest,compile_batch
from .decision_loop import DecisionLoop,Request
from .model_observer import ModelObserver,ModelFeedback,resolve_feedback


class ActorModel(ModelObserver):
    def __init__(self,actor,*,root_request,request_for,feedback,future_planner=None,future_planner_limit=None):
        if not isinstance(actor,DecisionLoop) or actor.pending is not None:raise ValueError('resolved owned actor required')
        if not isinstance(root_request,Request) or digest(root_request.context['scope'])!=actor.owner:raise ValueError('owned raw root Request required')
        if not callable(request_for) or not callable(feedback):raise ValueError('public request and feedback callbacks required')
        if future_planner is not None and not callable(future_planner):raise ValueError('explicit bounded future planner required')
        if future_planner_limit is not None and (future_planner is None or type(future_planner_limit) is not int or not 1<=future_planner_limit<=16):raise ValueError('bounded future planner call limit required')
        actor.record()
        self.owner=actor.owner;self.actor=copy.deepcopy(actor);self.root_request=copy.deepcopy(root_request)
        self.request_for=request_for;self.feedback=feedback;self._request=None;self._preview=None;self._preview_actor=None
        self.future_planner=future_planner;self._public_state=None;self._prepared_root=False
        self.future_planner_limit=future_planner_limit;self._future_calls=0
        self.executed=0;self.forced=0

    def fork(self):
        if self.actor.pending is not None:raise ValueError('resolve branch transition before forking')
        out=copy.copy(self);out.actor=copy.deepcopy(self.actor);out.root_request=copy.deepcopy(self.root_request)
        if self._request is not None:out.root_request=copy.deepcopy(self._request)
        out._request=None;out._preview=None;out._preview_actor=None
        out._public_state=None;out._prepared_root=False
        return out

    def validate_policy(self,policy):
        own=self.actor.policy
        if own.principle_priority!=policy.principle_priority or not np.array_equal(own.residual,policy.residual):raise ValueError('modeled owner policy changed')
        if own.commitments is not None and own.commitments[self.actor.character]!=policy.commitments.get(self.actor.character):raise ValueError('modeled owner commitment changed')

    def adopt_state(self,state):
        if self.actor.pending is None or state['intent_action']!=self.actor.pending['action']:raise ValueError('state must belong to executed modeled root')
        c=copy.deepcopy(self._request.context);c['state']=copy.deepcopy(state);compile_batch([c])
        self.actor.state=copy.deepcopy(state)

    def prepare(self,state,context,*,prepared=False):
        if self.actor.pending is not None:raise ValueError('resolve branch transition before preparing')
        req=copy.deepcopy(self.root_request) if prepared else self.request_for(copy.deepcopy(context))
        if not isinstance(req,Request) or req.context['tick']!=context['tick']:raise ValueError('same-clock public Request required')
        stage=self.actor._prepare(req)
        self._request=req;self._preview=None;self._preview_actor=None
        self._public_state=copy.deepcopy(state);self._prepared_root=prepared
        return stage['effective'],req,stage['progress_allowed'],stage['progress_audit']

    def preview(self):
        if self._request is None:raise ValueError('prepare branch request first')
        self._preview_actor=copy.deepcopy(self.actor)
        planner=None
        if self.future_planner is not None and not self._prepared_root and (self.future_planner_limit is None or self._future_calls<self.future_planner_limit):
            # No trajectory seed is passed to the controller. The caller owns
            # its finite recursion budget and separate forecast scenario bank.
            planner=self.future_planner(self._preview_actor,copy.deepcopy(self._request),copy.deepcopy(self._public_state))
            self._future_calls+=1
        self._preview=DecisionLoop.decide_batch([(self._preview_actor,self._request)],False,planner)[0]
        return copy.deepcopy(self._preview['decision'])

    def commit_choice(self,key):
        if self._preview is None:raise ValueError('preview branch decision first')
        if key==self._preview['decision']['action_id']:
            self.actor=self._preview_actor;decision=self._preview['decision']
        else:
            actor=copy.deepcopy(self.actor)
            result=DecisionLoop.decide_batch([(actor,self._request)],False,_counterfactual_roots=(key,))[0]
            decision=result['decision']
            if decision['action_id']!=key:raise ValueError('counterfactual root was not committed')
            self.actor=actor;self.forced+=1
        self.executed+=1
        return copy.deepcopy(decision)

    def advance(self,before,after,key,context,request):
        if self.actor.pending is None or self.actor.pending['action']!=key:raise ValueError('matching modeled owner selection required')
        fb=self.feedback(copy.deepcopy(before),copy.deepcopy(after),key,copy.deepcopy(context))
        if not isinstance(fb,ModelFeedback):raise ValueError('explicit modeled feedback required')
        ticket=self.actor.pending['ticket']
        resolve_feedback(self.actor,ticket,fb)
        self._request=None;self._preview=None;self._preview_actor=None
        self._public_state=None;self._prepared_root=False

    def record(self):
        result=dict(owner=self.owner,actor=self.actor.record(),executed=self.executed,forced=self.forced,
            semantics='detached immediate owner lifecycle; modeled learning never updates real owner')
        if self.future_planner is not None:result.update(future_calls=self._future_calls,future_limit=self.future_planner_limit,
            semantics='detached owner with explicitly bounded future planner; modeled learning never updates real owner')
        return result
