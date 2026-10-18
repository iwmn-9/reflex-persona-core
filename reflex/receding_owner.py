"""Continuous owned planning at the actual decision/observation boundary.

This integration deliberately makes no periodic reflex substitution. It creates
a fresh world forecast from the resolved learner on EVERY requested decision.
It does not advance a world or manufacture feedback. Prediction remains a finite
approximation; future owners learn but by default use the reflex tail policy.
"""
import copy
from .closed_loop import owned_rollout
from .decision_loop import DecisionLoop
from .model_observer import resolve_feedback


class RecedingOwner:
    def __init__(self, actor, *, observe, events, terminal, assess, record,
                 request_for, feedback, model_id, target, tail_steps=15,
                 seeds=(701,702,703), world_model='frozen', max_regret=.15,
                 purpose_estimator='sampled',tail_recovery=0,tail_recovery_limit=16,sampling_standard_errors=None,cumulative_preference=None,tail_goal_refinement=False):
        if not isinstance(actor,DecisionLoop) or actor.pending is not None:
            raise ValueError('one resolved DecisionLoop owner required')
        self.actor=actor
        self.observe=observe;self.request_for=request_for
        self.options=dict(observe=observe,events=events,terminal=terminal,assess=assess,
            record=record,request_for=request_for,feedback=feedback,model_id=model_id,
            target=target,tail_steps=tail_steps,seeds=tuple(seeds),world_model=world_model,max_regret=max_regret,
            purpose_estimator=purpose_estimator,tail_recovery=tail_recovery,tail_recovery_limit=tail_recovery_limit)
        if sampling_standard_errors is not None:self.options['sampling_standard_errors']=sampling_standard_errors
        if cumulative_preference is not None:self.options['cumulative_preference']=cumulative_preference
        if tail_goal_refinement is not False:self.options['tail_goal_refinement']=tail_goal_refinement
        # Validate the entire configuration before any real decision is pending.
        owned_rollout(actor,**self.options)
        self.last=None

    def decide(self, public):
        if self.actor.pending is not None:raise ValueError('resolve actual feedback before the next decision')
        if self.options['terminal'](public):raise ValueError('cannot decide in a terminal world')
        self.last=None
        req=self.request_for(self.observe(copy.deepcopy(public),copy.deepcopy(self.actor.state)))
        ctrl=owned_rollout(self.actor,**self.options);captured={}
        planner=ctrl.planner(self.actor,req,copy.deepcopy(public),captured=captured)
        def complete(contexts,decisions):
            forecast=planner(contexts,decisions)
            if forecast is None:raise ValueError('incomplete owned forecast; no silent reflex substitution')
            return forecast
        result=DecisionLoop.decide_batch([(self.actor,req)],False,complete)[0]
        self.last=captured
        return result

    def resolve(self, ticket, actual_feedback):
        return resolve_feedback(self.actor,ticket,actual_feedback)
