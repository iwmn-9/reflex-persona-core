"""Whole public-model/actual-driver comparisons without per-game tuning."""
import copy
from dataclasses import replace
from .core import Policy,character_key,digest
from .decision_loop import DecisionLoop
from .integrated_trials import request_for,compact
from .contingent_games import make_bridge
from .forecast_control import ForecastBudget,ForecastController,ForecastLedger
from .model_observer import ModelFeedback


BUDGETS={
    'short':ForecastBudget(horizon=2),
    'long':ForecastBudget(horizon=4),
    'more_samples':ForecastBudget(horizon=4,seeds=tuple(range(401,409))),
    'resampled':ForecastBudget(horizon=4,validation_seeds=(501,502,503,504),uncertainty_weight=0.),
    'feedback':ForecastBudget(horizon=4,future_levels=1,validation_seeds=(501,502,503,504),uncertainty_weight=0.),
    'checked':ForecastBudget(horizon=4,future_levels=1,validation_seeds=(501,502,503,504)),
    'adaptive':ForecastBudget(horizon=4,future_levels=1,validation_seeds=(501,502,503,504),extra_validation_seeds=(505,506,507,508)),
    'challenged':ForecastBudget(horizon=4,future_levels=1,validation_seeds=(501,502,503,504),audit_seeds=(701,702,703,704),purpose_challenger=True),
}
BUDGETS={name:replace(budget,direct_continuation=True) for name,budget in BUDGETS.items()}


def play(case,profile,seed,arm,*,budgets=None):
    budget=(budgets or BUDGETS)[arm];b=make_bridge(profile,case);w=b.start();c=b.observe(b.project(w))
    p=Policy(principle_priority='personal',commitments={character_key(c):.5})
    actor=DecisionLoop(c,policy=p);ledger=ForecastLedger(c['scope']);trace=[]
    req_for=lambda cc:request_for(cc,case['game'],learn=False)
    controller=ForecastController(observe=b.observe,advance=b.modeled_advance,terminal=b.terminal,
        assess=lambda s:b.goal(s).record(),request_for=req_for,feedback=lambda *a:ModelFeedback(None),
        unit=b.unit,target=b.target,budget=budget)
    while not b.terminal(w):
        projected=b.project(w);c=b.observe(projected,actor.state);req=req_for(c);captured={}
        saved=digest(actor.record());work=controller.work
        result=DecisionLoop.decide_batch([(actor,req)],False,controller.planner(actor,req,projected,captured=captured))[0]
        key=result['decision']['action_id'];f=captured['forecast'];audit=captured['audit']
        step=len(trace);ledger.commit(c,step,key,f.horizon,audit[key],b.goal(w).record())
        # Frozen discovery predictions are scored separately from validation.
        before=b.record(w);w,row=b.actual(w,key,seed);actor.abandon(result['ticket'])
        scores=ledger.observe(step+1,key,b.goal(w).record(),terminal=b.terminal(w))
        trace.append(dict(before=before,after=b.record(w),root=key,flow=row,goal=b.goal(w).record(),
            decision=result['decision'],deliberation={k:v for k,v in result['deliberation'].items() if k!='metadata'},
            check=f.audit.get('forecast_check'),forecast_scores=scores,modeled=compact({key:audit[key]})[key],
            adaptive=f.audit.get('adaptive_validation'),
            challenger=f.audit.get('purpose_challenger'),
            model_steps=controller.work-work,owner_before=saved,owner_after=digest(actor.record())))
        if len(trace)>1000:raise AssertionError('whole episode failed to settle')
    check=b.start()
    for t in trace:
        assert b.record(check)==t['before'];check,flow=b.actual(check,t['root'],seed)
        assert b.record(check)==t['after'] and flow==t['flow'] and b.goal(check).record()==t['goal']
    assert b.terminal(check) and not ledger.pending
    return dict(case=case,profile=profile['id'],seed=seed,arm=arm,budget=vars(budget),goal=b.goal(w).record(),
        steps=len(trace),actions=[t['root'] for t in trace],trace=trace,model_steps=controller.work,
        learned_entries=len(actor.memory.entries),action_replay_verified=True,
        limitations=['one bounded nested controller level, not full recursive actual-controller equivalence',
            'game-defined goal proxy at horizon; ledger is predictive scoring, not delayed causal credit',
            'same immediate learning OFF across arms to isolate planning; no personality or rule coefficient fitting'])
