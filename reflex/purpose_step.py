"""One-step purpose/persona continuation using only known public conditionals.

This is the horizon-one form of owned_rollout, without simulating unused child
owners. It is also an inexpensive identified policy for its imagined learner.
No state is committed here; DecisionLoop owns all real and imagined updates.
"""
import copy
from .core import digest
from .adaptive_dynamics import PublicEvent
from .intertemporal import Branch
from .purpose_plan import Goal, goal_forecast


def purpose_step(actor, world, *, events, terminal, assess, target, max_regret=.15, captured=None):
    if actor.pending is not None or not callable(getattr(actor.predictor,'frozen_forecaster',None)):
        raise ValueError('resolved owner with a public predictive distribution required')
    if any(not callable(f) for f in (events,terminal,assess)) or terminal(world):
        raise ValueError('live public state and conditional rules required')
    saved=digest(actor.record());public=copy.deepcopy(world)
    predict=actor.predictor.frozen_forecaster()
    origin=Goal(**assess(public))
    if origin.target!=target or origin.status!='running':raise ValueError('live same-target purpose required')
    def planner(contexts, decisions):
        if len(contexts)!=1 or digest(actor.record())!=saved:
            raise ValueError('rebuild purpose continuation after owner feedback')
        c=contexts[0];p=predict(c);paths={};audit={}
        for a in c['actions']:
            if not a['legal'] or a['known_failure']:continue
            key=a['id'];rows=events(copy.deepcopy(public),key)
            if (not isinstance(rows,(list,tuple)) or any(not isinstance(e,PublicEvent) for e in rows)
                    or len({e.symbol for e in rows})!=len(rows) or {e.symbol for e in rows}!=set(p)):
                raise ValueError('one conditional event per public response required')
            paths[key]=[];audit[key]=[]
            for e in rows:
                if not p[e.symbol]:continue
                g=Goal(**assess(e.public));ended=bool(terminal(e.public))
                if g.target!=target or ended!=(g.status!='running'):raise ValueError('coherent public endpoint required')
                paths[key].append(Branch(p[e.symbol],(copy.deepcopy(e.flow),),(1.,)))
                audit[key].append(dict(actions=[key],terminal=ended,assessment=g.record()))
        forecast=goal_forecast(c,paths,audit,horizon=1,unit='owner-decisions',target=target,
            policy=actor.policy,max_regret=max_regret,settlement_weight='absolute',
            opportunity_regret=True,objective_origin=origin,outcome_limit=256)
        if captured is not None:captured['node']=dict(forecast=forecast)
        return forecast
    return planner
