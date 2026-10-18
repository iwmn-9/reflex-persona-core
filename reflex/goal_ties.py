"""Refine coarse reflex ties with a public one-step purpose distribution.

This is a better informed reflex, not a proof of long-run state dominance.
Distinct intrinsic effects and crossing purpose risks remain incomparable.
"""
from dataclasses import dataclass
import copy
import math
from .core import digest
from .purpose_plan import Goal


def constant_signature(action, context):
    if not action['legal'] or action['known_failure'] or action['switch_cost']:
        return None
    effects=[{k:v for k,v in row.items() if k!='p'} for row in action['outcomes']]
    if not effects or any(row!=effects[0] for row in effects[1:]):return None
    other=context.get('opponent')
    reading=None if other is None else other['deltas'].get(action['id'])
    return digest([effects[0],action['target'],action['confidence'],action['familiarity'],reading])


def has_aliases(context):
    signatures=[constant_signature(a,context) for a in context['actions']]
    signatures=[s for s in signatures if s is not None]
    return len(signatures)!=len(set(signatures))


def dominates(left,right):
    """First-order dominance of finite, normalized purpose lotteries."""
    support=sorted({v for _,v in left+right});strict=False
    for value in support[:-1]:
        a=math.fsum(p for p,v in left if v<=value)
        b=math.fsum(p for p,v in right if v<=value)
        if a>b:return False
        strict|=a<b
    return strict


@dataclass(frozen=True)
class GoalTieRefinement:
    target: str
    distributions: dict

    def validate(self,forecast):
        if self.target!=forecast.target or len(forecast.contexts)!=1 or forecast.horizon!=1:
            raise ValueError('one owner and matching immediate purpose required')
        if forecast.return_frame is not None or forecast.continuity is not None or forecast.purpose_sampling is not None:
            raise ValueError('unscaled immediate coarse effects required for goal ties')
        if set(self.distributions)!=set(forecast.roots):raise ValueError('complete purpose lotteries required')
        for key,rows in self.distributions.items():
            if forecast.roots[key]!=(key,):raise ValueError('unchanged immediate roots required')
            if not isinstance(rows,tuple) or not 1<=len(rows)<=256:raise ValueError('bounded purpose lottery required')
            for p,v in rows:
                if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in (p,v)) or not 0<p<=1 or not -1<=v<=1:
                    raise ValueError('finite positive probability and bounded purpose required')
            if abs(math.fsum(p for p,_ in rows)-1)>1e-12:raise ValueError('normalized purpose lottery required')

    def resolve(self,context,names,eligible,subjective,selected):
        actions={a['id']:a for a in context['actions']};old=names[selected]
        signature=constant_signature(actions[old],context)
        audit=dict(original=old,selected=old,reason='no dominating coarse tie')
        if signature is None:return selected,audit
        # The complete effect signature is the semantic equality check. This
        # numerical check additionally rejects differences from other advice.
        tolerance=8*math.ulp(max(1.,abs(float(subjective[selected]))))
        peers=[j for j,k in enumerate(names) if eligible[j] and
            constant_signature(actions[k],context)==signature and
            abs(float(subjective[j]-subjective[selected]))<=tolerance]
        better=[j for j in peers if dominates(self.distributions[names[j]],self.distributions[old])]
        # Do not decide a new risk tradeoff between incomparable improvements.
        winners=[j for j in better if all(self.distributions[names[j]]==self.distributions[names[k]] or
            dominates(self.distributions[names[j]],self.distributions[names[k]]) for k in better if j!=k)]
        if winners:
            selected=min(winners);audit.update(selected=names[selected],reason='modeled first-order purpose dominance within an exact coarse tie')
        return selected,audit


def reflex_goal_planner(person,request,state,*,events,assess,target):
    """Optional tail callback; uses the future person's beliefs, never truth."""
    if person.social is not None or request.reader is not None or request.value_proposal is not None:
        return None
    if not has_aliases(request.context):return None
    from .adaptive_dynamics import PublicEvent
    from .deliberation import JointForecast
    def plan(contexts,decisions):
        if len(contexts)!=1 or len(decisions)!=1:raise ValueError('one reflex owner required')
        context=contexts[0]
        if not has_aliases(context):return None
        chosen=decisions[0]['action_id']
        old=next(a for a in context['actions'] if a['id']==chosen)
        signature=constant_signature(old,context)
        if signature is None:return None
        peers=[a for a in context['actions'] if constant_signature(a,context)==signature]
        if len(peers)<2:return None
        probabilities=person.predictor.predict(copy.deepcopy(context));distributions={}
        for action in peers:
            rows=events(copy.deepcopy(state),action['id'])
            if (not rows or any(not isinstance(e,PublicEvent) for e in rows) or
                len({e.symbol for e in rows})!=len(rows) or {e.symbol for e in rows}!=set(probabilities)):
                raise ValueError('complete known public conditional responses required')
            values=[]
            for event in rows:
                goal=Goal(**assess(copy.deepcopy(event.public)))
                if goal.target!=target:raise ValueError('matching public purpose required')
                if probabilities[event.symbol]>0:values.append((float(probabilities[event.symbol]),goal.value))
            distributions[action['id']]=tuple(values)
        if not any(dominates(rows,distributions[chosen]) for key,rows in distributions.items() if key!=chosen):return None
        refined=copy.deepcopy(context)
        refined['actions']=[a for a in refined['actions'] if a['id'] in distributions]
        return JointForecast((refined,),{k:(k,) for k in distributions},
            {k:math.fsum(p*v for p,v in rows) for k,rows in distributions.items()},1,
            max_regret=2.,target=target,goal_ties=GoalTieRefinement(target,distributions),
            audit=dict(controller='public-purpose-coarse-tie-v1',scope='immediate modeled lotteries; not future-state dominance'))
    return plan
