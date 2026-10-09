"""Optional endpoint purpose evaluation, separate from persona flow preferences.

Games own goal meaning, victory/draw predicates and nonterminal estimates.
The existing strongest-principle tier is determined before competence pruning.
No win probability calibration, universal game score or learning is claimed.
"""
from dataclasses import dataclass,replace,asdict
import copy
from .core import Policy,compile_batch,identifier,number
from .intertemporal import forecast
from .progress import ProgressWatch,ProgressConfig


@dataclass(frozen=True)
class Goal:
    target: str
    status: str
    value: float

    def __post_init__(self):
        identifier(self.target);number(self.value,-1,1)
        if self.status not in ('running','success','failure','draw','scored'):
            raise ValueError('explicit goal settlement required')
        expected={'success':1.,'failure':-1.,'draw':0.}.get(self.status)
        if expected is not None and self.value!=expected:
            raise ValueError('settled victory/loss/draw cannot retain a partial-progress proxy')

    def record(self):return asdict(self)


def goal_forecast(c,paths,rollout_audit,*,horizon,unit,target,max_regret=.15,plan_roots=None,policy=None,settlement_weight='discounted'):
    """Use one endpoint per coherent branch for objective and competence.

    Needs/values/style/cost retain their discounted flow treatment. Subjective
    goal return is discounted by default; absolute mode does not time-discount
    settled outcomes, while retaining cutoff-proxy/flow time weights. The
    competence endpoint is not diluted by horizon length or periodic income.
    Nonterminal value remains an explicit game proxy, not a success probability.
    First retain the owner's FULL-forecast strongest-value tier; only then apply
    the purpose corridor inside it. A principle may therefore rationally lose.
    """
    if settlement_weight not in ('discounted','absolute'):raise ValueError('known settlement time weight required')
    f=forecast(c,paths,horizon=horizon,unit=unit,target=target,max_regret=max_regret,plan_roots=plan_roots)
    fc=copy.deepcopy(f.contexts[0]);purpose={};endpoint={}
    if set(rollout_audit)!=set(paths):raise ValueError('endpoint coverage must match all viable roots')
    for a in fc['actions']:
        root=f.roots[a['id']][0];proposal=f.audit['plans'][a['id']].get('proposal',root);rows=rollout_audit[proposal]
        if len(rows)!=len(paths[proposal]):raise ValueError('one endpoint per coherent branch required')
        goals=[];expected=0.
        for out,trace,branch in zip(a['outcomes'],rows,paths[proposal]):
            g=Goal(**trace['assessment'])
            if g.target!=target:raise ValueError('same goal target across branches required')
            if bool(trace['terminal'])!=(g.status!='running'):
                raise ValueError('terminal and settled goal must agree')
            steps=trace['actions']
            if not isinstance(steps,list) or not 1<=len(steps)<=horizon or steps[0]!=root:
                raise ValueError('actual modeled settlement/cutoff time required')
            confidence=min(branch.confidence[:len(steps)]) if g.value>0 else 1.
            time_weight=1. if settlement_weight=='absolute' and g.status!='running' else f.audit['discount']**(len(steps)-1)
            out['objective']=g.value*confidence*time_weight
            expected+=out['p']*g.value*confidence*(a['confidence'] if g.value>0 else 1.)
            goals.append(dict(g.record(),settlement_step=len(steps),positive_confidence=confidence))
        purpose[a['id']]=expected
        endpoint[a['id']]=goals
    # Preserving the tier before the purpose gate prevents a competence band
    # from silently substituting a lower-principle option for the owner.
    b=compile_batch([fc]);d=(policy or Policy()).decide(b,False)
    retained={key for key,yes in zip(b.ids[0],d.eligible[0]) if yes}
    fc['actions']=[a for a in fc['actions'] if a['id'] in retained]
    metadata=copy.deepcopy(f.audit);metadata.update(endpoint=endpoint,original_tier_plans=sorted(retained),
        endpoint_semantics='settled result OR explicit nonterminal proxy; not calibrated win EV',
        readiness_learned=False)
    if settlement_weight=='absolute':metadata['subjective_settlement_semantics']='absolute terminal outcome once; running cutoff proxy and preference flows still discounted'
    return replace(f,contexts=(fc,),roots={k:v for k,v in f.roots.items() if k in retained},
        purpose={k:v for k,v in purpose.items() if k in retained},audit=metadata)


class PersonaProgressWatch(ProgressWatch):
    """Do not expire the owner's entire immediate strongest-principle tier."""
    def __init__(self,scope,config=None):
        # Persist the behavior in the ordinary ProgressWatch checkpoint. The
        # DecisionLoop restores that base type, so subclass-only masks would
        # otherwise disappear silently after save/restore.
        super().__init__(scope,replace(config or ProgressConfig(),preserve_persona_tier=True))
