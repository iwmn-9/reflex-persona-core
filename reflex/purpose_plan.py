"""Optional endpoint purpose evaluation, separate from persona flow preferences.

Games own goal meaning, victory/draw predicates and nonterminal estimates.
The existing strongest-principle tier is determined before competence pruning.
No win probability calibration, universal game score or learning is claimed.
"""
from dataclasses import dataclass,replace,asdict
import copy
import math
from .core import Policy,compile_batch,identifier,number,FEATURES,owner_key
from .intertemporal import forecast
from .progress import ProgressWatch,ProgressConfig
from .valuation import ReturnFrame
from .planning import vector


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


def goal_forecast(c,paths,rollout_audit,*,horizon,unit,target,max_regret=.15,plan_roots=None,policy=None,settlement_weight='discounted',
                  objective_origin=None,objective_unit=1.,opportunity_regret=False,competence_support=None,outcome_limit=8,
                  preference_basis=None,cumulative_preference=None,coherent_cumulative_mean=False):
    """Use one endpoint per coherent branch for objective and competence.

    Intrinsic needs/values/style/cost retain their discounted flow treatment;
    explicitly cumulative values use their declared endpoint difference. Subjective
    goal return is discounted by default; absolute mode does not time-discount
    settled outcomes, while retaining cutoff-proxy/flow time weights. The
    competence endpoint is not diluted by horizon length or periodic income.
    Nonterminal value remains an explicit game proxy, not a success probability.
    First retain the owner's FULL-forecast strongest-value tier; only then apply
    the purpose corridor inside it. A principle may therefore rationally lose.
    """
    if settlement_weight not in ('discounted','absolute'):raise ValueError('known settlement time weight required')
    if type(coherent_cumulative_mean) is not bool or coherent_cumulative_mean and cumulative_preference is None:
        raise ValueError('coherent cumulative means require an explicit cumulative declaration')
    if preference_basis is not None:
        from .preference_basis import PreferenceBasis
        if not isinstance(preference_basis,PreferenceBasis) or preference_basis.target!=target:
            raise ValueError('matching declared preference basis required')
        if horizon!=1 or objective_origin is not None or objective_unit!=1.:
            raise ValueError('preference basis currently requires one immediate flow and absolute purpose units')
    if cumulative_preference is not None:
        from .cumulative_preference import CumulativePreference
        if (not isinstance(cumulative_preference,CumulativePreference) or cumulative_preference.target!=target
            or preference_basis is not None or objective_origin is None):
            raise ValueError('matching cumulative declaration with factual origin and no terminal preference substitution required')
    number(objective_unit,.125,1.)
    number(max_regret,0.,2.)
    if objective_origin is not None and (not isinstance(objective_origin,Goal) or objective_origin.target!=target):
        raise ValueError('same-target factual objective origin required')
    if type(opportunity_regret) is not bool:raise ValueError('explicit regret basis required')
    if opportunity_regret and max_regret>1.:raise ValueError('bounded opportunity fraction required')
    if competence_support is not None and (not isinstance(competence_support,set) or any(not isinstance(k,str) for k in competence_support)):
        raise ValueError('supported real roots required')
    encoded_regret=max_regret if opportunity_regret or objective_unit==1. else max_regret*objective_unit
    f=forecast(c,paths,horizon=horizon,unit=unit,target=target,max_regret=encoded_regret,plan_roots=plan_roots,outcome_limit=outcome_limit)
    fc=copy.deepcopy(f.contexts[0]);purpose={};endpoint={};mean_audit={};corrections={}
    if set(rollout_audit)!=set(paths):raise ValueError('endpoint coverage must match all viable roots')
    for a in fc['actions']:
        root=f.roots[a['id']][0];proposal=f.audit['plans'][a['id']].get('proposal',root);rows=rollout_audit[proposal]
        if len(rows)!=len(paths[proposal]):raise ValueError('one endpoint per coherent branch required')
        goals=[];expected=0.;mean_correction=0.;adjusted=False
        for out,trace,branch in zip(a['outcomes'],rows,paths[proposal]):
            g=Goal(**trace['assessment'])
            if cumulative_preference is not None:
                if any(x!=1. for x in branch.confidence):
                    raise ValueError('cumulative values require explicit branch probabilities, not extra branch-confidence scaling')
                resolved=cumulative_preference.resolve(out,g.value,objective_origin.value,target=g.target)
                out.clear();out.update(resolved)
            if preference_basis is not None:
                if any(x!=1. for x in branch.confidence):
                    raise ValueError('preference basis requires explicit outcome probabilities, not additional branch-confidence scaling')
                resolved=preference_basis.resolve(out,g.value,target=target,
                    complete=g.status!='running' or trace.get('terminal_coverage') is True)
                out.clear();out.update(resolved)
            if g.target!=target:raise ValueError('same goal target across branches required')
            if bool(trace['terminal'])!=(g.status!='running'):
                raise ValueError('terminal and settled goal must agree')
            steps=trace['actions']
            if not isinstance(steps,list) or not 1<=len(steps)<=horizon or steps[0]!=root:
                raise ValueError('actual modeled settlement/cutoff time required')
            if g.status=='running' and len(steps)!=horizon:
                raise ValueError('unknown truncated future cannot be padded as a terminal')
            if any(any(vector(row)) for row in branch.effects[len(steps):]):
                raise ValueError('no reward or preference flow after modeled settlement')
            confidence=min(branch.confidence[:len(steps)]) if g.value>0 else 1.
            time_weight=1. if settlement_weight=='absolute' and g.status!='running' else f.audit['discount']**(len(steps)-1)
            if objective_origin is None:
                out['objective']=g.value*confidence*time_weight
                expected+=out['p']*g.value*confidence*(a['confidence'] if g.value>0 else 1.)
                entry=dict(g.record(),settlement_step=len(steps),positive_confidence=confidence)
            else:
                gain=g.value-objective_origin.value
                gain_confidence=min(a['confidence'],*branch.confidence[:len(steps)]) if gain>0 else 1.
                # Delta may span two endpoint units. Transport it at half size
                # and restore declared units. Policy applies root confidence
                # once, so do not double-penalize the same support estimate.
                ratio=gain_confidence/a['confidence'] if gain>0 and a['confidence'] else (0. if gain>0 else 1.)
                out['objective']=gain*ratio*time_weight/2.
                assessed=g.value if gain_confidence==1. else objective_origin.value+gain*gain_confidence
                expected+=out['p']*assessed
                entry=dict(g.record(),settlement_step=len(steps),positive_confidence=confidence,
                    reference_origin=objective_origin.value,gain_confidence=gain_confidence)
            goals.append(entry)
            if 'purpose_mean_estimate' in trace:
                estimate=trace['purpose_mean_estimate']
                if isinstance(estimate,bool) or not isinstance(estimate,(int,float)) or not math.isfinite(estimate):
                    raise ValueError('finite estimated purpose mean required')
                if a['confidence']!=1. or any(v!=1. for v in branch.confidence):
                    raise ValueError('conditional mean correction requires fully supported model probabilities')
                mean_correction+=out['p']*(estimate-g.value);adjusted=True
        # Valid bounded atoms can sum to 1 + one rounding unit. Preserve the
        # strict selector contract without masking a materially invalid score.
        if 1. < abs(expected) <= 1.+1e-12:expected=1. if expected>0 else -1.
        if adjusted:
            corrected=expected+mean_correction
            mean_audit[a['id']]=dict(sampled=expected,unprojected=corrected,projected=not -1<=corrected<=1)
            expected=max(-1.,min(1.,corrected))
        purpose[a['id']]=expected
        endpoint[a['id']]=goals
        if coherent_cumulative_mean:
            correction=[0.]*len(FEATURES)
            if adjusted:
                delta=expected-mean_audit[a['id']]['sampled']
                for key,conversion in cumulative_preference.channels:
                    correction[FEATURES.index(key.split('.')[1])]=delta*conversion
            corrections[a['id']]=tuple(correction)
    # Preserving the tier before the purpose gate prevents a competence band
    # from silently substituting a lower-principle option for the owner.
    frame=ReturnFrame.flows((fc,),f.audit['normalizer'],unit,endpoint=True)
    if objective_origin is not None or objective_unit!=1.:
        objective_scale=(2. if objective_origin is not None else 1.)/objective_unit
        frame=replace(frame,effects=tuple((objective_scale,)+row[1:] for row in frame.effects))
    if cumulative_preference is not None:frame=cumulative_preference.reframe(frame)
    return_means=None
    if coherent_cumulative_mean:
        from .return_means import ReturnMeans
        names=tuple(sorted(corrections))
        return_means=ReturnMeans((owner_key(fc),),names,(tuple(corrections[k] for k in names),))
    framed={} if return_means is None else dict(return_means=return_means)
    b=compile_batch([fc],outcome_limit=outcome_limit);d=(policy or Policy()).decide(b,False,return_frame=frame,**framed)
    retained={key for key,yes in zip(b.ids[0],d.eligible[0]) if yes}
    fc['actions']=[a for a in fc['actions'] if a['id'] in retained]
    if return_means is not None:return_means=return_means.retain(sorted(retained))
    metadata=copy.deepcopy(f.audit);metadata.update(endpoint=endpoint,original_tier_plans=sorted(retained),
        endpoint_semantics='settled result OR explicit nonterminal proxy; not calibrated win EV',
        readiness_learned=False,return_frame=frame.record())
    if mean_audit:
        metadata['purpose_mean_estimator']=dict(kind='conditional public goal control variate',roots=mean_audit,
            semantics='only the competence mean is adjusted; sampled preferences, risk and physical feedback stay intact; model bias remains; bounded projection may bias finite estimates')
    if preference_basis is not None:metadata['preference_basis']=preference_basis.identity
    if cumulative_preference is not None:
        metadata['cumulative_preference']=dict(identity=cumulative_preference.identity,
            channels=[list(row) for row in cumulative_preference.channels],unit=cumulative_preference.unit,
            origin=objective_origin.record(),
            semantics='declared cumulative goal change per sampled outcome; intrinsic flows retain their time weights; model uncertainty and cutoff bias remain')
    if return_means is not None:
        metadata['cumulative_mean_correction']=return_means.record()
        metadata['cumulative_mean_semantics']='same projected conditional goal mean for declared cumulative values and purpose; sampled downside risk and intrinsic flows unchanged; finite-sample/model error remains'
    if settlement_weight=='absolute':metadata['subjective_settlement_semantics']='absolute terminal outcome once; running cutoff proxy and preference flows still discounted'
    if settlement_weight=='absolute' and cumulative_preference is not None:
        metadata['subjective_settlement_semantics']+='; explicitly cumulative value channels use undiscounted sampled goal changes'
    if objective_origin is not None or objective_unit!=1.:
        metadata['objective_basis']=dict(origin=None if objective_origin is None else objective_origin.record(),unit=objective_unit,
            configured_regret=max_regret,encoded_regret=encoded_regret,
            semantics='confidence discounts uncertain gain once, not the known reference level; goal corridor uses declared units; no shaping guarantee')
    if opportunity_regret:
        supported=[purpose[k] for k in retained if competence_support is None or f.roots[k][0] in competence_support]
        spread=max(supported)-min(supported) if supported else 0.
        metadata['purpose_band']=dict(fraction=max_regret,spread=spread,supported_plans=len(supported),
            semantics='fraction of sampled purpose opportunity in full personality tier; not calibrated victory risk')
        f=replace(f,max_regret=max_regret*spread)
    return replace(f,contexts=(fc,),roots={k:v for k,v in f.roots.items() if k in retained},
        purpose={k:v for k,v in purpose.items() if k in retained},audit=metadata,return_frame=frame,return_means=return_means)


class PersonaProgressWatch(ProgressWatch):
    """Do not expire the owner's entire immediate strongest-principle tier."""
    def __init__(self,scope,config=None):
        # Persist the behavior in the ordinary ProgressWatch checkpoint. The
        # DecisionLoop restores that base type, so subclass-only masks would
        # otherwise disappear silently after save/restore.
        super().__init__(scope,replace(config or ProgressConfig(),preserve_persona_tier=True))
