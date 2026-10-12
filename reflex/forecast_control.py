"""Finite feedback planning and independent scenario checks for one owner.

Only public world callbacks enter this controller. Future controllers use a
separate bank, never the trajectory seed. One nested level is an approximation
of future replanning, not recursive optimal play. Validation is a sampled
decision check, not a calibrated confidence interval or a victory guarantee.
"""
from dataclasses import dataclass, replace
import copy
import math
import numpy as np
from .actor_model import ActorModel
from .core import compile_batch, digest, MAX_OUTCOMES
from .deliberation import select
from .flow_rollout import rollout
from .purpose_plan import goal_forecast, Goal
from .decision_loop import THRESHOLDS,Request
from .model_observer import ModelObserver,ModelFeedback


class _DirectPolicyObserver(ModelObserver):
    """Validate the explicit pure-Policy contract without copying empty memory."""
    def __init__(self,owner,root_request,request_for,feedback,validate):
        self.owner=owner;self.root_request=root_request;self.request_for=request_for;self.feedback=feedback;self.validate=validate

    def fork(self):return copy.copy(self)

    def prepare(self,state,context,*,prepared=False):
        req=self.root_request if prepared else self.request_for(copy.deepcopy(context));self.validate(req)
        if digest(context['scope'])!=self.owner:raise ValueError('owned direct continuation required')
        if (not isinstance(req,Request) or req.context['tick']!=context['tick'] or
            {k:v for k,v in req.context.items() if k!='state'}!={k:v for k,v in context.items() if k!='state'} or
            not prepared and req.context['state']!=context['state']):
            raise ValueError('same public direct continuation request required')
        allowed={a['id'] for a in context['actions'] if a['legal'] and not a['known_failure']}
        return context,req,allowed,None

    def advance(self,before,after,key,context,request):
        fb=self.feedback(copy.deepcopy(before),copy.deepcopy(after),key,copy.deepcopy(context))
        if (not isinstance(fb,ModelFeedback) or fb.purpose is not None or fb.needs is not None or fb.maintained or fb.completed
            or fb.observed is not None or fb.event is not None or fb.social_events):
            raise ValueError('direct continuation cannot omit owned feedback updates')

    def record(self):return dict(owner=self.owner,semantics='explicit unlearned immediate Policy continuation')


@dataclass(frozen=True)
class ForecastBudget:
    horizon: int = 4
    seeds: tuple = (401,402,403,404)
    future_horizon: int = 2
    future_seeds: tuple = (601,602)
    future_levels: int = 0
    future_replans: int = 1
    validation_seeds: tuple = ()
    extra_validation_seeds: tuple = ()
    audit_seeds: tuple = ()
    purpose_challenger: bool = False
    reference_horizon: int = 2
    max_regret: float = .15
    uncertainty_weight: float = 1.
    direct_continuation: bool = False

    def __post_init__(self):
        for h in (self.horizon,self.future_horizon,self.reference_horizon):
            if type(h) is not int or not 1<=h<=16:raise ValueError('bounded public horizon required')
        if type(self.future_levels) is not int or not 0<=self.future_levels<=2:raise ValueError('bounded controller nesting required')
        if type(self.future_replans) is not int or not 1<=self.future_replans<=16:raise ValueError('bounded future replans required')
        if type(self.direct_continuation) is not bool:raise ValueError('explicit direct continuation mode required')
        for bank in (self.seeds,self.future_seeds,self.validation_seeds,self.extra_validation_seeds,self.audit_seeds):
            if not isinstance(bank,tuple) or len(bank)>MAX_OUTCOMES or any(type(s) is not int for s in bank) or len(set(bank))!=len(bank):
                raise ValueError('distinct bounded scenario bank required')
        if not self.seeds or not self.future_seeds:raise ValueError('nonempty discovery and future banks required')
        if set(self.seeds)&set(self.future_seeds) or set(self.validation_seeds)&(set(self.seeds)|set(self.future_seeds)):
            raise ValueError('separate discovery, future and validation banks required')
        if self.validation_seeds and len(self.validation_seeds)<2:raise ValueError('paired variation requires two validation scenarios')
        if self.extra_validation_seeds and (not self.validation_seeds or len(self.validation_seeds)+len(self.extra_validation_seeds)>MAX_OUTCOMES or
            set(self.extra_validation_seeds)&(set(self.seeds)|set(self.future_seeds)|set(self.validation_seeds))):
            raise ValueError('bounded distinct extra validation scenarios required')
        if type(self.purpose_challenger) is not bool:raise ValueError('explicit purpose challenger required')
        if self.purpose_challenger and (not self.validation_seeds or len(self.audit_seeds)<2):raise ValueError('challenger selection requires a separate final audit bank')
        if self.audit_seeds and (not self.purpose_challenger or set(self.audit_seeds)&(set(self.seeds)|set(self.future_seeds)|set(self.validation_seeds)|set(self.extra_validation_seeds))):
            raise ValueError('independent challenger audit scenarios required')
        for value,upper in ((self.max_regret,2.),(self.uncertainty_weight,4.)):
            if isinstance(value,bool) or not math.isfinite(value) or not 0<=value<=upper:raise ValueError('bounded decision check required')


def restrict(f,key):
    c=copy.deepcopy(f.contexts[0]);c['actions']=[a for a in c['actions'] if a['id']==key]
    return replace(f,contexts=(c,),roots={key:f.roots[key]},purpose={key:f.purpose[key]},continuity=None)


def checked_choice(c,candidate,reference,validation,policy,*,uncertainty_weight=1.,appraisal=None,allowed=None):
    """Freeze discovery choices before scoring an independent paired bank.

    Keep validation's full strongest-principle tier and purpose corridor. A
    candidate may spend some purpose return for its persona, but not an
    unmeasured uncertainty allowance on top of that explicit corridor. The
    one-standard-error penalty is a heuristic, not a statistical certificate.
    """
    permission=None if allowed is None else [allowed]
    extra={} if appraisal is None else dict(appraisal=[appraisal])
    votes=[]
    for f in (candidate,reference,validation):
        if (f.target,f.audit['unit'])!=(validation.target,validation.audit['unit']):raise ValueError('same purpose and unit required')
        _,v=select([c],f,policy,permission,**extra);votes.append(v)
    if set(validation.audit.get('scenario_seeds',())) & (set(candidate.audit.get('scenario_seeds',()))|set(reference.audit.get('scenario_seeds',()))):
        raise ValueError('validation bank must be independent of discovery')
    new_key,old_key,held=(v.get('selected_plan') for v in votes)
    by_root={roots[0]:key for key,roots in validation.roots.items()}
    new=None if new_key is None else by_root.get(candidate.roots[new_key][0])
    old=None if old_key is None else by_root.get(reference.roots[old_key][0])
    report=dict(candidate=new,reference=old,validation_choice=held,accepted=False,
        discovery_candidate=new_key,discovery_reference=old_key,
        candidate_root=None if new_key is None else candidate.roots[new_key][0],
        reference_root=None if old_key is None else reference.roots[old_key][0],
        uncertainty_weight=uncertainty_weight,semantics='independent paired sampled check; no probability guarantee')
    chosen=held
    if new_key is None or old_key is None or held is None:report['reason']='no comparable supported decision'
    elif new is None or old is None:
        report['reason']='validation strongest-principle tier differs; use its ordinary persona vote'
    else:
        roots={k:validation.roots[k][0] for k in validation.roots}
        goals=validation.audit['endpoint']
        if len(goals[new])!=len(goals[old]):raise ValueError('aligned scenario count required')
        # These branches were evaluated in one bank, with equal paired weights.
        confidence={a['id']:a['confidence'] for a in c['actions']}
        value=lambda g,k:g['value']*(g['positive_confidence']*confidence[roots[k]] if g['value']>0 else 1.)
        delta=np.array([value(a,new)-value(b,old) for a,b in zip(goals[new],goals[old])])
        mean=float(delta.mean());se=float(delta.std(ddof=1)/math.sqrt(len(delta))) if len(delta)>1 else 0.
        best=max(validation.purpose[k] for k in validation.roots if allowed is None or roots[k] in allowed)
        viable={k for k in validation.roots if (allowed is None or roots[k] in allowed) and validation.purpose[k]>=best-validation.max_regret-1e-12}
        report.update(paired_goal_delta=delta.tolist(),mean_delta=mean,standard_error=se,
            candidate_in_corridor=new in viable,reference_in_corridor=old in viable,
            lower_gain=mean-uncertainty_weight*se)
        if new==old:
            chosen=new;report.update(accepted=True,reason='same discovery root')
        elif new in viable and (old not in viable or mean-uncertainty_weight*se>=-validation.max_regret-1e-12):
            # Re-score both frozen options with the owner's ordinary utility.
            cc=copy.deepcopy(validation.contexts[0]);cc['actions']=[a for a in cc['actions'] if a['id'] in (new,old)]
            pair=replace(validation,contexts=(cc,),roots={k:validation.roots[k] for k in (new,old)},purpose={k:validation.purpose[k] for k in (new,old)})
            _,vote=select([c],pair,policy,permission,**extra);chosen=vote.get('selected_plan',held)
            report.update(accepted=chosen==new,reason='independent paired persona vote')
        elif old in viable:
            chosen=old;report['reason']='uncertain candidate gain; retain planning reference'
        else:report['reason']='both discovery choices outside validation corridor; use its ordinary persona vote'
    metadata=copy.deepcopy(validation.audit);metadata['forecast_check']=report
    f=replace(validation,audit=metadata)
    return f if chosen is None else restrict(f,chosen)


class ForecastController:
    """Callback bundle: rules/feedback belong to a game, preference to its actor."""
    def __init__(self,*,observe,advance,terminal,assess,request_for,feedback,unit,target,budget=None):
        self.observe=observe;self.advance=advance;self.terminal=terminal;self.assess=assess
        self.request_for=request_for;self.feedback=feedback;self.unit=unit;self.target=target
        self.budget=budget or ForecastBudget();self.work=0

    def evaluate(self,c,initial,actor,request,budget,*,roots=None):
        future=None
        if budget.future_levels:
            child=replace(budget,horizon=budget.future_horizon,seeds=budget.future_seeds,
                future_seeds=budget.seeds,future_levels=budget.future_levels-1,validation_seeds=(),extra_validation_seeds=(),audit_seeds=(),purpose_challenger=False)
            future=lambda own,req,public:self.planner(own,req,public,budget=child)
        observe=self.observe
        if budget.direct_continuation and future is None:
            def plain_request(req):
                if (req.reader is not None or req.purpose is not None or req.value_proposal is not None or req.social_bindings is not None
                    or req.invalidate or any(b.estimated for b in req.bindings.values()) or any(req.exact.values())):
                    raise ValueError('direct continuation requires unmodified immediate Policy requests')
            if actor.memory.entries or any(x is not None for x in (actor.predictor,actor.pressure,actor.progress,actor.social)):
                raise ValueError('direct continuation cannot omit owned learned/dynamic components')
            plain_request(request)
            def observe(public,state):
                cc=self.observe(public,state);plain_request(self.request_for(copy.deepcopy(cc)));return cc
            model=_DirectPolicyObserver(actor.owner,request,self.request_for,self.feedback,plain_request)
        else:
            model=ActorModel(actor,root_request=request,request_for=self.request_for,feedback=self.feedback,future_planner=future,
                future_planner_limit=budget.future_replans if future is not None else None)
        saved=digest(actor.record())
        paths,audit=rollout(c,initial,observe=observe,advance=self.advance,terminal=self.terminal,
            assess=self.assess,horizon=budget.horizon,seeds=budget.seeds,observer=model,policy=actor.policy,roots=roots)
        self.work+=sum(len(t['actions']) for rows in audit.values() for t in rows)
        local=copy.deepcopy(c)
        if roots is not None:local['actions']=[a for a in local['actions'] if a['id'] in roots]
        f=goal_forecast(local,paths,audit,horizon=budget.horizon,unit=self.unit,target=self.target,
            max_regret=budget.max_regret,policy=actor.policy,settlement_weight='absolute')
        f=replace(f,audit=dict(f.audit,scenario_seeds=list(budget.seeds),future_levels=budget.future_levels,
            future_horizon=budget.future_horizon,future_seeds=list(budget.future_seeds)))
        if digest(actor.record())!=saved:raise AssertionError('hypothetical controller mutated real owner')
        return f,audit

    def planner(self,actor,request,initial,*,budget=None,captured=None):
        budget=budget or self.budget
        def call(contexts,decisions):
            if len(contexts)!=1:raise ValueError('one owner forecast controller required')
            c=contexts[0];discovery_budget=replace(budget,future_levels=0) if budget.validation_seeds else budget
            f,audit=self.evaluate(c,initial,actor,request,discovery_budget)
            discovery=f;discovery_audit=audit
            if budget.validation_seeds:
                rb=replace(budget,horizon=budget.reference_horizon,future_levels=0,validation_seeds=(),extra_validation_seeds=(),audit_seeds=(),purpose_challenger=False)
                reference,_=self.evaluate(c,initial,actor,request,rb)
                vb=replace(budget,seeds=budget.validation_seeds,validation_seeds=(),extra_validation_seeds=(),audit_seeds=(),purpose_challenger=False)
                # Root social appraisal is the same one used at real execution.
                stage=actor._prepare(request)
                extra={} if stage['social_scores'] is None else dict(appraisal=[stage['social_scores']])
                _,a=select([c],discovery,actor.policy,[stage['progress_allowed']],**extra)
                _,b=select([c],reference,actor.policy,[stage['progress_allowed']],**extra)
                heads={f.roots[k][0] for f,k in ((discovery,a.get('selected_plan')),(reference,b.get('selected_plan'))) if k is not None}
                # The short reference cannot silently override the discovery's
                # full strongest-principle tier. A change outside that tier is
                # unprotected rather than forcing a different character.
                full_tier={row[0] for row in discovery.roots.values()}
                heads&=full_tier
                challenger=None
                if budget.purpose_challenger:
                    valid=[k for k,roots in discovery.roots.items() if roots[0] in stage['progress_allowed']]
                    if valid:
                        challenger=max(valid,key=lambda k:(discovery.purpose[k],k))
                        heads.add(discovery.roots[challenger][0])
                    # Discovery's noisy estimate is not the character's fixed
                    # value system. Re-evaluate the persona-selected reference
                    # as well; final banks still apply unchanged value weights.
                    if b.get('selected_plan') is not None:heads.add(reference.roots[b['selected_plan']][0])
                if not heads:
                    if captured is not None:captured.update(forecast=f,audit=audit,discovery=discovery,discovery_audit=discovery_audit)
                    return f
                validated,audit=self.evaluate(c,initial,actor,request,vb,roots=tuple(sorted(heads)))
                if budget.purpose_challenger:
                    _,vote=select([c],validated,actor.policy,[stage['progress_allowed']],**extra)
                    selected=vote.get('selected_plan')
                    if selected is not None:
                        vetted=restrict(validated,selected)
                        audited_heads={vetted.roots[selected][0]}
                        if b.get('selected_plan') is not None:audited_heads.add(reference.roots[b['selected_plan']][0])
                        ab=replace(vb,seeds=budget.audit_seeds)
                        final,audit=self.evaluate(c,initial,actor,request,ab,roots=tuple(sorted(audited_heads)))
                        f=checked_choice(c,vetted,reference,final,actor.policy,
                            uncertainty_weight=budget.uncertainty_weight,appraisal=stage['social_scores'],allowed=stage['progress_allowed'])
                        f=replace(f,audit=dict(f.audit,purpose_challenger=dict(discovery_roots=sorted(heads),
                            goal_root=None if challenger is None else discovery.roots[challenger][0],
                            vetted_root=vetted.roots[selected][0],final_audit_roots=sorted(audited_heads),
                            selection_seeds=list(budget.validation_seeds),audit_seeds=list(budget.audit_seeds),
                            semantics='three frozen proposals, independent persona selection, separate paired audit; no stronger-play guarantee')))
                        if captured is not None:captured.update(forecast=f,audit=audit,discovery=discovery,discovery_audit=discovery_audit)
                        return f
                f=checked_choice(c,discovery,reference,validated,actor.policy,
                    uncertainty_weight=budget.uncertainty_weight,appraisal=stage['social_scores'],allowed=stage['progress_allowed'])
                check=f.audit['forecast_check']
                if (budget.extra_validation_seeds and check['reason']=='uncertain candidate gain; retain planning reference'
                    and check.get('candidate_in_corridor') and
                    check['mean_delta']+budget.uncertainty_weight*check['standard_error']>=-budget.max_regret-1e-12):
                    # Add scenarios only when uncertainty can change this
                    # frozen comparison. No new candidates or game weights.
                    eb=replace(vb,seeds=budget.validation_seeds+budget.extra_validation_seeds)
                    expanded,audit=self.evaluate(c,initial,actor,request,eb,roots=tuple(sorted(heads)))
                    f=checked_choice(c,discovery,reference,expanded,actor.policy,
                        uncertainty_weight=budget.uncertainty_weight,appraisal=stage['social_scores'],allowed=stage['progress_allowed'])
                    f=replace(f,audit=dict(f.audit,adaptive_validation=dict(before=check,
                        added_scenarios=len(budget.extra_validation_seeds),replayed_scenarios=len(budget.validation_seeds),
                        total_scenarios=len(eb.seeds),reason='uncertainty could reverse frozen candidate/reference comparison')))
            if captured is not None:captured.update(forecast=f,audit=audit,discovery=discovery,discovery_audit=discovery_audit)
            return f
        return call


class ForecastLedger:
    """Delayed factual forecast scoring; no unchosen action or fake experience.

    Compare goal@declared-horizon with observed goal at that horizon (absorbing
    real settlement closes earlier). A running proxy is not a win probability.
    Actual intervening replanning is recorded, not silently treated as the
    modeled action sequence. These scores do not automatically veto a decision:
    good prediction alone proves neither decision value nor causal credit.
    """
    def __init__(self,scope,capacity=64):
        if type(capacity) is not int or not 1<=capacity<=1024:raise ValueError('bounded ledger required')
        self.owner=digest(scope);self.capacity=capacity;self.pending=[];self.scored=[];self.last_step=0

    def commit(self,c,step,root,horizon,rows,current_goal):
        if digest(c['scope'])!=self.owner:raise ValueError('owned forecast required')
        if type(step) is not int or step!=self.last_step or type(horizon) is not int or not 1<=horizon<=16:raise ValueError('same-clock finite forecast required')
        if len(self.pending)>=self.capacity:raise ValueError('resolve or explicitly censor pending forecasts')
        if any(p['start']==step for p in self.pending):raise ValueError('one selected forecast per owner step required')
        g=Goal(**current_goal);samples=[];programs=[]
        for row in rows:
            end=Goal(**row['assessment'])
            if end.target!=g.target or not row['actions'] or row['actions'][0]!=root:raise ValueError('same root and goal target required')
            samples.append(end.value);programs.append(tuple(row['actions']))
        if not samples:raise ValueError('nonempty frozen forecast required')
        self.pending.append(dict(start=step,due=step+horizon,root=root,target=g.target,samples=samples,
            persistence=g.value,programs=programs,actual_actions=[]))

    def observe(self,step,root,goal,*,terminal=False):
        if type(step) is not int or step!=self.last_step+1 or type(terminal) is not bool:raise ValueError('consecutive factual clock required')
        g=Goal(**goal)
        if terminal!=(g.status!='running'):raise ValueError('explicit absorbing goal settlement required')
        if any(p['target']!=g.target for p in self.pending):raise ValueError('same delayed target required')
        self.last_step=step;keep=[];new=[]
        for p in self.pending:
            p=copy.deepcopy(p);p['actual_actions'].append(root)
            if step>=p['due'] or terminal:
                samples=np.array(p['samples']);actual=(g.value<=THRESHOLDS)
                prediction=(samples[:,None]<=THRESHOLDS).mean(0)
                loss=float(np.mean((prediction-actual)**2));prior=float(np.mean(((p['persistence']<=THRESHOLDS)-actual.astype(float))**2))
                matches=sum(tuple(p['actual_actions'])==program[:len(p['actual_actions'])] for program in p['programs'])
                row=dict(start=p['start'],due=p['due'],resolved=step,target=g.target,root=p['root'],
                    predicted_mean=float(samples.mean()),observed=g.value,status=g.status,
                    forecast_loss=loss,persistence_loss=prior,gain=prior-loss,
                    matching_action_branches=matches,branches=len(samples),actual_actions=p['actual_actions'])
                new.append(row)
            else:keep.append(p)
        self.pending=keep;self.scored=(self.scored+new)[-self.capacity:]
        return new

    def record(self):
        return copy.deepcopy(dict(owner=self.owner,capacity=self.capacity,last_step=self.last_step,pending=self.pending,scored=self.scored))
