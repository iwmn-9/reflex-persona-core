"""Common observation -> forecast -> decision -> evidence correction lifecycle.

Games supply perceived snapshots, method/condition equivalence and observed
effects. Prequential finite-CDF scores monitor bounded empirical updates and
gate additional reader overrides. This is
an engineering validation signal, not a guarantee of optimal play/calibration.
Only forecasts for the SAME outcome target may be scored against feedback.
"""
from collections import OrderedDict, deque
from dataclasses import dataclass, asdict,replace
import copy
import numpy as np
from .core import FEATURES, Policy, compile_batch, digest, identifier
from .judgment import Binding, OutcomeMemory, ReadControl, avoid_waste
from .planning import vector
from .routes import RouteState,choose_route
from .pressure import NeedPressure
from .progress import ProgressWatch,PurposeRequest

THRESHOLDS=np.linspace(-1,1,9)[1:-1]


def cdf_score(outcomes,observed,names):
    """Mean squared error of finite CDF threshold events; bounded [0,1]."""
    idx=[FEATURES.index(name) for name in names]
    if not idx: return None
    samples=np.array([vector(row)[idx] for row in outcomes])
    probability=np.array([row['p'] for row in outcomes])
    forecast=np.einsum('k,kdt->dt',probability,samples[:,:,None]<=THRESHOLDS)
    actual=np.asarray(observed)[idx,None]<=THRESHOLDS
    return float(np.mean((forecast-actual)**2))


class EvidenceGate:
    """Bounded, before-update prediction comparisons against a live fallback.

    Four informative comparisons before adoption; recent comparative failures
    revoke adoption. Good old evidence cannot hide four new bad comparisons.
    Equal predictions give no evidence. No reward/value preference is trained.
    """
    def __init__(self,capacity=64,window=12,min_trials=4,margin=.02):
        if type(capacity) is not int or not 1<=capacity<=1024 or type(window) is not int or not 4<=window<=64:
            raise ValueError('bounded evidence storage required')
        if type(min_trials) is not int or not 4<=min_trials<=window or not 0<margin<1:
            raise ValueError('valid adoption controls required')
        self.capacity=capacity;self.window=window;self.min_trials=min_trials;self.margin=margin
        self.entries=OrderedDict()

    def status(self,key):
        e=self.entries.get(key)
        return dict(state='unobserved' if e is None else e['state'],
            trials=0 if e is None else len(e['gains']),
            recent_gain=0. if e is None else float(np.mean(e['gains'])))

    def accepted(self,key): return self.status(key)['state']=='active'

    def compare(self,key,prior,candidate,observed,names):
        a=cdf_score(prior,observed,names);b=cdf_score(candidate,observed,names)
        if a is None: return dict(scored=False,reason='no comparable observed dimensions')
        # Compare distributions, not just this realized loss. Equal forecasts
        # must not pad the sample count or prevent revocation of a bad model.
        signature=lambda rows: tuple(float(sum(r['p']*(vector(r)[FEATURES.index(f)]<=t) for r in rows)) for f in names for t in THRESHOLDS)
        if signature(prior)==signature(candidate): return dict(scored=False,reason='identical forecasts')
        return self._update(key,a,b)

    def categorical(self,key,prior,candidate,observed):
        if set(prior)!=set(candidate) or observed not in prior:raise ValueError('same revealed response space required')
        a=np.array(list(prior.values()),float);b=np.array([candidate[k] for k in prior],float)
        if not np.isfinite(a).all() or not np.isfinite(b).all() or (a<0).any() or (b<0).any() or abs(a.sum()-1)>1e-8 or abs(b.sum()-1)>1e-8:
            raise ValueError('normalized response forecast required')
        if np.array_equal(a,b):return dict(scored=False,reason='identical forecasts')
        actual=np.array([k==observed for k in prior],float)
        return self._update(key,float(np.mean((a-actual)**2)),float(np.mean((b-actual)**2)))

    def _update(self,key,a,b):
        e=self.entries.pop(key,dict(state='trial',gains=deque(maxlen=self.window)))
        old=e['state'];e['gains'].append(a-b)
        ready=len(e['gains'])>=self.min_trials
        recent=float(np.mean(list(e['gains'])[-self.min_trials:]))
        if ready and recent < -self.margin: e['state']='revoked'
        elif ready and recent > self.margin and float(np.mean(e['gains']))>self.margin: e['state']='active'
        elif e['state']=='active' and ready and float(np.mean(e['gains']))<=0: e['state']='trial'
        self.entries[key]=e
        while len(self.entries)>self.capacity:self.entries.popitem(last=False)
        return dict(scored=True,prior_loss=a,candidate_loss=b,gain=a-b,
            before=old,after=e['state'],revoked=old=='active' and e['state']!='active')


@dataclass(frozen=True)
class Reading:
    context: dict
    nodes: int
    key: str
    target: str='immediate'
    responses: dict | None=None
    known_conditionals: bool=False
    response_candidate: dict | None=None


@dataclass(frozen=True)
class Request:
    context: dict
    bindings: dict
    exact: dict
    reader: object=None
    maintains_advantage: bool=False
    threatened: bool=False
    outcome_target: str='immediate'
    invalidate: tuple=()
    purpose: object=None


@dataclass(frozen=True)
class IntentRequest:
    """Game supplies route opportunities and builds same-target action effects."""
    context: dict
    build: object
    uncertainty: float=0.
    route_planner: object=None


def _same_exact_distribution(left,right,names):
    idx=[FEATURES.index(f) for f in names]
    def mass(rows):
        result={}
        for r in rows:
            if not r['p']:continue
            key=tuple(vector(r)[idx]);result[key]=result.get(key,0.)+r['p']
        return result
    a,b=mass(left),mass(right)
    return a.keys()==b.keys() and all(abs(a[k]-b[k])<=1e-8 for k in a)


def _validate_reading(base,proposal,bindings,target,cap):
    if not isinstance(proposal,Reading) or type(proposal.nodes) is not int or not 0<=proposal.nodes<=cap:
        raise ValueError('bounded Reading with counted nodes required')
    identifier(proposal.key)
    if proposal.target!=target:raise ValueError('forecast/feedback outcome target mismatch')
    if type(proposal.known_conditionals) is not bool:raise ValueError('explicit conditional-rule declaration required')
    if proposal.known_conditionals:
        p=proposal.responses
        if not isinstance(p,dict) or not p or any(not isinstance(k,str) or not k for k in p):raise ValueError('response forecast required')
        values=np.array(list(p.values()),float)
        if not np.isfinite(values).all() or (values<0).any() or abs(values.sum()-1)>1e-8:raise ValueError('normalized response forecast required')
        for a in proposal.context['actions']:
            if len(a['outcomes'])!=len(p) or any(abs(row['p']-v)>1e-8 for row,v in zip(a['outcomes'],p.values())):
                raise ValueError('one conditional outcome per declared response, in response order required')
    c=copy.deepcopy(proposal.context);compile_batch([c])
    for key in base:
        if key not in ('actions','facts') and c.get(key)!=base[key]:raise ValueError('reader changed actor/observation contract: '+key)
    if any(c['facts'].get(k)!=v for k,v in base['facts'].items()):raise ValueError('reader changed observed facts')
    old={a['id']:a for a in base['actions']};new={a['id']:a for a in c['actions']}
    if old.keys()!=new.keys():raise ValueError('reader changed action identities')
    for key,a in old.items():
        if {k:v for k,v in a.items() if k!='outcomes'}!={k:v for k,v in new[key].items() if k!='outcomes'}:
            raise ValueError('reader changed root legality/certainty/fees')
        exact=set(FEATURES)-set(bindings[key].estimated)
        if not _same_exact_distribution(a['outcomes'],new[key]['outcomes'],exact):
            raise ValueError('reader changed non-estimated effect distribution')
    return c


class DecisionLoop:
    """One owner per game/episode/NPC. Public feedback, never world access.

    prepare/decide/commit is transactional across a batch. observe resolves one
    selected ticket; missing/censored feedback can be explicitly abandoned.
    Internal intent state is carried automatically. JSON boundary is audited;
    this path does not replace the existing fast numeric Population runtime.
    """
    def __init__(self,context,policy=None,read_control=None,capacity=64,predictor=None,pressure=None,progress=None):
        compile_batch([context])
        self.scope=copy.deepcopy(context['scope']);self.owner=digest(self.scope)
        self.personality=copy.deepcopy(context['personality']);self.values=copy.deepcopy(context['values']);self.seed=context['seed']
        self.policy=policy or Policy();self.read_control=read_control or ReadControl(max_nodes=16)
        self.memory=OutcomeMemory(self.scope,capacity);self.evidence=EvidenceGate(capacity)
        self.state=None;self.pending=None;self.last_tick=-1;self.strategy=RouteState()
        self.predictor=copy.deepcopy(predictor)
        if predictor is not None and getattr(predictor,'owner',None)!=self.owner:raise ValueError('predictor owner mismatch')
        if pressure is not None and (not isinstance(pressure,NeedPressure) or pressure.owner!=self.owner):raise ValueError('pressure owner mismatch')
        self.pressure=copy.deepcopy(pressure)
        if progress is not None and (not isinstance(progress,ProgressWatch) or progress.owner!=self.owner):raise ValueError('progress owner mismatch')
        self.progress=copy.deepcopy(progress)

    def _prepare(self,request):
        if not isinstance(request,Request):raise ValueError('Request required')
        if self.pending is not None:raise ValueError('observe or abandon selected trial before advancing')
        c=copy.deepcopy(request.context)
        if digest(c['scope'])!=self.owner:raise ValueError('wrong actor owner')
        if c['personality']!=self.personality or c['values']!=self.values:raise ValueError('fixed personality/value axes changed')
        if c['seed']!=self.seed:raise ValueError('actor random stream changed')
        if c['tick']<=self.last_tick:raise ValueError('tick must increase')
        if type(request.maintains_advantage) is not bool or type(request.threatened) is not bool:
            raise ValueError('explicit boolean game evidence required')
        identifier(request.outcome_target)
        # This loop's learner compares one observed outcome. A future target
        # requires a separate delayed-feedback contract, never instant feedback.
        if request.outcome_target!='immediate':raise ValueError('delayed outcome targets require a separate evaluator')
        if self.state is not None:c['state']=copy.deepcopy(self.state)
        if self.pressure is not None:c=self.pressure.apply(c)
        compile_batch([c])
        allowed={a['id'] for a in c['actions'] if a['legal'] and not a['known_failure']};progress_audit=None
        if self.progress is not None:allowed,progress_audit=self.progress.mask(c,request.purpose)
        elif request.purpose is not None:raise ValueError('purpose request requires configured progress watch')
        if self.progress is not None and self.progress.config.proof_margin is not None and (request.reader is not None or self.predictor is not None):
            raise ValueError('recovery proof currently requires the joint planner path without a separate reader')
        bindings=copy.deepcopy(request.bindings);exact=copy.deepcopy(request.exact)
        if set(exact)!={a['id'] for a in c['actions']} or any(set(v)-set(FEATURES) for v in exact.values()):
            raise ValueError('exact effect mask for every action required')
        if any(not isinstance(b,Binding) for b in bindings.values()):raise ValueError('explicit method bindings required')
        if any(set(b.estimated)&set(exact.get(k,())) for k,b in bindings.items()):raise ValueError('exact effects cannot be learned')
        memory=copy.deepcopy(self.memory);evidence=copy.deepcopy(self.evidence)
        for method,situation in request.invalidate:
            identifier(method);identifier(situation)
            memory.entries.pop((method,situation),None)
            evidence.entries.pop('experience:'+digest([method,situation]),None)
        shadow,stats=memory.prepare(c,bindings);effective=copy.deepcopy(shadow)
        for a in effective['actions']:
            b=bindings[a['id']];key='experience:'+digest([b.method,b.situation])
            stats[a['id']]['gate']=evidence.status(key)
            # Preserve the existing capped empirical fallback. Prediction-error
            # diagnostics alone cannot safely revoke it: the raw prior may be
            # even more harmful in the counterfactual action trajectory.
            stats[a['id']]['used']=True
            stats[a['id']]['predictive_warning']=evidence.status(key)['state']=='revoked'
        response_trial=None
        if self.predictor is not None and getattr(self.predictor,'known_conditionals',False):
            prior,candidate=self.predictor.forecasts(copy.deepcopy(c))
            response_trial=dict(key='response:'+digest(self.predictor.key),prior=prior,candidate=candidate)
        return dict(loop=self,request=request,raw=c,shadow=shadow,effective=effective,
            bindings=bindings,exact=exact,memory=memory,evidence=evidence,learning=stats,reading=None,response_trial=response_trial,
            progress_allowed=allowed,progress_audit=progress_audit)

    @staticmethod
    def decide_batch(items,stochastic=True,planner=None):
        """Batch all reflex scoring; readers are bounded and optional per actor."""
        if not items:return []
        if len({loop.owner for loop,_ in items})!=len(items):raise ValueError('duplicate actor owner')
        policy=items[0][0].policy
        if any(not np.array_equal(loop.policy.residual,policy.residual) or loop.policy.principle_priority!=policy.principle_priority for loop,_ in items):
            raise ValueError('batch must share policy coefficients')
        stages=[]
        for loop,req in items:
            strategy=loop.strategy;route_audit=None
            if isinstance(req,IntentRequest):
                rc=copy.deepcopy(req.context)
                if (rc['personality']!=loop.personality or rc['values']!=loop.values or rc['seed']!=loop.seed or
                    rc['scope']['episode']!=loop.scope['episode'] or rc['scope']['npc']!=loop.scope['npc']):
                    raise ValueError('strategy/action actor contract mismatch')
                if loop.pressure is not None:rc=loop.pressure.apply_route(rc)
                strategy,route_audit=choose_route(rc,loop.strategy,uncertainty=req.uncertainty,policy=policy)
                if req.route_planner is not None:
                    from .deliberation import select
                    forecast=req.route_planner(copy.deepcopy(rc))
                    decisions,audit=select([rc],forecast,policy)
                    if decisions is not None:
                        chosen=decisions[0]['action_id'];old=loop.strategy
                        switched=old.chosen is not None and chosen!=old.chosen
                        strategy=replace(strategy,chosen=chosen,switches=old.switches+int(switched))
                        route_audit.update(chosen=chosen,switched=switched,reason='horizon purpose and personality')
                    route_audit['deliberation']=audit
                built=req.build(strategy.chosen)
                if not isinstance(built,Request) or built.context['tick']!=rc['tick']:raise ValueError('same-tick action Request required')
                req=built
            s=loop._prepare(req);s.update(strategy=strategy,route=route_audit)
            s['reader']=req.reader if req.reader is not None else loop.predictor
            stages.append(s)
        b=compile_batch([s['effective'] for s in stages])
        masks=np.array([[[name in s['exact'][key] for name in FEATURES] if j<len(b.ids[i]) else [False]*len(FEATURES)
            for j,key in enumerate(list(b.ids[i])+[None]*(b.legal.shape[1]-len(b.ids[i])))] for i,s in enumerate(stages)],dtype=bool)
        progress_mask=np.array([[k in s['progress_allowed'] for k in list(b.ids[i])+[None]*(b.legal.shape[1]-len(b.ids[i]))] for i,s in enumerate(stages)])
        guarded,removed=avoid_waste(replace(b,legal=b.legal&progress_mask),masks);numeric=policy.decide(guarded,stochastic)
        records=numeric.records(guarded)
        proof_indices=[i for i,s in enumerate(stages) if s['loop'].progress is not None and
            s['loop'].progress.config.proof_margin is not None and s['progress_audit']['applied']]
        baseline_records=records
        base_allowed=[s['progress_allowed'].copy() for s in stages]
        if proof_indices:
            for i in proof_indices:base_allowed[i]={a['id'] for a in stages[i]['effective']['actions'] if a['legal'] and not a['known_failure']}
            base_mask=np.array([[k in base_allowed[i] for k in list(b.ids[i])+[None]*(b.legal.shape[1]-len(b.ids[i]))] for i in range(len(stages))])
            unguarded,_=avoid_waste(replace(b,legal=b.legal&base_mask),masks)
            baseline_records=policy.decide(unguarded,stochastic).records(unguarded)
        for i,s in enumerate(stages):
            loop=s['loop'];req=s['request'];s['decision']=records[i]
            s['guard']=[b.ids[i][j] for j in np.flatnonzero(removed[i])]
            local=guarded.take([i])
            from .core import Decisions
            local_d=Decisions(**{name:getattr(numeric,name)[i:i+1] for name in Decisions.__dataclass_fields__})
            gate=loop.read_control.request(local,local_d,np.array([s['reader'] is not None]),
                np.array([req.maintains_advantage]),np.array([req.threatened]))
            audit=dict(requested=bool(gate['requested'][0]),nodes=0,adopted=False,reason='reflex')
            s['read_audit']=audit
            if not audit['requested']:continue
            proposal=s['reader'](copy.deepcopy(s['effective']),int(gate['nodes'][0]))
            if proposal is None:audit['reason']='no_complete_forecast';continue
            if proposal.known_conditionals and proposal.response_candidate is not None:
                # Evaluate each conditional rule effect once; choosing between
                # response forecasts only reweights this same known table.
                key='response:'+digest(proposal.key)
                s['baseline_read']=copy.deepcopy(proposal.context)
                c=copy.deepcopy(proposal.context)
                if set(proposal.response_candidate)!=set(proposal.responses):raise ValueError('same conditional response space required')
                probs=np.array(list(proposal.response_candidate.values()),float)
                if not np.isfinite(probs).all() or (probs<0).any() or abs(probs.sum()-1)>1e-8:raise ValueError('normalized candidate responses required')
                for a in c['actions']:
                    for row,response in zip(a['outcomes'],proposal.responses):row['p']=proposal.response_candidate[response]
                s['decision_trial_context']=_validate_reading(s['effective'],replace(proposal,context=c,responses=proposal.response_candidate),s['bindings'],req.outcome_target,int(gate['nodes'][0]))
                if s['evidence'].accepted(key) and s['evidence'].accepted(key+':decision'):
                    proposal=replace(proposal,context=c,responses=proposal.response_candidate)
            rc=_validate_reading(s['effective'],proposal,s['bindings'],req.outcome_target,int(gate['nodes'][0]))
            key=('response:' if proposal.known_conditionals else 'reader:')+digest(proposal.key)
            s['reading']=dict(context=rc,key=key,responses=copy.deepcopy(proposal.responses),known=proposal.known_conditionals)
            audit.update(nodes=proposal.nodes,gate=s['evidence'].status(key),reason='forecast_on_trial')
            if not proposal.known_conditionals and not s['evidence'].accepted(key):continue
            audit['response_model']='validated_hypotheses' if proposal.known_conditionals and s['evidence'].accepted(key) and s['evidence'].accepted(key+':decision') else 'bounded_recent_response'
            rb=compile_batch([rc]);rm=np.array([[[name in s['exact'][k] for name in FEATURES] for k in rb.ids[0]]])
            rb=replace(rb,legal=rb.legal&np.array([[k in s['progress_allowed'] for k in rb.ids[0]]]))
            rb,_=avoid_waste(rb,rm)
            if 'decision_trial_context' in s:
                # Competing choices are frozen before reveal. Known immediate
                # conditional rules can later compare purpose effects under
                # the SAME publicly revealed response. Never empirical samples.
                def read_choice(c):
                    bb=compile_batch([c])
                    mm=np.array([[[name in s['exact'][k] for name in FEATURES] for k in bb.ids[0]]])
                    bb=replace(bb,legal=bb.legal&np.array([[k in s['progress_allowed'] for k in bb.ids[0]]]))
                    bb,_=avoid_waste(bb,mm)
                    dd=policy.decide(bb,stochastic)
                    j=int(dd.action[0]);old_j=bb.ids[0].index(s['decision']['action_id'])
                    use=not dd.eligible[0,old_j] or dd.scores[0,j]-dd.scores[0,old_j]>.05
                    chosen=bb.ids[0][j] if use else s['decision']['action_id']
                    return copy.deepcopy(next(a['outcomes'] for a in c['actions'] if a['id']==chosen)),chosen
                bc=s.get('baseline_read',proposal.context)
                baseline,base_choice=read_choice(bc)
                candidate,candidate_choice=read_choice(s['decision_trial_context'])
                s['decision_trial']=dict(key=key+':decision',responses=list(proposal.responses),
                    baseline=baseline,candidate=candidate,base_choice=base_choice,candidate_choice=candidate_choice)
                audit['decision_gate']=s['evidence'].status(key+':decision')
            old=rb.ids[0].index(s['decision']['action_id'])
            # A model being good for A proves nothing about untried B. Require
            # same method/condition evidence for EVERY changed root decision.
            approved=np.array([[proposal.known_conditionals or s['evidence'].accepted(key+':'+digest([s['bindings'][k].method,s['bindings'][k].situation]))
                or j==old for j,k in enumerate(rb.ids[0])]])
            rb=replace(rb,legal=rb.legal&approved);rd=policy.decide(rb,stochastic)
            audit['unvalidated_roots']=[k for j,k in enumerate(rb.ids[0]) if not approved[0,j]]
            new=int(rd.action[0])
            gain=float(rd.scores[0,new]-rd.scores[0,old])
            if not rd.eligible[0,old] or gain>.05:
                s['decision']=rd.records(rb)[0];s['effective']=rc
                audit.update(adopted=True,reason='conditional_rule_persona_gain' if proposal.known_conditionals else 'validated_forecast_persona_gain',gain=gain)
            else:audit['reason']='no_material_persona_gain'
        # Optional joint horizon selection has a distinct target. Commit only
        # the selected legal immediate root to experience; simulated future
        # effects never become empirical observations or prediction diagnostics.
        forecast=None
        if planner is not None:
            from .deliberation import select
            contexts=[copy.deepcopy(s['effective']) for s in stages]
            forecast=planner(contexts,copy.deepcopy(baseline_records) if proof_indices else [copy.deepcopy(s['decision']) for s in stages])
            if forecast is not None:
                decisions,audit=select(contexts,forecast,policy,root_allowed=[s['progress_allowed'] for s in stages])
                if proof_indices:
                    original,old_audit=select(contexts,forecast,policy,root_allowed=base_allowed)
                    original=original or baseline_records
                    needed=any(original[i]['action_id'] not in stages[i]['progress_allowed'] for i in proof_indices)
                    # A game may supply extra constrained options when its
                    # ordinary finite proposal set cannot form a compatible
                    # recovery. Never relax personality tiers or invent roots.
                    if needed and decisions is None and getattr(planner,'recover',None) is not None:
                        recovery=planner.recover(copy.deepcopy(contexts),copy.deepcopy(original),
                            [s['progress_allowed'].copy() for s in stages])
                        if recovery is not None:
                            if recovery.horizon!=forecast.horizon or recovery.max_regret!=forecast.max_regret or recovery.target!=forecast.target:
                                raise ValueError('recovery must keep purpose target, horizon and persona corridor')
                            decisions,extra_audit=select(contexts,recovery,policy,root_allowed=[s['progress_allowed'] for s in stages])
                            extra_audit['recovery_options']=dict(before=copy.deepcopy(audit),expanded=True)
                            audit=extra_audit
                            old_audit['recovery_attempt']=copy.deepcopy(extra_audit)
                    old_roots=tuple(d['action_id'] for d in original)
                    base_purpose=old_audit.get('selected_purpose') if old_audit['adopted'] else max(
                        (v for k,v in forecast.purpose.items() if tuple(forecast.roots[k])==old_roots),default=None)
                    proof=dict(needed=needed,base_purpose=base_purpose,candidate_purpose=audit.get('selected_purpose'),
                        margin=max(stages[i]['loop'].progress.config.proof_margin for i in proof_indices))
                    approvals=[]
                    for i in proof_indices:
                        s=stages[i];s['progress_allowed'],s['progress_audit']=s['loop'].progress.mask(s['effective'],s['request'].purpose,proof)
                        approvals.append(s['progress_audit']['recovery']['approved'])
                    if not needed or not all(approvals):decisions,audit=original,old_audit
                for i,s in enumerate(stages):
                    if decisions is not None:s['decision']=decisions[i]
                    s['deliberation']=copy.deepcopy(audit)
        if proof_indices and forecast is None:
            for i,s in enumerate(stages):s['decision']=baseline_records[i]
            for i in proof_indices:
                s=stages[i];proof=dict(needed=True,base_purpose=None,candidate_purpose=None)
                s['progress_allowed'],s['progress_audit']=s['loop'].progress.mask(s['effective'],s['request'].purpose,proof)
        # A game may explicitly complete declined joint roots, but only within
        # each owner's unchanged immediate Policy tier and existing near-tie
        # band, after the progress proof has settled its final mask. This is
        # not horizon adoption and does not create simulated experience.
        if (planner is not None and getattr(planner,'complete_fallback',None) is not None
                and forecast is not None and not stages[0].get('deliberation',{}).get('adopted')):
            if any(s['reader'] is not None for s in stages):
                raise ValueError('root completion currently requires the joint planner path without a separate reader')
            from .root_completion import constrained_completion
            completed,completion_audit=constrained_completion(
                [s['effective'] for s in stages],[s['decision'] for s in stages],
                [s['progress_allowed'] for s in stages],masks,policy,planner.complete_fallback,
                [set((s['progress_audit'] or {}).get('blocked',{})) for s in stages])
            for s,d in zip(stages,completed):
                s['decision']=d;s['root_completion']=copy.deepcopy(completion_audit)
        # All validation/scoring, including every reader, completes BEFORE any
        # actor advances. Exceptions leave every original actor untouched.
        results=[]
        for s in stages:
            loop=s['loop'];c=s['effective'];d=s['decision']
            selected=d['action_id'];binding=s['bindings'][selected]
            select=lambda source:copy.deepcopy(next(a['outcomes'] for a in source['actions'] if a['id']==selected))
            ticket=s['memory'].commit(c,d,s['bindings'])
            pending=dict(ticket=ticket,action=selected,tick=c['tick'],binding=binding,
                prior=select(s['raw']),candidate=select(s['shadow']),raw=copy.deepcopy(s['raw']),
                exact=tuple(s['exact'][selected]),read=None,response_trial=s['response_trial'],decision_trial=s.get('decision_trial'))
            if loop.progress is not None:pending['purpose']=copy.deepcopy(s['request'].purpose)
            if s['reading'] is not None:
                pending['read']=dict(key=s['reading']['key'],prior=select(s['shadow'] if s['learning'][selected]['used'] else s['raw']),
                    candidate=select(s['reading']['context']),responses=s['reading']['responses'],known=s['reading']['known'])
            s['pending']=pending;s['ticket']=ticket
        for s in stages:
            loop=s['loop'];loop.memory=s['memory'];loop.evidence=s['evidence'];loop.pending=s['pending']
            loop.state=copy.deepcopy(s['decision']['next_state']);loop.last_tick=s['effective']['tick']
            loop.strategy=s['strategy']
            result=dict(ticket=s['ticket'],decision=s['decision'],context=s['effective'],
                learning=s['learning'],reading=s['read_audit'],strategy=s['route'],waste_removed=s['guard'])
            if 'deliberation' in s:result['deliberation']=s['deliberation']
            if 'root_completion' in s:result['root_completion']=s['root_completion']
            if s['progress_audit'] is not None:result['progress']=s['progress_audit']
            results.append(result)
        return results

    def decide(self,request,stochastic=True):
        return self.decide_batch([(self,request)],stochastic)[0]

    def observe(self,ticket,observed,event=None,need_progress=None,maintained=(),completed=(),purpose_feedback=None):
        if self.pending is None or ticket!=self.pending['ticket']:raise ValueError('matching outstanding ticket required')
        memory=copy.deepcopy(self.memory);evidence=copy.deepcopy(self.evidence)
        update=memory.observe(ticket,observed)  # validates vector BEFORE evidence changes
        p=self.pending;b=p['binding'];key='experience:'+digest([b.method,b.situation])
        idx=[FEATURES.index(name) for name in p['exact']]
        if idx and not any(r['p']>0 and np.allclose(np.asarray(observed)[idx],vector(r)[idx],rtol=0,atol=1e-8) for r in p['prior']):
            raise ValueError('observed outcome violates certified exact effects')
        learning=evidence.compare(key,p['prior'],p['candidate'],observed,b.estimated)
        reading=None
        if p['response_trial'] is not None:
            if not isinstance(event,dict) or set(event)!=set(('revealed_action',)):raise ValueError('revealed response required for forecast validation')
            trial=p['response_trial']
            reading=evidence.categorical(trial['key'],trial['prior'],trial['candidate'],event['revealed_action'])
        if p['read'] is not None:
            r=p['read']
            if r['known']:
                if not isinstance(event,dict) or set(event)!=set(('revealed_action',)):raise ValueError('revealed response required for forecast validation')
                if p['response_trial'] is None or p['response_trial']['key']!=r['key']:
                    uniform={k:1/len(r['responses']) for k in r['responses']}
                    reading=evidence.categorical(r['key'],uniform,r['responses'],event['revealed_action'])
                if event['revealed_action'] not in r['responses']:raise ValueError('revealed conditional response required')
                row=r['candidate'][list(r['responses']).index(event['revealed_action'])]
                if not np.allclose(vector(row),np.asarray(observed),rtol=0,atol=1e-8):raise ValueError('revealed outcome contradicts declared conditional rules')
            else:
                reading=evidence.compare(r['key'],r['prior'],r['candidate'],observed,b.estimated)
                root_key=r['key']+':'+digest([b.method,b.situation])
                reading['root']=evidence.compare(root_key,r['prior'],r['candidate'],observed,b.estimated)
        predictor=self.predictor;model_update=None
        decision_comparison=None
        if p['decision_trial'] is not None:
            trial=p['decision_trial'];j=trial['responses'].index(event['revealed_action'])
            if trial['base_choice']!=trial['candidate_choice']:
                a=trial['baseline'][j]['objective'];b=trial['candidate'][j]['objective']
                decision_comparison=evidence._update(trial['key'],(1-a)/2,(1-b)/2)
                decision_comparison['kind']='known-rule purpose comparison; not observed unchosen samples'
        if predictor is not None:
            predictor,model_update=predictor.updated(p['raw'],observed,event,ticket)
        pressure,pressure_update=self._progress(need_progress,maintained,completed)
        progress,progress_update=self._purpose(purpose_feedback)
        self.memory=memory;self.evidence=evidence;self.predictor=predictor;self.pending=None;self.pressure=pressure
        self.progress=progress
        result=dict(memory=update,learning=learning,reading=reading,decision_comparison=decision_comparison,predictor=model_update)
        if self.pressure is not None:result['pressure']=pressure_update
        if self.progress is not None:result['progress']=progress_update
        return result

    def _purpose(self,feedback):
        progress=copy.deepcopy(self.progress);update=None
        if progress is not None:
            p=self.pending;update=progress.observe(p['tick'],p['action'],p['purpose'],feedback)
        elif feedback is not None:raise ValueError('purpose feedback requires configured progress watch')
        return progress,update

    def _progress(self,progress,maintained,completed):
        pressure=copy.deepcopy(self.pressure);update=None
        if progress is not None:
            if pressure is None:raise ValueError('need progress requires configured pressure')
            update=pressure.observe(self.pending['tick'],progress,maintained,completed)
        elif maintained or completed:raise ValueError('need labels require observed progress')
        return pressure,update

    def abandon(self,ticket,need_progress=None,maintained=(),completed=(),purpose_feedback=None):
        """No effect-vector learning; optionally resolve separate actual need progress.

        Game goal progress need not equal an immediate predicted effect vector.
        Never insert projected multi-tick effects into immediate OutcomeMemory.
        """
        if self.pending is None or ticket!=self.pending['ticket']:raise ValueError('matching outstanding ticket required')
        pressure,_=self._progress(need_progress,maintained,completed)
        progress,_=self._purpose(purpose_feedback)
        self.memory.pending=None;self.memory.last_tick=self.pending['tick'];self.pending=None;self.pressure=pressure
        self.progress=progress

    def record(self):
        if self.pending is not None:raise ValueError('resolve observation before checkpointing')
        result=dict(version='decision-loop-v1',scope=copy.deepcopy(self.scope),personality=copy.deepcopy(self.personality),seed=self.seed,
            values=copy.deepcopy(self.values),state=copy.deepcopy(self.state),last_tick=self.last_tick,
            policy_residual=self.policy.residual.tolist(),read_control=asdict(self.read_control),
            strategy=asdict(self.strategy),predictor=None if self.predictor is None else self.predictor.record(),
            memory=self.memory.record(),evidence=dict(capacity=self.evidence.capacity,window=self.evidence.window,
                min_trials=self.evidence.min_trials,margin=self.evidence.margin,
                entries=[dict(key=k,state=e['state'],gains=list(e['gains'])) for k,e in self.evidence.entries.items()]))
        if self.policy.principle_priority!='lexicographic':result['principle_priority']=self.policy.principle_priority
        if self.pressure is not None:result['pressure']=self.pressure.record()
        if self.progress is not None:result['progress']=self.progress.record()
        return result

    @classmethod
    def from_record(cls,context,record,policy=None,read_control=None,predictor=None):
        saved_policy=Policy(record['policy_residual'],principle_priority=record.get('principle_priority','lexicographic'));saved_control=ReadControl(**record['read_control'])
        if policy is not None and (not np.array_equal(policy.residual,saved_policy.residual) or policy.principle_priority!=saved_policy.principle_priority):raise ValueError('checkpoint policy changed')
        if read_control is not None and read_control!=saved_control:raise ValueError('checkpoint reading budget changed')
        loop=cls(context,policy or saved_policy,read_control or saved_control,record['memory']['capacity'],predictor)
        if record.get('pressure') is not None:loop.pressure=NeedPressure.from_record(loop.scope,record['pressure'])
        if record.get('progress') is not None:loop.progress=ProgressWatch.from_record(loop.scope,record['progress'])
        if loop.progress is not None and loop.progress.last_tick!=record['last_tick']:raise ValueError('progress checkpoint timeline mismatch')
        if loop.pressure is not None and loop.pressure.last_tick>record['last_tick']:raise ValueError('pressure checkpoint timeline mismatch')
        if record['version']!='decision-loop-v1' or any(record[k]!=getattr(loop,k) for k in ('scope','personality','values','seed')):
            raise ValueError('checkpoint actor contract mismatch')
        tick=record['last_tick']
        if type(tick) is not int or not -1<=tick<2**63:raise ValueError('invalid checkpoint tick')
        loop.memory=OutcomeMemory.from_record(loop.scope,record['memory'])
        if loop.memory.last_tick!=tick:raise ValueError('checkpoint timeline mismatch')
        state=record['state']
        if (tick==-1)!=(state is None):raise ValueError('checkpoint state/tick mismatch')
        if state is not None:
            check=copy.deepcopy(context);check['state']=state;compile_batch([check])
        loop.state=copy.deepcopy(state);loop.last_tick=tick
        route=record['strategy']
        if type(route['switches']) is not int or route['switches']<0:raise ValueError('invalid strategy checkpoint')
        if route['chosen'] is not None:identifier(route['chosen'])
        loop.strategy=RouteState(**route)
        e=record['evidence'];loop.evidence=EvidenceGate(e['capacity'],e['window'],e['min_trials'],e['margin'])
        if not isinstance(e['entries'],list) or len(e['entries'])>e['capacity']:raise ValueError('invalid evidence capacity')
        for row in e['entries']:
            identifier(row['key']);g=np.asarray(row['gains'])
            if row['key'] in loop.evidence.entries or row['state'] not in ('trial','active','revoked'):
                raise ValueError('invalid evidence key/state')
            if g.ndim!=1 or not 1<=len(g)<=e['window'] or not np.isfinite(g).all() or (abs(g)>1).any():raise ValueError('invalid evidence scores')
            if row['state']=='active' and len(g)<e['min_trials']:raise ValueError('unsupported active evidence')
            loop.evidence.entries[row['key']]=dict(state=row['state'],gains=deque(g.tolist(),maxlen=e['window']))
        if (predictor is None)!=(record['predictor'] is None):raise ValueError('checkpoint predictor contract mismatch')
        if predictor is not None:loop.predictor=loop.predictor.restored(record['predictor'])
        return loop
