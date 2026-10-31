"""Finite public transition trees with the same owned chooser at every node.

This is distributional backward induction with personality-dependent selection,
not a theorem of optimal MDP control. Exact kernels enumerate public transition
probabilities; sampled kernels approximate them. Neither supplies actual RNG.
Complete return distributions retain loss aversion instead of averaging away
bad outcomes. Work limits decline the WHOLE proposal, never selected roots.
"""
from dataclasses import dataclass,asdict,replace
from . import cloning as copy
import math
from .core import digest,compile_batch
from .decision_loop import DecisionLoop,Request
from .intertemporal import Branch
from .purpose_plan import goal_forecast,Goal
from .model_observer import ModelFeedback,resolve_feedback as reveal
from .examples import effect


@dataclass(frozen=True)
class Transition:
    probability: float
    state: object
    flow: dict


@dataclass(frozen=True)
class TreeBudget:
    horizon: int=4
    nodes: int=4096
    outcomes: int=256

    def __post_init__(self):
        for value,limit in ((self.horizon,16),(self.nodes,100000),(self.outcomes,4096)):
            if type(value) is not int or not 1<=value<=limit:raise ValueError('bounded tree budget required')


class WorkLimit(Exception):
    """Incomplete trees cannot compare roots fairly."""


class DecisionWindow:
    """Remaining owner decisions in a fixed planning window, not wall-clock time.

    Offer is pure. Advance only after a real resolved transition; same-window
    observations can change the plan without extending its evaluation deadline.
    """
    def __init__(self,context,horizon=4):
        self.horizon=TreeBudget(horizon=horizon).horizon
        self.owner=digest([context['scope'],context['objective']]);self.used=0;self.last_tick=-1

    def remaining(self,context):
        if digest([context['scope'],context['objective']])!=self.owner or context['tick']<=self.last_tick:
            raise ValueError('same owner/goal and a new decision required')
        return self.horizon-self.used%self.horizon

    def advance(self,context):
        self.remaining(context);self.last_tick=context['tick'];self.used+=1

    def record(self):return dict(owner=self.owner,horizon=self.horizon,used=self.used,last_tick=self.last_tick)

    @classmethod
    def from_record(cls,context,record):
        if set(record)!={'owner','horizon','used','last_tick'}:raise ValueError('complete decision window required')
        out=cls(context,record['horizon'])
        if record['owner']!=out.owner or type(record['used']) is not int or record['used']<0 or type(record['last_tick']) is not int or record['last_tick']< -1:
            raise ValueError('valid owned decision window required')
        out.used=record['used'];out.last_tick=record['last_tick'];return out


class SampledKernel:
    """Equal public transition samples, independent of the actual world seed.

    advance must operate on a public state/belief, not a hidden determinization.
    All roots receive the same declared scenario bank at each public node.
    Persistent hidden uncertainty must live in the supplied public belief state.
    """
    def __init__(self,advance,seeds=(401,402)):
        if not callable(advance) or not isinstance(seeds,tuple) or not 1<=len(seeds)<=8 or any(type(s) is not int for s in seeds) or len(set(seeds))!=len(seeds):
            raise ValueError('distinct bounded public model seeds required')
        self.advance=advance;self.seeds=seeds

    def __call__(self,state,key):
        out=[]
        for seed in self.seeds:
            after,flow=self.advance(copy.deepcopy(state),key,seed)
            out.append(Transition(1/len(self.seeds),after,flow))
        return out


class ClosedLoop:
    def __init__(self,*,observe,transitions,terminal,assess,record,request_for,feedback,unit,target,
                 model_id,budget=None,max_regret=.15,future_choices='planned',frontier=None,preference_basis=None,owner_transitions=None,
                 opportunity_regret=False,relative_objective=False,tail_preferences=False,sampling_standard_errors=None,cumulative_preference=None):
        if any(not callable(x) for x in (observe,transitions,terminal,assess,record,request_for,feedback)):
            raise ValueError('explicit public model callbacks required')
        if not isinstance(model_id,str) or not model_id:raise ValueError('model identity required')
        if isinstance(max_regret,bool) or not math.isfinite(max_regret) or not 0<=max_regret<=2:raise ValueError('bounded purpose corridor required')
        if future_choices not in ('planned','reflex'):raise ValueError('explicit future choice rule required')
        if frontier is not None and (not isinstance(getattr(frontier,'identity',None),str) or not frontier.identity or not callable(getattr(frontier,'evaluate',None))):
            raise ValueError('identified endpoint evaluator required')
        self.observe=observe;self.transitions=transitions;self.terminal=terminal;self.assess=assess;self.record=record
        self.request_for=request_for;self.feedback=feedback;self.unit=unit;self.target=target
        self.model_id=model_id;self.budget=budget or TreeBudget();self.max_regret=max_regret
        self.future_choices=future_choices
        self.frontier=frontier
        self.preference_basis=preference_basis
        self.cumulative_preference=cumulative_preference
        if owner_transitions is not None and not callable(owner_transitions):
            raise ValueError('explicit owner-conditioned public kernel required')
        self.owner_transitions=owner_transitions
        if type(opportunity_regret) is not bool or type(relative_objective) is not bool:
            raise ValueError('explicit purpose scale semantics required')
        self.opportunity_regret=opportunity_regret;self.relative_objective=relative_objective
        if type(tail_preferences) is not bool or tail_preferences and (frontier is None or not callable(getattr(frontier,'trajectories',None))):
            raise ValueError('explicit complete tail trajectory provider required')
        self.tail_preferences=tail_preferences
        if cumulative_preference is not None:
            from .cumulative_preference import CumulativePreference
            if (not isinstance(cumulative_preference,CumulativePreference) or cumulative_preference.target!=target
                or not relative_objective or preference_basis is not None or (frontier is not None and not tail_preferences)):
                raise ValueError('cumulative values require a factual origin and complete declared preference paths')
        self.sampling_standard_errors=sampling_standard_errors
        if sampling_standard_errors is not None:
            from .frontier_value import PolicyTail
            from .sampling_band import PurposeSampling
            if (not tail_preferences or self.budget.horizon!=1 or not isinstance(frontier,PolicyTail)
                or frontier.purpose_adjustment is None):
                raise ValueError('sampling margin requires one exact root and a complete conditional owner tail')
            PurposeSampling(frontier.seeds,(('validate',tuple(0. for _ in frontier.seeds)),),frontier.identity,sampling_standard_errors)
        if getattr(frontier,'purpose_adjustment',None) is not None and (not tail_preferences or self.budget.horizon!=1):
            raise ValueError('conditional purpose mean currently requires one root plus full owner tail')
        if tail_preferences and preference_basis is not None:
            raise ValueError('full preference trajectories cannot replace whole channels with endpoint purpose')
        self.source_owner=None
        if preference_basis is not None:
            from .preference_basis import PreferenceBasis
            if not isinstance(preference_basis,PreferenceBasis) or preference_basis.target!=target or self.budget.horizon!=1:
                raise ValueError('matching preference basis requires one-step flow plus full terminal endpoint')
        self.cache={};self.work=0;self.hits=0;self.last={};self._used=set()

    def clear(self):
        self.cache.clear()

    def _node(self,world,actor,request,remaining,*,root=False,_forecast_only=False,_prepared_context=None):
        if not isinstance(request,Request) or actor.pending is not None:raise ValueError('resolved owned Request required')
        # Independent readers/value proposals are outside this controller's
        # contract: an unseen post-planner override would invalidate its tree.
        if request.reader is not None or request.value_proposal is not None or (actor.predictor is not None and self.owner_transitions is None):
            raise ValueError('tree requires an explicit joint planner without external advice overrides')
        baseline=None
        if self.owner_transitions is None:c=actor._prepare(request)['effective']
        elif _prepared_context is not None:
            if not (_forecast_only and root and remaining==1):raise ValueError('prepared context requires one-step forecast root')
            c=_prepared_context
        else:c,baseline=actor.preview(request)
        compile_batch([c])
        identity=[self.model_id,self.target,self.unit,self.max_regret,self.record(world),actor.record(),asdict(request),c,remaining,
            actor.policy.principle_priority,actor.policy.residual.tolist(),actor.policy.commitments,self.future_choices,
            root if self.future_choices=='reflex' else False,self.opportunity_regret,self.relative_objective,self.tail_preferences]
        if self.frontier is not None:identity.append(self.frontier.identity)
        if self.preference_basis is not None:identity.append(self.preference_basis.identity)
        if self.cumulative_preference is not None:identity.append(self.cumulative_preference.identity)
        if self.sampling_standard_errors is not None:identity.append(['paired-purpose-sampling-v1',self.sampling_standard_errors])
        if _forecast_only:identity.append('one-step-forecast-without-unused-successor-owner-v1')
        stamp=digest(identity)
        self._used.update(self.cache[stamp]['dependencies'] if stamp in self.cache else (stamp,))
        self.work=len(self._used)
        if self.work>self.budget.nodes:raise WorkLimit('node capacity')
        if stamp in self.cache:self.hits+=1;return self.cache[stamp]
        paths={};audit={};children={};dependencies={stamp};frontiers=[];purpose_samples={}
        fixed_leaf=self.future_choices=='reflex' and not root and remaining==1 and self.frontier is None
        return_horizon=remaining+(self.frontier.steps if self.tail_preferences else 0)
        if return_horizon>256:raise ValueError('bounded complete return horizon required')
        keys=sorted(a['id'] for a in c['actions'] if a['legal'] and not a['known_failure'])
        if self.future_choices=='reflex' and not root:
            # A fixed continuation needs its actual action, not a decision tree
            # for alternatives that this continuation will never select.
            if baseline is None:baseline=actor.preview(request)[1]
            keys=[baseline['action_id']]
        for key in keys:
            options=(self.transitions(copy.deepcopy(world),key) if self.owner_transitions is None else
                self.owner_transitions(copy.deepcopy(world),key,copy.deepcopy(actor),copy.deepcopy(c)))
            if not isinstance(options,(tuple,list)) or not options:raise ValueError('nonempty public transition distribution required')
            probs=[t.probability for t in options if isinstance(t,Transition)]
            if len(probs)!=len(options) or any(isinstance(p,bool) or not math.isfinite(p) or p<=0 for p in probs) or abs(sum(probs)-1)>1e-8:
                raise ValueError('normalized positive transition probabilities required')
            # Equal public successor AND realized flow are the same event for
            # this declared Markov model. Combine their mass, never average
            # unlike rewards or erase a possible bad outcome.
            merged={}
            for t in options:
                signature=digest([self.record(t.state),t.flow])
                if signature in merged:
                    old=merged[signature];merged[signature]=Transition(old.probability+t.probability,old.state,old.flow)
                else:merged[signature]=t
            options=list(merged.values())
            paths[key]=[];audit[key]=[];children[key]=[]
            if self.sampling_standard_errors is not None:purpose_samples[key]=[0. for _ in self.frontier.seeds]
            for transition in options:
                if fixed_leaf or _forecast_only:
                    # At this endpoint no later choice or actor-dependent
                    # evaluator consumes the updated learner. The forecast-only
                    # caller also performs its own ordinary selection/feedback.
                    # Preserve every physical outcome/flow; omit only that
                    # unused post-horizon owner update (and do not claim its hash).
                    ended=bool(self.terminal(transition.state));g=Goal(**self.assess(transition.state))
                    if g.target!=self.target or ended!=(g.status!='running'):raise ValueError('coherent fixed continuation endpoint required')
                    paths[key].append(Branch(transition.probability,(copy.deepcopy(transition.flow),),(1.,)))
                    audit[key].append(dict(actions=[key],terminal=ended,assessment=g.record(),terminal_coverage=ended))
                    children[key].append(dict(probability=transition.probability,public=self.record(transition.state),
                        owner=None,child=None,post_horizon_owner_omitted=True))
                    if len(paths[key])>self.budget.outcomes:raise WorkLimit('return distribution capacity')
                    continue
                mean_estimates=None
                clone=copy.deepcopy(actor)
                forced=DecisionLoop.decide_batch([(clone,request)],False,_counterfactual_roots=(key,))[0]
                fb=self.feedback(world,transition.state,key,c,transition.flow)
                reveal(clone,forced['ticket'],fb)
                ended=bool(self.terminal(transition.state));g=Goal(**self.assess(transition.state))
                terminal_coverage=ended
                if g.target!=self.target or ended!=(g.status!='running'):raise ValueError('consistent endpoint target and settlement required')
                if ended or remaining==1:
                    tails=[(1.,(),(),g.record(),ended)]
                    child_key=None
                    if not ended and self.tail_preferences:
                        if getattr(self.frontier,'purpose_adjustment',None) is not None:
                            distribution=self.frontier.distribution(copy.deepcopy(transition.state),copy.deepcopy(clone))
                            tails=distribution.trajectories;mean_estimates=distribution.purpose_estimates
                        else:tails=self.frontier.trajectories(copy.deepcopy(transition.state),copy.deepcopy(clone))
                        terminal_coverage=all(t[-1] for t in tails)
                        frontiers.append(dict(root=key,public=self.record(transition.state),supported=True,
                            source=self.frontier.identity,settled_fraction=sum(t[0] for t in tails if t[-1])))
                    elif not ended and self.frontier is not None:
                        from .frontier_value import EndpointEstimate
                        estimate=self.frontier.evaluate(copy.deepcopy(transition.state),copy.deepcopy(clone))
                        if estimate is not None:
                            if not isinstance(estimate,EndpointEstimate):raise ValueError('explicit endpoint estimate required')
                            # Keep the state RUNNING. Predicted settlement is not
                            # actual settlement, nor personal factual experience.
                            tails=[(p,(),(),Goal(self.target,'running',v).record(),False) for p,v in estimate.outcomes]
                            terminal_coverage=estimate.settled_fraction==1.
                        frontiers.append(dict(root=key,public=self.record(transition.state),supported=estimate is not None,
                            source=None if estimate is None else estimate.source,
                            settled_fraction=None if estimate is None else estimate.settled_fraction))
                else:
                    cc=self.observe(copy.deepcopy(transition.state),copy.deepcopy(clone.state))
                    child=self._node(transition.state,clone,self.request_for(cc),remaining-1)
                    dependencies.update(child['dependencies'])
                    tails=child['returns'];child_key=child['stamp']
                if self.sampling_standard_errors is not None:
                    estimates=(g.value,)*len(self.frontier.seeds) if ended else mean_estimates
                    if estimates is None or len(estimates)!=len(self.frontier.seeds):
                        raise ValueError('complete aligned conditional-root sample estimates required')
                    for i,value in enumerate(estimates):purpose_samples[key][i]+=transition.probability*value
                for index,(p,flows,actions,goal,settled) in enumerate(tails):
                    probability=transition.probability*p;actual=(copy.deepcopy(transition.flow),)+flows
                    path=(key,)+actions
                    paths[key].append(Branch(probability,actual+(effect(),)*(return_horizon-len(actual)),(1.,)*return_horizon))
                    audit[key].append(dict(actions=list(path),terminal=settled,assessment=goal,terminal_coverage=terminal_coverage))
                    if mean_estimates is not None:audit[key][-1]['purpose_mean_estimate']=mean_estimates[index]
                    if len(paths[key])>self.budget.outcomes:raise WorkLimit('return distribution capacity')
                children[key].append(dict(probability=transition.probability,public=self.record(transition.state),
                    owner=digest(clone.record()),child=child_key,feedback=fb,flow=transition.flow,state=transition.state))
        valuation=copy.deepcopy(c)
        valuation['actions']=[a for a in valuation['actions'] if a['id'] in paths]
        forecast=goal_forecast(valuation,paths,audit,horizon=return_horizon,unit=self.unit,target=self.target,policy=actor.policy,
            max_regret=self.max_regret,settlement_weight='absolute',outcome_limit=self.budget.outcomes,preference_basis=self.preference_basis,
            opportunity_regret=self.opportunity_regret,
            cumulative_preference=self.cumulative_preference,
            objective_origin=Goal(**self.assess(world)) if self.relative_objective else None)
        if self.sampling_standard_errors is not None:
            from .sampling_band import PurposeSampling
            aligned=tuple((plan,tuple(purpose_samples[roots[0]])) for plan,roots in forecast.roots.items())
            forecast=replace(forecast,purpose_sampling=PurposeSampling(self.frontier.seeds,aligned,
                self.frontier.identity,self.sampling_standard_errors))
        if self.frontier is not None:
            forecast.audit['frontier']=dict(identity=self.frontier.identity,
                semantics=('full sampled owner continuation and preference flows; running cutoff retained' if self.tail_preferences else
                    'estimated purpose beyond cutoff under reference continuation; prefix personality flows only; running state retained'))
        if _forecast_only:
            # The caller will select and observe through its normal lifecycle.
            # No consumer can use a post-cutoff learner or this tree's own choice:
            # there is no frontier, continuation, or returned successor owner.
            node=dict(stamp=stamp,forecast=forecast,dependencies=frozenset(dependencies),action=None)
            self.cache[stamp]=node
            return node
        chosen=None if fixed_leaf else copy.deepcopy(actor)
        planner=None if self.future_choices=='reflex' and not root else lambda cs,ds:forecast
        result=dict(decision=baseline) if fixed_leaf else DecisionLoop.decide_batch([(chosen,request)],False,planner)[0]
        key=result['decision']['action_id'];returns=[]
        for branch,row in zip(paths[key],audit[key]):
            count=len(row['actions']);returns.append((branch.probability,branch.effects[:count],tuple(row['actions']),row['assessment'],row['terminal']))
        # Audit the lifecycle which the future tree assumed. This catches lost
        # social/need/learning/progress state instead of silently using a model
        # of a different person after a planned choice.
        for item in (() if fixed_leaf else children[key]):
            actual=copy.deepcopy(chosen);reveal(actual,result['ticket'],item['feedback'])
            if digest(actual.record())!=item['owner']:raise ValueError('planned and counterfactual owner continuation disagree')
        node=dict(stamp=stamp,forecast=forecast,returns=returns,action=key,decision=result['decision'],dependencies=frozenset(dependencies),
            purpose=result.get('deliberation',{}).get('selected_purpose'),children={k:[{n:v for n,v in x.items() if n not in ('feedback','flow','state')} for x in xs] for k,xs in children.items()})
        if self.frontier is not None:node['frontiers']=frontiers
        self.cache[stamp]=node
        return node

    def planner(self,actor,request,initial,*,remaining=None,captured=None,_forecast_only=False):
        if self.source_owner is not None and digest(actor.record())!=self.source_owner:
            raise ValueError('rebuild owned rollout from the current resolved owner')
        h=self.budget.horizon if remaining is None else remaining
        if type(h) is not int or not 1<=h<=self.budget.horizon:raise ValueError('bounded remaining decision window required')
        if type(_forecast_only) is not bool or (_forecast_only and (h!=1 or self.frontier is not None or captured is not None)):
            raise ValueError('forecast-only evaluation requires one step without continuation or owner audit capture')
        def call(contexts,decisions,*,_prepared=None):
            if self.source_owner is not None and digest(actor.record())!=self.source_owner:
                raise ValueError('stale owned rollout after real feedback')
            if captured is not None:
                captured.pop('node',None);captured.pop('audit',None)
            if len(contexts)!=1 or self.terminal(initial):raise ValueError('one live owner required')
            before=digest(actor.record());self.work=0;self.hits=0;self._used=set()
            if len(self.cache)>self.budget.nodes:self.clear()
            try:
                prepared_context=None
                if _prepared is not None:
                    if not _forecast_only:raise ValueError('prepared context requires forecast-only planner')
                    source,resolved,effective=_prepared
                    # The ordinary lifecycle has already computed this same
                    # deterministic preview. No persistent cache or save field:
                    # evidence changes, foreign requests and foreign owners all
                    # fall back to the original independent preview.
                    if digest(source.record())==before and asdict(resolved)==asdict(request):
                        prepared_context=copy.deepcopy(effective)
                node=self._node(copy.deepcopy(initial),actor,request,h,root=True,_forecast_only=_forecast_only,
                    _prepared_context=prepared_context)
                # Forecast replaces actions and appends facts; select checks
                # those contracts, too, before the real owner commits a root.
                if any(contexts[0][k]!=node['forecast'].contexts[0][k] for k in contexts[0] if k not in ('actions','facts')):
                    raise ValueError('planner received a different effective owner')
                self.last=dict(complete=True,nodes=self.work,hits=self.hits,remaining=h,stamp=node['stamp'],action=node['action'])
                if captured is not None:captured.update(node=node,audit=self.last.copy())
                return node['forecast']
            except WorkLimit as error:
                self.last=dict(complete=False,reason=str(error),nodes=self.work,hits=self.hits,remaining=h)
                if captured is not None:captured['audit']=self.last.copy()
                return None
            finally:
                if digest(actor.record())!=before:raise AssertionError('tree mutated real owner')
        if _forecast_only:
            call._from_prepared=lambda cs,ds,source,resolved,effective:call(cs,ds,_prepared=(source,resolved,effective))
        return call


def owned_rollout(actor,*,observe,events,terminal,assess,record,request_for,feedback,
                  model_id,target,tail_steps=15,seeds=(701,702,703),max_regret=.15,
                  future_planner=None,future_planner_id=None,future_planner_limit=1,
                  sampling='independent',purpose_estimator='sampled',world_model='frozen',tree_steps=1,future_choices='planned',tail_recovery=0,tail_recovery_limit=16,tail_recovery_mode='stalled',tail_recovery_horizon=2,sampling_standard_errors=None,cumulative_preference=None,tail_goal_refinement=False):
    """Rebuild after real feedback: root world model + evolving owned rollout.

    `events(public_state, action)` enumerates known conditional PublicEvents.
    Probabilities come from a snapshot of the owner's learned public predictor.
    That snapshot generates the simulated WORLD; each simulated PERSON learns
    separately through normal feedback. A belief change cannot rewrite physics.
    The one-step tree plus complete tail transports all preference flows and
    one endpoint purpose, then reuses the existing persona selector. Tail choice
    defaults to the immediate owner policy. An identified future_planner can
    reproduce scheduled replanning, up to the declared finite call limit.
    `world_model='posterior'` instead integrates the explicit segmented world's
    conditional continuation from the factual root plus public future events.
    This changes the modeled joint distribution, not the person's preferences.
    `tail_recovery_mode='always'` gives every future owner a two-step decision,
    including when a positive immediate gain exists. It requires a budget for
    the entire tail: silently switching back to reflex would change the policy
    being valued. This is still a finite approximation of the real controller.
    A longer `tail_recovery_horizon` uses an exact first response followed by
    sampled learning-owner continuations. Its inner model seeds are separate
    from the outer evaluation bank. This is one finite policy-improvement
    level, not recursive perfect foresight or a promise to discover all plans.
    """
    import numpy as np
    from .frontier_value import PolicyTail
    from .adaptive_dynamics import PublicEvent
    from .model_observer import ModelFeedback
    if actor.pending is not None or not callable(getattr(actor.predictor,'frozen_forecaster',None)):
        raise ValueError('resolved owner with a public predictive distribution required')
    if type(tail_steps) is not int or not 0<=tail_steps<=255:
        raise ValueError('0..255 tail decisions plus one root required')
    if type(tail_goal_refinement) is not bool or tail_goal_refinement and tail_steps==0:
        raise ValueError('explicit goal refinement on a nonempty tail required')
    TreeBudget(horizon=tree_steps)
    if tree_steps+tail_steps>256:raise ValueError('bounded combined return horizon required')
    if type(tail_recovery) is not int or tail_recovery not in (0,2):raise ValueError('disabled or two-step continuation recovery required')
    if type(tail_recovery_limit) is not int or not 1<=tail_recovery_limit<=256:
        raise ValueError('1..256 future recovery calls required')
    if not tail_recovery and tail_recovery_limit!=16:raise ValueError('recovery budget requires a recovery controller')
    if tail_recovery and (future_planner is not None or tail_steps==0):raise ValueError('one identified tail controller required')
    if tail_recovery_mode not in ('stalled','always'):
        raise ValueError('explicit stalled or always future recovery mode required')
    if tail_recovery_mode=='always' and (tail_recovery!=2 or tail_recovery_limit<tail_steps):
        raise ValueError('always replanning requires two-step recovery throughout the tail')
    if type(tail_recovery_horizon) is not int or not 2<=tail_recovery_horizon<=128:
        raise ValueError('2..128 future policy decisions required')
    if tail_recovery_horizon!=2 and (tail_recovery!=2 or tail_recovery_mode!='always'):
        raise ValueError('longer future policy requires complete-tail replanning')
    if not callable(events):raise ValueError('known conditional event enumeration required')
    if sampling not in ('independent','stratified'):raise ValueError('known rollout sampling mode required')
    if sampling_standard_errors is not None and (sampling!='independent' or purpose_estimator!='conditional' or tail_steps==0 or tree_steps!=1):
        raise ValueError('paired sampling margin requires independent samples and a one-root conditional tail')
    if purpose_estimator not in ('sampled','conditional'):raise ValueError('known purpose mean estimator required')
    if world_model not in ('frozen','posterior','persistent'):raise ValueError('explicit world continuation model required')
    seeds=tuple(seeds)
    if tail_recovery_horizon>2 and set(seeds).intersection((1701,1702,1703)):
        raise ValueError('future policy search seeds must be separate from outer evaluation seeds')
    if sampling=='stratified':
        from .rollout_sampling import stratified_uniforms
        stratified_uniforms(seeds,0)
    source=digest(actor.record());predict=actor.predictor.frozen_forecaster()
    posterior=None
    if world_model!='frozen':
        posterior=actor.predictor.predictive_continuation(steps=tail_steps+tree_steps,persistent=world_model=='persistent')
    identity=digest(['owned-rollout-v1',model_id,source])
    if sampling!='independent':identity=digest([identity,sampling,seeds])
    if posterior is not None:identity=digest([identity,'retained-root-segment-posterior-v1'])
    if world_model=='persistent':identity=digest([identity,'persistent-current-regime-v1'])
    if cumulative_preference is not None:
        from .cumulative_preference import CumulativePreference
        if not isinstance(cumulative_preference,CumulativePreference) or cumulative_preference.target!=target:
            raise ValueError('matching explicit cumulative preference required')
        identity=digest([identity,cumulative_preference.identity])
    if tail_goal_refinement:identity=digest([identity,'public-purpose-coarse-tie-v1'])
    def transitions(w,key,own,c):
        rows=events(w,key);p=predict(c) if posterior is None else posterior(own.predictor,c)
        if (not isinstance(rows,(tuple,list)) or not rows or any(not isinstance(e,PublicEvent) for e in rows)
                or len({e.symbol for e in rows})!=len(rows) or {e.symbol for e in rows}!=set(p)):
            raise ValueError('one known conditional event per predicted public response required')
        if posterior is not None:
            # The conditional world reads future evidence from the detached
            # observer. Silently omitting or relabeling an event would instead
            # generate a different joint process while claiming the same model.
            # Callbacks are pure; validate even events a finite sample may miss.
            for e in rows:
                observed=feedback(copy.deepcopy(w),copy.deepcopy(e.public),key,copy.deepcopy(c),copy.deepcopy(e.flow))
                if not isinstance(observed,ModelFeedback) or observed.event!={'revealed_action':e.symbol}:
                    raise ValueError('posterior world requires matching public response feedback for every event')
        return [Transition(p[e.symbol],copy.deepcopy(e.public),copy.deepcopy(e.flow)) for e in rows if p[e.symbol]>0]
    def advance(w,key,seed,own,c):
        rows=transitions(w,key,own,c)
        u=(stratified_uniforms(seeds,c['tick'])[seeds.index(seed)] if sampling=='stratified' else
            np.random.default_rng(np.random.SeedSequence([seed,c['tick']])).random())
        total=0.
        for row in rows:
            total+=row.probability
            if u<total:return row.state,row.flow
        return rows[-1].state,rows[-1].flow
    def adjustment(w,key,own,c,after):
        rows=transitions(w,key,own,c)
        # Martingale difference under the declared MODEL, not a correction for
        # an incorrect world model. Keep actual samples for risk and learning.
        def value(state):
            goal=Goal(**assess(state))
            if goal.target!=target:raise ValueError('same-target conditional goal required')
            return goal.value
        return math.fsum(row.probability*value(row.state) for row in rows)-value(after)
    if tail_recovery:
        def recover(person,request,state):
            if tail_recovery_mode=='stalled':
                context=person.preview(request)[0];probabilities=person.predictor.predict(context)
                origin=Goal(**assess(state)).value
                expected=[]
                for action in context['actions']:
                    if not action['legal'] or action['known_failure']:continue
                    rows=events(copy.deepcopy(state),action['id'])
                    if len({e.symbol for e in rows})!=len(rows) or {e.symbol for e in rows}!=set(probabilities):
                        raise ValueError('complete conditional response coverage required')
                    expected.append(math.fsum(probabilities[e.symbol]*Goal(**assess(e.public)).value for e in rows))
                if max(expected)>origin+1e-12:return None
            # Preserve the registered two-step policy exactly. Longer policies
            # sample a complete tail instead of enumerating an exponential
            # response tree. No action is rewarded merely for being preparation.
            longer=tail_recovery_horizon>2
            inner=owned_rollout(person,observe=observe,events=events,terminal=terminal,assess=assess,
                record=record,request_for=request_for,feedback=feedback,model_id=model_id,target=target,
                tail_steps=tail_recovery_horizon-1 if longer else 0,tree_steps=1 if longer else 2,
                seeds=(1701,1702,1703),purpose_estimator='conditional' if longer else 'sampled',
                future_choices='reflex',world_model=world_model,max_regret=max_regret,
                cumulative_preference=cumulative_preference,
                tail_goal_refinement=tail_goal_refinement if longer else False)
            planner=inner.planner(person,request,state)
            if tail_recovery_mode=='stalled':return planner
            def complete(contexts,decisions):
                forecast=planner(contexts,decisions)
                if forecast is None:raise ValueError('incomplete future replanning; no silent reflex substitution')
                return forecast
            return complete
        future_planner=recover
        future_planner_id=('no-immediate-purpose-gain-two-step-owner-recovery-v1' if tail_recovery_mode=='stalled'
                           else 'every-decision-two-step-owner-replanning-v1')
        if tail_recovery_horizon!=2:
            future_planner_id=digest(['every-decision-long-owner-replanning-v1',tail_recovery_horizon,
                                      (1701,1702,1703),tail_goal_refinement])
        future_planner_limit=tail_recovery_limit
    if tail_steps==0 and (future_planner is not None or purpose_estimator!='sampled'):
        raise ValueError('zero tail has no future planning or sampling correction')
    refinement=None
    if tail_goal_refinement and (cumulative_preference is None or all(conversion>0 for _,conversion in cumulative_preference.channels)):
        from .goal_ties import reflex_goal_planner
        def refinement(person,request,state):
            return reflex_goal_planner(person,request,state,events=events,assess=assess,target=target)
    tail=None if tail_steps==0 else PolicyTail(observe=observe,advance=lambda *a:None,owner_advance=advance,
        terminal=terminal,assess=assess,request_for=request_for,feedback=feedback,
        model_id=identity,target=target,steps=tail_steps,seeds=seeds,
        future_planner=future_planner,future_planner_id=future_planner_id,future_planner_limit=future_planner_limit,
        purpose_adjustment=adjustment if purpose_estimator=='conditional' else None,
        purpose_adjustment_id='conditional-public-goal-martingale-v1' if purpose_estimator=='conditional' else None,
        reflex_planner=refinement,reflex_planner_id='public-purpose-coarse-tie-v1' if tail_goal_refinement else None)
    ctrl=ClosedLoop(observe=observe,transitions=lambda *a:None,owner_transitions=transitions,
        terminal=terminal,assess=assess,record=record,request_for=request_for,feedback=feedback,
        model_id=identity,unit='owner-decisions',target=target,
        budget=TreeBudget(horizon=tree_steps,outcomes=256),max_regret=max_regret,future_choices=future_choices,
        frontier=tail,tail_preferences=tail is not None,relative_objective=True,opportunity_regret=True,
        sampling_standard_errors=sampling_standard_errors,cumulative_preference=cumulative_preference)
    ctrl.source_owner=source
    return ctrl
