"""Finite public transition trees with the same owned chooser at every node.

This is distributional backward induction with personality-dependent selection,
not a theorem of optimal MDP control. Exact kernels enumerate public transition
probabilities; sampled kernels approximate them. Neither supplies actual RNG.
Complete return distributions retain loss aversion instead of averaging away
bad outcomes. Work limits decline the WHOLE proposal, never selected roots.
"""
from dataclasses import dataclass,asdict
import copy
import math
from .core import digest,compile_batch
from .decision_loop import DecisionLoop,Request
from .intertemporal import Branch
from .purpose_plan import goal_forecast,Goal
from .model_observer import ModelFeedback
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


def reveal(actor,ticket,feedback):
    if not isinstance(feedback,ModelFeedback):raise ValueError('explicit modeled feedback required')
    kwargs=dict(need_progress=feedback.needs,maintained=feedback.maintained,completed=feedback.completed,
                purpose_feedback=feedback.purpose,social_events=feedback.social_events)
    if feedback.observed is not None:actor.observe(ticket,feedback.observed,event=feedback.event,**kwargs)
    else:
        if feedback.event is not None:raise ValueError('public event requires observed feedback')
        actor.abandon(ticket,**kwargs)


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
                 model_id,budget=None,max_regret=.15,future_choices='planned',frontier=None,preference_basis=None):
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
        if preference_basis is not None:
            from .preference_basis import PreferenceBasis
            if not isinstance(preference_basis,PreferenceBasis) or preference_basis.target!=target or self.budget.horizon!=1:
                raise ValueError('matching preference basis requires one-step flow plus full terminal endpoint')
        self.cache={};self.work=0;self.hits=0;self.last={};self._used=set()

    def clear(self):
        self.cache.clear()

    def _node(self,world,actor,request,remaining,*,root=False):
        if not isinstance(request,Request) or actor.pending is not None:raise ValueError('resolved owned Request required')
        # Independent readers/value proposals are outside this controller's
        # contract: an unseen post-planner override would invalidate its tree.
        if request.reader is not None or request.value_proposal is not None or actor.predictor is not None:
            raise ValueError('tree requires an explicit joint planner without external advice overrides')
        stage=actor._prepare(request);c=stage['effective'];compile_batch([c])
        identity=[self.model_id,self.target,self.unit,self.max_regret,self.record(world),actor.record(),asdict(request),c,remaining,
            actor.policy.principle_priority,actor.policy.residual.tolist(),actor.policy.commitments,self.future_choices,
            root if self.future_choices=='reflex' else False]
        if self.frontier is not None:identity.append(self.frontier.identity)
        if self.preference_basis is not None:identity.append(self.preference_basis.identity)
        stamp=digest(identity)
        self._used.update(self.cache[stamp]['dependencies'] if stamp in self.cache else (stamp,))
        self.work=len(self._used)
        if self.work>self.budget.nodes:raise WorkLimit('node capacity')
        if stamp in self.cache:self.hits+=1;return self.cache[stamp]
        paths={};audit={};children={};dependencies={stamp};frontiers=[]
        for key in sorted(a['id'] for a in c['actions'] if a['legal'] and not a['known_failure']):
            options=self.transitions(copy.deepcopy(world),key)
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
            for transition in options:
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
                    if not ended and self.frontier is not None:
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
                for p,flows,actions,goal,settled in tails:
                    probability=transition.probability*p;actual=(copy.deepcopy(transition.flow),)+flows
                    path=(key,)+actions
                    paths[key].append(Branch(probability,actual+(effect(),)*(remaining-len(actual)),(1.,)*remaining))
                    audit[key].append(dict(actions=list(path),terminal=settled,assessment=goal,terminal_coverage=terminal_coverage))
                    if len(paths[key])>self.budget.outcomes:raise WorkLimit('return distribution capacity')
                children[key].append(dict(probability=transition.probability,public=self.record(transition.state),
                    owner=digest(clone.record()),child=child_key,feedback=fb,flow=transition.flow,state=transition.state))
        forecast=goal_forecast(c,paths,audit,horizon=remaining,unit=self.unit,target=self.target,policy=actor.policy,
            max_regret=self.max_regret,settlement_weight='absolute',outcome_limit=self.budget.outcomes,preference_basis=self.preference_basis)
        if self.frontier is not None:
            forecast.audit['frontier']=dict(identity=self.frontier.identity,
                semantics='estimated purpose beyond cutoff under reference continuation; prefix personality flows only; running state retained')
        chosen=copy.deepcopy(actor)
        planner=None if self.future_choices=='reflex' and not root else lambda cs,ds:forecast
        result=DecisionLoop.decide_batch([(chosen,request)],False,planner)[0]
        key=result['decision']['action_id'];returns=[]
        for branch,row in zip(paths[key],audit[key]):
            count=len(row['actions']);returns.append((branch.probability,branch.effects[:count],tuple(row['actions']),row['assessment'],row['terminal']))
        # Audit the lifecycle which the future tree assumed. This catches lost
        # social/need/learning/progress state instead of silently using a model
        # of a different person after a planned choice.
        for item in children[key]:
            actual=copy.deepcopy(chosen);reveal(actual,result['ticket'],item['feedback'])
            if digest(actual.record())!=item['owner']:raise ValueError('planned and counterfactual owner continuation disagree')
        node=dict(stamp=stamp,forecast=forecast,returns=returns,action=key,decision=result['decision'],dependencies=frozenset(dependencies),
            purpose=result.get('deliberation',{}).get('selected_purpose'),children={k:[{n:v for n,v in x.items() if n not in ('feedback','flow','state')} for x in xs] for k,xs in children.items()})
        if self.frontier is not None:node['frontiers']=frontiers
        self.cache[stamp]=node
        return node

    def planner(self,actor,request,initial,*,remaining=None,captured=None):
        h=self.budget.horizon if remaining is None else remaining
        if type(h) is not int or not 1<=h<=self.budget.horizon:raise ValueError('bounded remaining decision window required')
        def call(contexts,decisions):
            if captured is not None:
                captured.pop('node',None);captured.pop('audit',None)
            if len(contexts)!=1 or self.terminal(initial):raise ValueError('one live owner required')
            before=digest(actor.record());self.work=0;self.hits=0;self._used=set()
            if len(self.cache)>self.budget.nodes:self.clear()
            try:
                node=self._node(copy.deepcopy(initial),actor,request,h,root=True)
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
        return call
