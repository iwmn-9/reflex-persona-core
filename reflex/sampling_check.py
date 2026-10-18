"""Independent sample-bank agreement before committing a real owner decision.

Agreement is an empirical repeatability check, not a confidence guarantee.
Complete correlated paths are pooled BEFORE persona tiers or purpose filtering.
"""
from dataclasses import dataclass
from collections import OrderedDict
import copy
from reflex.core import digest
from reflex.closed_loop import owned_rollout
from reflex.decision_loop import DecisionLoop
from reflex.frontier_value import TailDistribution


@dataclass(frozen=True)
class DecisionSampling:
    banks: tuple=((701,702,703),(711,712,713),(721,722,723))
    cache_entries: int=4096

    def __post_init__(self):
        if not isinstance(self.banks,tuple) or len(self.banks)!=3 or any(not isinstance(b,tuple) or not b for b in self.banks):
            raise ValueError('three finite independent sample banks required')
        seeds=tuple(s for bank in self.banks for s in bank)
        if len(seeds)>32 or len(set(seeds))!=len(seeds) or any(type(s) is not int or s<0 for s in seeds):
            raise ValueError('at most 32 distinct nonnegative model seeds required')
        if type(self.cache_entries) is not int or not 1<=self.cache_entries<=4096:
            raise ValueError('bounded per-decision sample cache required')


class SampleBanks:
    """Private to one factual root; controllers differ ONLY in model seeds."""
    def __init__(self,owner,options,spec):
        if not isinstance(spec,DecisionSampling) or options['tail_steps']<1:
            raise ValueError('explicit sampling check with a nonempty tail required')
        if tuple(options['seeds'])!=spec.banks[0] or options.get('sampling','independent')!='independent':
            raise ValueError('first bank must match independent base model seeds')
        if options.get('future_planner') is not None:
            raise ValueError('bank reuse requires the built-in seed-independent future policies')
        self.owner=owner;self.before=digest(owner.record());self.options=copy.copy(options);self.spec=spec
        self.cache=OrderedDict();self.hits=0;self.misses=0
        self.tails=[owned_rollout(owner,**dict(options,seeds=bank)).frontier for bank in spec.banks]

    def _distribution(self,world,person,indices):
        if digest(self.owner.record())!=self.before:raise ValueError('sample banks belong to an unchanged factual owner')
        key=digest([self.options['record'](copy.deepcopy(world)),person.record()])
        trajectories=[];estimates=[];count=sum(len(self.spec.banks[i]) for i in indices)
        for i in indices:
            stamp=(i,key)
            if stamp in self.cache:
                self.hits+=1;sample=self.cache[stamp];self.cache.move_to_end(stamp)
            else:
                self.misses+=1;sample=self.tails[i].distribution(world,person)
                if len(sample.trajectories)!=len(self.spec.banks[i]) or len(sample.purpose_estimates)!=len(sample.trajectories):
                    raise ValueError('one complete correlated trajectory per independent seed required')
                if any(row[0]!=1/len(self.spec.banks[i]) for row in sample.trajectories):
                    raise ValueError('uniform independent model samples required')
                self.cache[stamp]=sample
                if len(self.cache)>self.spec.cache_entries:self.cache.popitem(last=False)
            trajectories.extend((1/count,*row[1:]) for row in sample.trajectories)
            estimates.extend(sample.purpose_estimates)
        return copy.deepcopy(TailDistribution(tuple(trajectories),tuple(estimates)))

    def evaluate(self,public,indices):
        if digest(self.owner.record())!=self.before:raise ValueError('cannot reuse samples after real feedback')
        reference=copy.deepcopy(self.owner);options=self.options
        seeds=tuple(s for i in indices for s in self.spec.banks[i])
        ctrl=owned_rollout(reference,**dict(options,seeds=seeds))
        ctrl.frontier.distribution=lambda world,person:self._distribution(world,person,indices)
        request=options['request_for'](options['observe'](copy.deepcopy(public),copy.deepcopy(reference.state)))
        captured={};planner=ctrl.planner(reference,request,copy.deepcopy(public),captured=captured);inputs=[]
        def complete(contexts,decisions):
            inputs.append(digest([contexts,decisions]))
            forecast=planner(contexts,decisions)
            if forecast is None:raise ValueError('incomplete sampled forecast; no silent fallback')
            return forecast
        result=DecisionLoop.decide_batch([(reference,request)],False,complete)[0]
        if len(inputs)!=1:raise ValueError('one complete factual-root preview required')
        return dict(result=result,captured=captured,inputs=inputs[0],seeds=seeds)


def checked_decide(controller,public,spec):
    """Return a normal pending decision; all trials remain detached."""
    banks=SampleBanks(controller.actor,controller.options,spec)
    first=banks.evaluate(public,(0,));second=banks.evaluate(public,(1,));pooled=banks.evaluate(public,(0,1))
    def choice(row):return row['result']['decision']['action_id']
    agreement=choice(first)==choice(second)==choice(pooled)
    stages=[dict(seeds=list(row['seeds']),action=choice(row)) for row in (first,second,pooled)]
    if not agreement:
        pooled=banks.evaluate(public,(0,1,2));stages.append(dict(seeds=list(pooled['seeds']),action=choice(pooled)))
    if digest(controller.actor.record())!=banks.before:raise ValueError('sampling preview mutated factual owner')
    request=controller.request_for(controller.observe(copy.deepcopy(public),copy.deepcopy(controller.actor.state)))
    def adopt(contexts,decisions):
        if digest([contexts,decisions])!=pooled['inputs']:
            raise ValueError('actual owner differs from sampled preview')
        return pooled['captured']['node']['forecast']
    result=DecisionLoop.decide_batch([(controller.actor,request)],False,adopt)[0]
    if result['decision']!=pooled['result']['decision']:
        raise AssertionError('sampled and committed owner decisions disagree')
    audit=dict(kind='independent-decision-banks-v1',stages=stages,agreement=agreement,
        expanded=not agreement,selected_samples=len(pooled['seeds']),cache_hits=banks.hits,
        computed_bank_frontiers=banks.misses,
        semantics='Complete paths pooled before persona and purpose gates; agreement is not a statistical confidence guarantee. World-model and horizon bias remain.')
    controller.last=pooled['captured'];controller.last['sampling_check']=audit
    result['sampling_check']=copy.deepcopy(audit)
    return result
