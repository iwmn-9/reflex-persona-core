"""Bounded short-plan search inspired by rolling horizon planning.

Plans are legal-list quantiles at successive OWN decisions, not semantic
attack/defend policies. Past terminal estimates are never reused as fresh facts.
Discovery and switch validation use independent streams and one total budget.
"""
from dataclasses import dataclass,asdict
import math
import numpy as np
from .core import Policy,compile_batch,digest
from .examples import action,effect
from .planning import compress_outcomes
from .rule_baseline import MODES,make_context,reward_effect


@dataclass(frozen=True)
class Plan:
    genes: tuple
    def __post_init__(self):
        if not 1<=len(self.genes)<=8 or any(not np.isfinite(q) or not 0<=q<1 for q in self.genes):
            raise ValueError('one to eight legal-choice quantiles required')
    @property
    def id(self):return 'PLAN:'+digest(self.genes)[:16]
    def move(self,names,index,u):
        q=self.genes[index] if index<len(self.genes) else u
        return names[min(int(q*len(names)),len(names)-1)]


@dataclass(frozen=True)
class PlanMemory:
    owner: str
    remaining: tuple
    generation: int=0
    def __post_init__(self):
        if self.remaining:Plan(self.remaining)
    def record(self):return asdict(self)
    @classmethod
    def from_record(cls,row):return cls(row['owner'],tuple(row['remaining']),row['generation'])


@dataclass(frozen=True)
class PlanBudget:
    horizon: int=3
    variants: int=6
    search_samples: int=8
    validation_samples: int=24
    min_samples: int=4
    max_nodes: int=18000
    max_steps: int=512
    bootstrap_samples: int=48
    margin: float=.025
    lower_quantile: float=.1
    persist: bool=True
    def __post_init__(self):
        for k,lo,hi in (('horizon',1,8),('variants',0,16),('search_samples',1,128),('validation_samples',1,128),
                        ('min_samples',1,128),('max_nodes',0,1000000),('max_steps',1,4096),('bootstrap_samples',8,256)):
            x=getattr(self,k)
            if type(x) is not int or not lo<=x<=hi:raise ValueError('invalid '+k)
        if self.min_samples>min(self.search_samples,self.validation_samples):raise ValueError('sample floor too high')
        if not 0<=self.margin<=1 or not 0<self.lower_quantile<.5 or type(self.persist) is not bool:raise ValueError('invalid switch controls')


class Counter:
    def __init__(self,maximum):self.maximum=maximum;self.nodes=0
    def debit(self,limit):
        if self.nodes>=min(limit,self.maximum):raise Cut()
        self.nodes+=1


class Cut(Exception):pass


def terminal_trial(rules,state,viewer,plan,seed,counter,limit,max_steps):
    """Only public model state; root included in transition accounting."""
    rng=np.random.default_rng(seed);chance_rng=np.random.default_rng(seed^0xD1B54A32D192ED03)
    own=0
    for _ in range(max_steps):
        if rules.terminal(state):return rules.rewards(state)
        counter.debit(limit)
        if rules.simultaneous:
            name=plan.move(rules.legal(state,viewer),own,float(rng.random()));own+=1
            state=rules.sample(rules.root(state,name,viewer),chance_rng,viewer)
        elif rules.chance(state):state=rules.sample(state,chance_rng,viewer)
        else:
            names=rules.legal(state);u=float(rng.random())
            if rules.actor(state)==viewer:name=plan.move(names,own,u);own+=1
            else:name=names[min(int(u*len(names)),len(names)-1)]
            state=rules.step(state,name)
    if rules.terminal(state):return rules.rewards(state)
    raise Cut() # no invented reward for unfinished trials


def sample_plans(rules,state,viewer,plans,count,scope,counter,limit,max_steps):
    results={p.id:[] for p in plans};discarded=0;complete=0
    for i in range(count):
        packet={};seed=int(digest([scope,i])[:16],16)
        try:
            for p in plans:packet[p.id]=terminal_trial(rules,state,viewer,p,seed,counter,limit,max_steps)
        except Cut:
            discarded+=len(packet);break
        for key,r in packet.items():results[key].append(r)
        complete+=1
    return results,dict(completed_samples=complete,discarded_terminal_samples=discarded)


def score_plans(rules,state,viewer,profile,mode,plans,results,seed,tick,episode,policy_state=None,indices=None):
    acts=[]
    for p in plans:
        rows=results[p.id]
        if indices is not None:rows=[rows[i] for i in indices]
        if not rows:packed=[effect()]
        else:
            groups={}
            for r in rows:
                e=reward_effect(r,viewer,mode);key=digest(e)
                if key not in groups:groups[key]=[e,0]
                groups[key][1]+=1
            packed=compress_outcomes([dict(e,p=n/len(rows)) for e,n in groups.values()])
        acts.append(action(p.id,*packed))
    c=make_context(rules,state,viewer,profile,mode,seed,tick,episode,acts,policy_state)
    c['facts']['evaluation']='未来の自分は有限の行動計画、その後は一様。相手は独立一様。終局だけで評価'
    c['facts']['plan_encoding']='各自分手番での合法手一覧の選択順位。攻め/守りの意味ラベルではない'
    b=compile_batch([c]);s=Policy().decide(b,stochastic=False)
    return c,b,s


def candidate_plans(names,budget,rng,incumbent):
    # Keep every current legal root and its flat continuation as an anchor.
    plans=[Plan(((i+.5)/len(names),)) for i in range(len(names))]
    if budget.horizon>1:
        plans += [Plan(((i+.5)/len(names),)+tuple(float(x) for x in rng.random(budget.horizon-1))) for i in range(len(names))]
        for i in range(budget.variants):
            if incumbent is not None and i%2==0:
                g=list(incumbent.genes)+list(float(x) for x in rng.random(max(0,budget.horizon-len(incumbent.genes))))
                j=int(rng.integers(len(g)));g[j]=float(rng.random());plans.append(Plan(tuple(g)))
            else:plans.append(Plan(tuple(float(x) for x in rng.random(budget.horizon))))
    if incumbent is not None:plans.append(incumbent)
    return list({p.id:p for p in plans}.values())


def validation_gate(gain,lower_gap,new_eligible,old_eligible,priority_support,budget):
    if not new_eligible:return False,'challenger_outside_value_tier'
    if not old_eligible and priority_support>=1-budget.lower_quantile:return True,'confirmed_value_tier_change'
    if gain>budget.margin and lower_gap>budget.margin:return True,'confirmed_persona_gain'
    return False,'gain_not_confirmed'


def decide(rules,state,viewer,profile,mode,budget,seed,tick,episode,policy_state=None,plan_memory=None):
    if mode not in MODES:raise ValueError('invalid persona mode')
    if rules.terminal(state) or not 0<=viewer<rules.players or (not rules.simultaneous and (rules.chance(state) or rules.actor(state)!=viewer)):
        raise ValueError('live own decision required')
    owner=digest([rules.game,viewer,episode,profile,mode])
    if plan_memory is not None and plan_memory.owner!=owner:raise ValueError('plan belongs to another actor/personality/episode')
    incumbent=Plan(plan_memory.remaining) if budget.persist and plan_memory is not None and plan_memory.remaining else None
    rng=np.random.default_rng(int(digest([seed,rules.game,viewer,episode,tick,'proposals'])[:16],16))
    names=rules.legal(state,viewer);plans=candidate_plans(names,budget,rng,incumbent);counter=Counter(budget.max_nodes)
    results,search=sample_plans(rules,state,viewer,plans,budget.search_samples,[seed,rules.game,viewer,episode,tick,'discovery'],
        counter,math.floor(budget.max_nodes*.65),budget.max_steps)
    c,b,scored=score_plans(rules,state,viewer,profile,mode,plans,results,seed,tick,episode,policy_state)
    lookup={p.id:p for p in plans};positions={name:i for i,name in enumerate(b.ids[0])}
    proposed=lookup[b.ids[0][int(scored.action[0])]]
    # Best one-action anchor under the SAME discovery data; no game heuristic.
    flat=[p for p in plans if len(p.genes)==1]
    available=[p for p in flat if scored.eligible[0,positions[p.id]]] or flat
    anchor=max(available,key=lambda p:scored.scores[0,positions[p.id]])
    current=incumbent or anchor;chosen=current;reason='search_below_sample_floor'
    validation=dict(used=False,completed_samples=0);point_gain=lower_gap=None;priority_support=None
    if search['completed_samples']>=budget.min_samples:
        reason='same_plan'
        if proposed.id!=current.id:
            pair=[current,proposed]
            heldout,v=sample_plans(rules,state,viewer,pair,budget.validation_samples,[seed,rules.game,viewer,episode,tick,'validation'],
                counter,budget.max_nodes,budget.max_steps)
            validation.update(v)
            if v['completed_samples']>=budget.min_samples:
                vc,vb,vs=score_plans(rules,state,viewer,profile,mode,pair,heldout,seed,tick,episode,policy_state)
                old=vb.ids[0].index(current.id);new=vb.ids[0].index(proposed.id)
                point_gain=float(vs.scores[0,new]-vs.scores[0,old]);gaps=[];priorities=[]
                brng=np.random.default_rng(int(digest([seed,rules.game,viewer,episode,tick,'paired-bootstrap'])[:16],16))
                for _ in range(budget.bootstrap_samples):
                    ix=brng.integers(v['completed_samples'],size=v['completed_samples'])
                    _,_,bs=score_plans(rules,state,viewer,profile,mode,pair,heldout,seed,tick,episode,policy_state,ix)
                    gaps.append(float(bs.scores[0,new]-bs.scores[0,old]));priorities.append(bool(bs.eligible[0,new] and not bs.eligible[0,old]))
                lower_gap=float(np.quantile(gaps,budget.lower_quantile));priority_support=float(np.mean(priorities))
                accepted,reason=validation_gate(point_gain,lower_gap,bool(vs.eligible[0,new]),bool(vs.eligible[0,old]),priority_support,budget)
                if accepted:chosen=proposed
                validation.update(used=True,persona_gain=point_gain,bootstrap_lower_gap=lower_gap,priority_support=priority_support,
                    estimates={p.id:dict(share=float(np.mean([r.credits[viewer] for r in heldout[p.id]])),persona_score=float(vs.scores[0,vb.ids[0].index(p.id)])) for p in pair},
                    limitation='paired bootstrap quantile is a sampling heuristic, not calibrated real-opponent confidence')
            else:reason='validation_below_sample_floor'
    changed=incumbent is not None and chosen.id!=incumbent.id
    final_context,_,_=score_plans(rules,state,viewer,profile,mode,[chosen],
        {chosen.id:results[chosen.id] if search['completed_samples']>=budget.min_samples else []},seed,tick,episode,policy_state)
    name=chosen.move(names,0,0.)
    final_context['actions'][0]['id']=name
    final_context['facts']['commitment']='全合法根と短い計画を比較後、監督器が選んだ今回の実行手。全候補の採点は別ログ'
    fb=compile_batch([final_context]);fs=Policy().decide(fb,stochastic=False);d=fs.records(fb)[0]
    generation=(plan_memory.generation if plan_memory else 0)+int(changed or incumbent is None)
    next_plan=PlanMemory(owner,chosen.genes[1:],generation)
    stats=dict(config=asdict(budget),total_nodes=counter.nodes,search=search,validation=validation,
        reason=reason,plan_changed=changed,plan_started=incumbent is None,
        current_plan=current.id,proposed_plan=proposed.id,selected_plan=chosen.id,
        selected_genes=list(chosen.genes),remaining_genes=list(next_plan.remaining),
        immediate_action_changed=chosen.move(names,0,0.)!=current.move(names,0,0.),
        discovery=[dict(id=p.id,genes=list(p.genes),persona_score=float(scored.scores[0,positions[p.id]]),eligible=bool(scored.eligible[0,positions[p.id]]),
            estimated_share=float(np.mean([r.credits[viewer] for r in results[p.id]])) if results[p.id] else None) for p in plans],
        opponent_assumption='independent uniform, no hidden real controllers/personas',
        plan_semantics='short conditional legal-choice sequence; no attack/defend labels or learned value function')
    assert counter.nodes<=budget.max_nodes
    return final_context,d,stats,next_plan
