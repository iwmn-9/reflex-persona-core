"""Optional evidence boundary for the same persona policy; no world access.

Game adapters own method/context equivalence and the meaning of effects. These
are bounded engineering heuristics, not calibrated psychological estimates.
"""
from collections import OrderedDict, deque
from dataclasses import dataclass, replace
import copy
import numpy as np
from .core import FEATURES, Policy, compile_batch, digest, identifier, number, MAX_OUTCOMES
from .planning import vector, from_vector, compress_outcomes
from .runtime import Population


@dataclass(frozen=True)
class Binding:
    method: str
    situation: str
    estimated: tuple

    def __post_init__(self):
        identifier(self.method); identifier(self.situation)
        if not isinstance(self.estimated, tuple) or len(set(self.estimated)) != len(self.estimated) or set(self.estimated)-set(FEATURES):
            raise ValueError('distinct estimated feature names required')


def _compress_estimates(outcomes, estimated):
    """Bound estimated detail without inventing non-estimated rule outcomes.

    Preserve each joint non-estimated branch and its probability. Spare slots
    go to the highest probability mass per slot. Within a branch, compression
    still approximates estimated risk; a one-slot branch retains only its mean.
    """
    if len(outcomes)<=MAX_OUTCOMES:return copy.deepcopy(outcomes)
    fixed=np.flatnonzero(~estimated);groups=OrderedDict()
    for row in outcomes:
        if row['p']<=0:continue
        v=vector(row);group=groups.setdefault(tuple(v[fixed]),OrderedDict())
        key=tuple(v);group[key]=group.get(key,0.)+row['p']
    # Repeated identical observations need no extra slot or risk approximation.
    groups=OrderedDict((key,[from_vector(v,p) for v,p in group.items()]) for key,group in groups.items())
    if len(groups)>MAX_OUTCOMES:raise ValueError('non-estimated branches exceed outcome capacity')
    rows=list(groups.values());mass=[sum(r['p'] for r in group) for group in rows]
    slots=[1]*len(rows)
    while sum(slots)<MAX_OUTCOMES:
        remaining=[i for i,group in enumerate(rows) if slots[i]<len(group)]
        if not remaining:break
        i=max(remaining,key=lambda i:mass[i]/slots[i]);slots[i]+=1
    result=[]
    for (key,group),limit in zip(groups.items(),slots):
        for row in compress_outcomes(group,limit):
            v=vector(row)
            # Even a weighted mean of identical constants can drift by one ULP.
            v[fixed]=key;result.append(from_vector(v,row['p']))
    return result


class OutcomeMemory:
    """Learn selected, subsequently observed outcomes; keep personality untouched.

    Last four outcomes form a joint empirical distribution, blended with the
    current prior after two observations. Only adapter-labelled estimated
    dimensions change. No missing outcome or unchosen counterfactual is learned.
    One outstanding trial per owner; callers must resolve it before advancing.
    """
    def __init__(self, scope, capacity=64):
        if type(capacity) is not int or not 1 <= capacity <= 1024:
            raise ValueError('bounded capacity required')
        self.owner=digest(scope); self.capacity=capacity
        self.entries=OrderedDict(); self.pending=None; self.last_tick=-1

    def _check(self, context, bindings):
        if digest(context['scope']) != self.owner: raise ValueError('wrong game/episode/NPC owner')
        if set(bindings) != {a['id'] for a in context['actions']}: raise ValueError('bind every current action explicitly')
        if any(not isinstance(b, Binding) for b in bindings.values()): raise ValueError('Binding required')
        for b in bindings.values():
            old=self.entries.get((b.method,b.situation))
            if old is not None and old['estimated'] != b.estimated: raise ValueError('method semantics changed without a new key')

    def prepare(self, context, bindings):
        self._check(context,bindings)
        result=copy.deepcopy(context); stats={}
        for a in result['actions']:
            b=bindings[a['id']]; e=self.entries.get((b.method,b.situation))
            count=0 if e is None else e['count']
            weight=0. if count < 2 or not b.estimated else min(.75,count/(count+3))
            stats[a['id']]=dict(observations=count, empirical_weight=weight,
                               provenance='estimated' if b.estimated else 'not_empirically_updated',
                               confidence_kind='not a calibrated probability')
            if not weight: continue
            mask=np.array([name in b.estimated for name in FEATURES])
            prior=copy.deepcopy(a['outcomes'])
            # Exact rule effects may vary by branch. Preserve their marginal
            # distribution; empirical estimated effects are independent of those
            # branches here. This assumption must be declared by the adapter.
            rows=[]
            for row in prior:
                rows.append(from_vector(vector(row),row['p']*(1-weight)))
                for observed in e['samples']:
                    v=np.where(mask,observed,vector(row))
                    rows.append(from_vector(v,row['p']*weight/len(e['samples'])))
            a['outcomes']=_compress_estimates(rows,mask)
            result['facts']['experience_'+digest(a['id'])[:12]]=f"selected observed trials={count}; empirical mixture={weight:.3f}; no counterfactual observations"
        return result,stats

    def commit(self, context, decision, bindings):
        self._check(context,bindings)
        if self.pending is not None: raise ValueError('resolve the outstanding selected trial first')
        if context['tick'] <= self.last_tick: raise ValueError('observation ticks must increase')
        if decision['context_hash'] != digest(context): raise ValueError('decision/context mismatch')
        selected=next((a for a in context['actions'] if a['id']==decision['action_id']),None)
        if selected is None or not selected['legal'] or selected['known_failure']: raise ValueError('selected viable action required')
        b=bindings[selected['id']]
        self.pending=dict(tick=context['tick'],binding=b,action=selected['id'],
                          predicted=tuple(vector(row) for row in selected['outcomes'] if row['p']>0))
        self.ticket=digest([self.owner,context['tick'],decision['context_hash'],selected['id']])
        return self.ticket

    def observe(self, ticket, observed):
        if self.pending is None: raise ValueError('no outstanding selected trial')
        t=self.pending
        # Ticket is stored at commit by the wrapper below; no world identity or
        # private future is embedded in the observation interface.
        if ticket != self.ticket: raise ValueError('wrong observation ticket')
        v=np.asarray(observed,dtype=float)
        if v.shape != (len(FEATURES),) or not np.isfinite(v).all() or (abs(v)>1).any() or v[-1]<0:
            raise ValueError('bounded realized feature vector required')
        b=t['binding']; key=(b.method,b.situation)
        mask=np.array([name in b.estimated for name in FEATURES])
        e=self.entries.pop(key,None)
        reset=False
        if b.estimated:
            if e is None: e=dict(count=0,samples=deque(maxlen=4),estimated=b.estimated)
            # A large out-of-support result weakens obsolete evidence. A loss
            # already present in the predicted lottery is NOT such a signal.
            distance=min(float(np.max(abs(v[mask]-p[mask]))) for p in t['predicted'])
            reset=e['count']>=3 and distance>.6
            if reset: e['count']=0; e['samples'].clear()
            e['count']=min(12,e['count']+1); e['samples'].append(v.copy()); self.entries[key]=e
            while len(self.entries)>self.capacity: self.entries.popitem(last=False)
        self.last_tick=t['tick']; self.pending=None
        return dict(action=t['action'],regime_reset=reset,observations=0 if e is None else e['count'])

    def record(self):
        if self.pending is not None: raise ValueError('resolve the selected trial before checkpointing')
        return dict(version='outcome-memory-v1',owner=self.owner,capacity=self.capacity,last_tick=self.last_tick,
                    entries=[dict(method=m,situation=s,estimated=list(e['estimated']),count=e['count'],
                                  samples=[v.tolist() for v in e['samples']]) for (m,s),e in self.entries.items()])

    @classmethod
    def from_record(cls,scope,record):
        if not isinstance(record,dict) or set(record)!=set(('version','owner','capacity','last_tick','entries')):
            raise ValueError('complete memory record required')
        m=cls(scope,record['capacity'])
        if record['version']!='outcome-memory-v1' or record['owner']!=m.owner: raise ValueError('memory checkpoint owner/version mismatch')
        tick=record['last_tick']
        if type(tick) is not int or not -1 <= tick < 2**63: raise ValueError('invalid checkpoint tick')
        entries=record['entries']
        if not isinstance(entries,list) or len(entries)>m.capacity: raise ValueError('invalid checkpoint capacity')
        for e in entries:
            if not isinstance(e,dict) or set(e)!=set(('method','situation','estimated','count','samples')): raise ValueError('invalid method record')
            if not isinstance(e['estimated'],list): raise ValueError('feature list required')
            b=Binding(e['method'],e['situation'],tuple(e['estimated']));key=(b.method,b.situation)
            if not b.estimated or key in m.entries: raise ValueError('duplicate or empty learned method')
            samples=e['samples'];count=e['count']
            if not isinstance(samples,list) or not 1<=len(samples)<=4 or type(count) is not int or not len(samples)<=count<=12:
                raise ValueError('invalid sample count')
            values=[]
            for sample in samples:
                v=np.asarray(sample,dtype=float)
                if v.shape!=(len(FEATURES),) or not np.isfinite(v).all() or (abs(v)>1).any() or v[-1]<0: raise ValueError('invalid observed checkpoint effects')
                values.append(v.copy())
            m.entries[key]=dict(count=count,estimated=b.estimated,samples=deque(values,maxlen=4))
        if entries and tick<0: raise ValueError('learned memory requires an observed tick')
        m.last_tick=tick
        return m

def avoid_waste(batch, exact):
    """Remove only proven same-benefit deterministic actions with higher cost.

    All feature dimensions must be certified exact by the game for BOTH actions.
    Estimated equality, unknown correlation and lottery means prove nothing.
    Equal familiarity/switch penalties and reading effects are required. The
    guard deliberately leaves value/risk conflicts and stochastic gambles alone.
    """
    exact=np.asarray(exact)
    if exact.dtype != np.bool_ or exact.shape != batch.effects.shape[:2]+(len(FEATURES),):
        raise ValueError('boolean exact mask shaped NPC/action/feature required')
    legal=batch.legal & ~batch.known_failure; removed=np.zeros_like(legal)
    possible=legal & exact.all(2) & (batch.confidence==1) & ((batch.probability>0).sum(2)==1)
    idx=batch.probability.argmax(2)
    v=np.take_along_axis(batch.effects,idx[:,:,None,None],axis=2)[:,:,0,:]
    fees=batch.switch_cost*(np.arange(legal.shape[1])[None,:]!=batch.intent[:,None])
    # Vectorize NPCs; temporary storage is O(N*A*D), not O(N*A*A*D).
    for j in np.flatnonzero(possible.any(0)):
        same=possible & possible[:,j,None]
        same &= np.all(v[:,:,:-1]==v[:,j,None,:-1],axis=2)
        same &= batch.familiarity==batch.familiarity[:,j,None]
        same &= batch.reading==batch.reading[:,j,None]
        same &= fees==fees[:,j,None]
        removed[:,j]=np.any(same & (v[:,:,-1]<v[:,j,None,-1]),axis=1)
    guarded=replace(batch,legal=batch.legal & ~removed)
    guarded.legal.flags.writeable=False
    return guarded,removed


def choose(context, exact=None, stochastic=True, policy=None):
    batch=compile_batch([context]); removed=np.zeros_like(batch.legal)
    if exact is not None:
        if set(exact) != set(batch.ids[0]): raise ValueError('exact mask for every action required')
        if any(set(names)-set(FEATURES) for names in exact.values()): raise ValueError('unknown exact feature')
        mask=np.array([[[name in exact[key] for name in FEATURES] for key in batch.ids[0]]])
        batch,removed=avoid_waste(batch,mask)
    result=(policy or Policy()).decide(batch,stochastic)
    return result.records(batch)[0],dict(waste_removed=[batch.ids[0][j] for j in np.flatnonzero(removed[0])])


@dataclass(frozen=True)
class ReadControl:
    max_nodes: int = 256
    close_gap: float = .05
    downside: float = .3

    def __post_init__(self):
        if type(self.max_nodes) is not int or not 0 <= self.max_nodes <= 30000: raise ValueError('bounded nodes required')
        number(self.close_gap,0,1); number(self.downside,0,1)

    def request(self, batch, decision, model_available, maintains_advantage, threatened):
        """Numeric batch gate; caller enforces the returned search node cap.

        Advantage/threat are GAME-provided evidence assessments, not inferred
        from an actor label. A request is not a performed search or a confidence
        guarantee. No uncertainty penalty can erase personality's value tier.
        """
        n=len(batch.ids)
        flags=[]
        for v in (model_available,maintains_advantage,threatened):
            x=np.asarray(v)
            if x.dtype != np.bool_ or x.shape != (n,): raise ValueError('one boolean per NPC required')
            flags.append(x)
        available,ahead,threat=flags
        scores=np.where(decision.eligible,decision.scores,-np.inf)
        ordered=np.sort(scores,axis=1)
        gap=ordered[:,-1]-ordered[:,-2] if scores.shape[1]>1 else np.full(n,np.inf)
        rows=np.arange(n); chosen=decision.action
        worst=np.min(np.where(batch.probability>0,batch.effects[:,:,:,0],np.inf),axis=2)[rows,chosen]
        uncertainty=1-batch.confidence[rows,chosen]
        close=(decision.eligible.sum(1)>1) & (gap<self.close_gap) & (uncertainty>.1)
        endangered=worst < -self.downside
        request=available & (threat | close | endangered) & (~ahead | threat) & (self.max_nodes>0)
        return dict(nodes=np.where(request,self.max_nodes,0),requested=request,close_uncertain=close,downside=endangered)


class AdaptivePopulation(Population):
    """Numeric fast path for fixed method/context keys and deterministic priors.

    Same bounded empirical mixture as OutcomeMemory, stored in arrays across
    NPCs. Changing method identity/context/prior requires rebuilding at the game
    event boundary. Supports dynamics via Population's other numeric updates.
    No per-frame JSON/hash, no simulated outcome learning. Observe once after
    each step; unavailable outcomes explicitly learn nothing.
    """
    def __init__(self,contexts,estimated,policy=None):
        super().__init__(contexts,policy)
        if self.batch.effects.shape[2]!=1: raise ValueError('numeric memory requires single deterministic prior branch')
        mask=np.asarray(estimated)
        if mask.dtype!=np.bool_ or mask.shape!=self.batch.effects.shape[:2]+(len(FEATURES),):
            raise ValueError('estimated mask shaped NPC/action/feature required, in batch.ids order')
        self.estimated=mask.copy(); self.prior=self.batch.effects[:,:,0,:].copy()
        n,a,d=self.prior.shape
        self.count=np.zeros((n,a),int);self.cursor=np.zeros((n,a),int)
        self.samples=np.zeros((n,a,4,d));self.pending_action=None
        effects=np.zeros((n,a,5,d));probabilities=np.zeros((n,a,5))
        effects.flags.writeable=False;probabilities.flags.writeable=False
        self.batch=replace(self.batch,effects=effects,probability=probabilities)

    def step(self,stochastic=True,**updates):
        if self.pending_action is not None: raise ValueError('observe the outstanding selected actions first')
        if set(updates)&{'effects','probability'}: raise ValueError('rebuild when deterministic priors change')
        weight=np.where((self.count>=2)&self.estimated.any(2),np.minimum(.75,self.count/(self.count+3)),0)
        effects=np.empty_like(self.batch.effects);effects[:,:,0,:]=self.prior
        effects[:,:,1:,:]=np.where(self.estimated[:,:,None,:],self.samples,self.prior[:,:,None,:])
        probabilities=np.zeros_like(self.batch.probability);probabilities[:,:,0]=np.where(self.present,1-weight,0)
        sizes=np.minimum(self.count,4)
        probabilities[:,:,1:]=np.where(np.arange(4)[None,None,:]<sizes[:,:,None],weight[:,:,None]/np.maximum(sizes,1)[:,:,None],0)
        result=super().step(stochastic,effects=effects,probability=probabilities,**updates)
        self.pending_action=result.action.copy()
        return result

    def observe(self,realized,available=None):
        if self.pending_action is None: raise ValueError('no outstanding selected actions')
        v=np.asarray(realized,dtype=float);n,a,d=self.prior.shape
        if v.shape!=(n,d) or not np.isfinite(v).all() or (abs(v)>1).any() or (v[:,-1]<0).any(): raise ValueError('bounded realized effect per NPC required')
        available=np.ones(n,bool) if available is None else np.asarray(available)
        if available.dtype!=np.bool_ or available.shape!=(n,): raise ValueError('boolean availability per NPC required')
        rows=np.arange(n);chosen=self.pending_action;mask=self.estimated[rows,chosen]
        supported=self.batch.probability[rows,chosen]>0
        distance=np.max(np.where(mask[:,None,:],abs(self.batch.effects[rows,chosen]-v[:,None,:]),0),axis=2)
        distance=np.min(np.where(supported,distance,np.inf),axis=1)
        learn=available & mask.any(1)
        reset=learn & (self.count[rows,chosen]>=3) & (distance>.6)
        rr=rows[reset];cc=chosen[reset]
        self.count[rr,cc]=0;self.cursor[rr,cc]=0;self.samples[rr,cc]=0
        rr=rows[learn];cc=chosen[learn];slots=self.cursor[rr,cc]
        self.samples[rr,cc,slots]=v[rr]
        self.cursor[rr,cc]=(slots+1)%4;self.count[rr,cc]=np.minimum(12,self.count[rr,cc]+1)
        self.pending_action=None
        return dict(regime_reset=reset,learned=learn)
