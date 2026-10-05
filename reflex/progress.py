"""Finite waiting and observed progress; never equate animation with purpose.

Games declare the purpose/readiness scale and why each current means matters.
This layer expires unsupported waiting, remembers repeated deterministic
failures, and preserves actual preparation, maintenance and uncertain attempts.
It is a competence constraint, not a change to fixed personality/value axes.
"""
from collections import OrderedDict
from dataclasses import dataclass,asdict
import copy
import math
from .core import digest,identifier


def level(x):
    if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or not -1<=x<=1:
        raise ValueError('bounded observed purpose/readiness required')


@dataclass(frozen=True)
class Activity:
    kind: str
    reason: str
    condition: str='same'
    patience: int=3
    release: str=''
    replacement: bool=True

    def __post_init__(self):
        if self.kind not in ('attempt','uncertain','wait','idle','maintain'):raise ValueError('known purpose activity required')
        identifier(self.reason);identifier(self.condition)
        if type(self.patience) is not int or not 1<=self.patience<=16:raise ValueError('finite patience required')
        if self.kind=='wait' and not self.release:raise ValueError('waiting requires an explicit release condition')
        if self.release:identifier(self.release)
        if type(self.replacement) is not bool:raise ValueError('explicit sensible-alternative declaration required')


@dataclass(frozen=True)
class PurposeRequest:
    level: float
    readiness: float
    activities: dict
    maintained: bool=False

    def __post_init__(self):
        level(self.level);level(self.readiness)
        if not isinstance(self.activities,dict) or not self.activities or any(not isinstance(a,Activity) for a in self.activities.values()):
            raise ValueError('activity evidence for every root required')
        if type(self.maintained) is not bool:raise ValueError('actual newly observed preparation/maintenance required')


@dataclass(frozen=True)
class PurposeFeedback:
    level: float
    readiness: float
    maintained: bool=False
    completed: bool=False

    def __post_init__(self):
        level(self.level);level(self.readiness)
        if type(self.maintained) is not bool or type(self.completed) is not bool:raise ValueError('actual purpose maintenance/completion required')


@dataclass(frozen=True)
class ProgressConfig:
    grace: int=3
    repeat_limit: int=3
    floor: float=.005
    capacity: int=64
    proof_margin: float | None=None

    def __post_init__(self):
        for k,lo,hi in (('grace',1,16),('repeat_limit',1,16),('capacity',1,1024)):
            v=getattr(self,k)
            if type(v) is not int or not lo<=v<=hi:raise ValueError('bounded '+k)
        if isinstance(self.floor,bool) or not isinstance(self.floor,(int,float)) or not math.isfinite(self.floor) or not 0<self.floor<=1:
            raise ValueError('positive progress threshold required')
        if self.proof_margin is not None and (isinstance(self.proof_margin,bool) or not isinstance(self.proof_margin,(int,float)) or not math.isfinite(self.proof_margin) or not 0<=self.proof_margin<=2):
            raise ValueError('bounded optional recovery margin required')


class ProgressWatch:
    def __init__(self,scope,config=None):
        self.owner=digest(scope);self.config=config or ProgressConfig()
        self.peak=None;self.ready_peak=None;self.stalls=0;self.wait_spent=0;self.last_tick=-1
        self.failures=OrderedDict()

    def improving(self,p):
        return p.maintained or self.peak is not None and (p.level>self.peak+self.config.floor or p.readiness>self.ready_peak+self.config.floor)

    def mask(self,context,p,recovery=None):
        if digest(context['scope'])!=self.owner or not isinstance(p,PurposeRequest):raise ValueError('owned purpose request required')
        roots={a['id'] for a in context['actions']}
        if set(p.activities)!=roots:raise ValueError('activity labels must cover the current roots exactly')
        improved=self.improving(p)
        stalls=0 if improved else self.stalls;spent=0 if improved else self.wait_spent
        viable={a['id'] for a in context['actions'] if a['legal'] and not a['known_failure']}
        blocked={}
        for root in viable:
            a=p.activities[root];key=digest([root,a.condition])
            if a.kind=='wait' and spent>=a.patience:blocked[root]='wait exhausted without new observed progress'
            elif a.kind in ('idle','maintain') and stalls>=self.config.grace:blocked[root]='no demonstrated progress or useful maintenance'
            elif a.kind=='attempt' and not improved and self.failures.get(key,0)>=self.config.repeat_limit:
                blocked[root]='repeated deterministic attempt without progress'
        allowed=viable-set(blocked)
        unresolved=not allowed or not any(p.activities[k].replacement for k in allowed)
        # No forced meaningless movement or unsafe move when every supplied
        # means lacks support. The adapter must supply an actual recovery option.
        if unresolved:allowed=viable
        audit=dict(blocked=blocked,applied=bool(blocked) and not unresolved,
            unresolved=unresolved,stalls=stalls,wait_spent=spent,
            reason='no supported alternative; retain viable roots' if unresolved else 'observed purpose evidence',
            activity={k:asdict(a) for k,a in p.activities.items()})
        if recovery is not None:
            if self.config.proof_margin is None or not isinstance(recovery,dict) or set(recovery) not in ({'needed','base_purpose','candidate_purpose'},{'needed','base_purpose','candidate_purpose','margin'}) or type(recovery['needed']) is not bool:
                raise ValueError('configured explicit recovery comparison required')
            for k in ('base_purpose','candidate_purpose'):
                if recovery[k] is not None:level(recovery[k])
            a,b=recovery['base_purpose'],recovery['candidate_purpose']
            margin=recovery.get('margin',self.config.proof_margin)
            if isinstance(margin,bool) or not isinstance(margin,(int,float)) or not math.isfinite(margin) or not self.config.proof_margin<=margin<=2:
                raise ValueError('shared recovery margin must honor each actor requirement')
            approved=not recovery['needed'] or a is not None and b is not None and b>a+margin+1e-12
            audit['recovery']=dict(recovery,margin=margin,approved=bool(approved))
            if audit['applied'] and not approved:
                allowed=viable;audit.update(applied=False,unresolved=True,reason='reconsidered; no demonstrated purpose gain over current plan')
        return allowed,audit

    def observe(self,tick,root,request,feedback=None):
        if type(tick) is not int or not self.last_tick<tick<2**63:raise ValueError('increasing purpose observation tick required')
        if not isinstance(request,PurposeRequest) or root not in request.activities:raise ValueError('selected purpose root required')
        if feedback is not None and not isinstance(feedback,PurposeFeedback):raise ValueError('observed PurposeFeedback required')
        # Compute on copies, so validation/feedback rejection cannot consume a lease.
        peak=request.level if self.peak is None else self.peak
        ready=request.readiness if self.ready_peak is None else self.ready_peak
        improved=request.maintained or request.level>peak+self.config.floor or request.readiness>ready+self.config.floor
        if request.level>peak+self.config.floor:peak=request.level
        if request.readiness>ready+self.config.floor:ready=request.readiness
        if feedback is not None:
            improved|=feedback.level>peak+self.config.floor or feedback.readiness>ready+self.config.floor or feedback.maintained or feedback.completed
            if feedback.level>peak+self.config.floor:peak=feedback.level
            if feedback.readiness>ready+self.config.floor:ready=feedback.readiness
        stalls=0 if improved else self.stalls;spent=0 if improved else self.wait_spent
        failures=OrderedDict() if improved else self.failures.copy();a=request.activities[root]
        if not improved:
            # A genuinely useful uncertain attempt can fail by chance. It ends
            # a waiting episode without pretending that the goal progressed.
            # Idle/collision/renaming a prediction cannot renew the wait budget.
            if a.kind=='uncertain' and feedback is not None:spent=0
            if a.kind=='wait':spent=min(1000000,spent+1) # chosen waits consumed even when outcome is missing
            if feedback is not None:
                stalls=min(1000000,stalls+1)
                if a.kind=='attempt':
                    key=digest([root,a.condition]);count=failures.pop(key,0)
                    failures[key]=min(1000000,count+1)
                    while len(failures)>self.config.capacity:failures.popitem(last=False)
        self.peak=peak;self.ready_peak=ready;self.stalls=stalls;self.wait_spent=spent;self.failures=failures;self.last_tick=tick
        return dict(improved=bool(improved),stalls=stalls,wait_spent=spent,completed=feedback.completed if feedback else False)

    def record(self):
        return dict(version='progress-watch-v1',owner=self.owner,config=asdict(self.config),peak=self.peak,ready_peak=self.ready_peak,
            stalls=self.stalls,wait_spent=self.wait_spent,last_tick=self.last_tick,failures=[[k,v] for k,v in self.failures.items()])

    @classmethod
    def from_record(cls,scope,r):
        x=cls(scope,ProgressConfig(**r['config']))
        if r['version']!='progress-watch-v1' or r['owner']!=x.owner:raise ValueError('purpose checkpoint owner mismatch')
        if (r['peak'] is None)!=(r['ready_peak'] is None):raise ValueError('complete purpose peaks required')
        if r['peak'] is not None:level(r['peak']);level(r['ready_peak'])
        for k in ('stalls','wait_spent'):
            if type(r[k]) is not int or not 0<=r[k]<=1000000:raise ValueError('invalid purpose counters')
        if type(r['last_tick']) is not int or not -1<=r['last_tick']<2**63:raise ValueError('invalid purpose timeline')
        if (r['last_tick']==-1)!=(r['peak'] is None):raise ValueError('purpose peaks/timeline mismatch')
        if not isinstance(r['failures'],list) or len(r['failures'])>x.config.capacity:raise ValueError('invalid failure storage')
        for key,count in r['failures']:
            identifier(key)
            if key in x.failures or type(count) is not int or not 1<=count<=1000000:raise ValueError('invalid failure counter')
            x.failures[key]=count
        x.peak=r['peak'];x.ready_peak=r['ready_peak'];x.stalls=r['stalls'];x.wait_spent=r['wait_spent'];x.last_tick=r['last_tick']
        return x
