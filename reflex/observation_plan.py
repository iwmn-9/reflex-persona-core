"""Finite public-history decisions; no world state or model seed in this API.

The adapter still owns perception. A context must contain only information its
owner can use. Unknown/obsolete nodes defer to the declared persona reflex.
"""
from dataclasses import dataclass
import copy
import re
from .core import compile_batch,digest,identifier,owner_key


def owner(c):
    return owner_key(c)


def history(value):
    if value != '' and (not isinstance(value,str) or re.fullmatch('[0-9a-f]{64}',value) is None):
        raise ValueError('public-history hash required')
    return value


def node_key(prefix,c,allowed):
    """Canonical perceived input, owned state and past actions, not a scenario ID."""
    history(prefix);compile_batch([c])
    view=copy.deepcopy(c)
    view['actions']=sorted(view['actions'],key=lambda a:a['id'])
    for a in view['actions']:a['outcomes']=sorted(a['outcomes'],key=digest)
    viable={a['id'] for a in c['actions'] if a['legal'] and not a['known_failure']}
    permitted=tuple(sorted(viable if allowed is None else set(allowed)))
    if not permitted or not set(permitted)<=viable:raise ValueError('nonempty observed viable permission required')
    return digest([prefix,view,permitted])


def after_action(node,key):
    history(node);identifier(key)
    return digest([node,key])


@dataclass(frozen=True)
class View:
    context: dict
    allowed: tuple
    prefix: str
    node: str

    @classmethod
    def make(cls,prefix,c,allowed=None):
        viable={a['id'] for a in c['actions'] if a['legal'] and not a['known_failure']}
        permitted=tuple(sorted(viable if allowed is None else set(allowed)))
        return cls(copy.deepcopy(c),permitted,history(prefix),node_key(prefix,c,permitted))


@dataclass(frozen=True)
class ObservationPlan:
    owner: str
    target: str
    unit: str
    entries: tuple=()

    def __post_init__(self):
        history(self.owner);identifier(self.target);identifier(self.unit)
        if not self.owner:raise ValueError('nonempty observation program owner required')
        if not isinstance(self.entries,tuple) or len(self.entries)>128:
            raise ValueError('bounded immutable observation program required')
        names=[]
        for item in self.entries:
            if not isinstance(item,tuple) or len(item)!=2:raise ValueError('node/action pairs required')
            node,key=item;history(node);identifier(key);names.append(node)
            if not node:raise ValueError('nonempty observation node required')
        if len(names)!=len(set(names)) or list(self.entries)!=sorted(self.entries):
            raise ValueError('unique sorted observation program required')

    @classmethod
    def make(cls,c,target,unit,entries=None):
        return cls(owner(c),target,unit,tuple(sorted((entries or {}).items())))

    def choose(self,view):
        if not isinstance(view,View) or owner(view.context)!=self.owner:
            raise ValueError('observation program belongs to another owner')
        key=dict(self.entries).get(view.node)
        return key if key in view.allowed else None

    def patch(self,node,key):
        data=dict(self.entries);data[node]=key
        return ObservationPlan(self.owner,self.target,self.unit,tuple(sorted(data.items())))

    def record(self):
        return dict(format='observation-plan-v1',owner=self.owner,target=self.target,unit=self.unit,entries=[list(p) for p in self.entries])

    @classmethod
    def restore(cls,record):
        if record['format']!='observation-plan-v1':raise ValueError('known observation program format required')
        return cls(record['owner'],record['target'],record['unit'],tuple(tuple(p) for p in record['entries']))


class PlanMemory:
    """Keep the executed public prefix and a selected conditional program.

    No forecast becomes empirical learning. If an actual observation differs,
    its unmatched node falls back; new search can patch that current reality.
    """
    def __init__(self,c,target,unit='public-turns'):
        compile_batch([c]);self.owner=owner(c);self.target=target;self.unit=unit
        identifier(target);identifier(unit);self.prefix='';self.plan=None

    def offer(self,c):
        if owner(c)!=self.owner:raise ValueError('owned plan memory required')
        return self.plan,self.prefix

    def remember(self,c,forecast,selected,key):
        if owner(c)!=self.owner or forecast.target!=self.target:
            raise ValueError('owned same-purpose conditional plan required')
        bank=forecast.audit['contingent_search']
        if bank['unit']!=self.unit:raise ValueError('same conditional plan time unit required')
        if bank['prefix']!=self.prefix:raise ValueError('stale conditional plan prefix')
        plan=None
        if selected is not None:
            if forecast.roots[selected]!=(key,):raise ValueError('remember only the executed root')
            proposal=forecast.audit['plans'][selected]['proposal']
            plan=ObservationPlan.restore(bank['proposals'][proposal]['program'])
            if (plan.owner,plan.target,plan.unit)!=(self.owner,self.target,self.unit):
                raise ValueError('selected program owner, purpose or time unit mismatch')
        if key not in bank['all_viable_roots']:raise ValueError('remember only a viable executed root')
        self.prefix=after_action(bank['root_node'],key);self.plan=plan

    def record(self):
        return dict(format='observation-memory-v1',owner=self.owner,target=self.target,unit=self.unit,prefix=self.prefix,
                    plan=None if self.plan is None else self.plan.record())

    @classmethod
    def restore(cls,c,record):
        if record['format']!='observation-memory-v1':raise ValueError('known owned plan checkpoint required')
        result=cls(c,record['target'],record['unit'])
        if record['owner']!=result.owner:raise ValueError('checkpoint belongs to another owner')
        result.prefix=history(record['prefix'])
        result.plan=None if record['plan'] is None else ObservationPlan.restore(record['plan'])
        if result.plan is not None and (result.plan.owner,result.plan.target,result.plan.unit)!=(result.owner,result.target,result.unit):
            raise ValueError('checkpoint purpose or owner mismatch')
        return result
