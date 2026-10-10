"""Finite support for caller-declared public prediction opportunities."""


class SupportBanks:
    """Bounded, game-independent isolation of mutable learning banks.

    Classes are declared before observations. Unknown classes fail, rather than
    implicitly borrowing unrelated experience or creating unlimited memories.
    Adapters, not personality traits or private identities, define support.
    """
    def __init__(self,classes,factory):
        classes=tuple(classes)
        if not 1<=len(classes)<=16 or any(not isinstance(k,str) or not 0<len(k)<=128 for k in classes) or len(set(classes))!=len(classes):
            raise ValueError('1..16 distinct bounded public opportunity classes required')
        self.classes=classes
        self._banks={k:factory() for k in classes}
        if len({id(bank) for bank in self._banks.values()})!=len(classes):raise ValueError('support banks must own distinct state')

    def select(self,opportunity):
        if not isinstance(opportunity,str) or opportunity not in self._banks:raise ValueError('unregistered public opportunity class')
        return self._banks[opportunity]


def gate_record(gate):
    return dict(capacity=gate.capacity,window=gate.window,min_trials=gate.min_trials,margin=gate.margin,
                entries=[dict(key=k,state=v['state'],gains=list(v['gains'])) for k,v in gate.entries.items()])


def restored_gate(record):
    import math
    from collections import deque
    from .decision_loop import EvidenceGate
    gate=EvidenceGate(record['capacity'],record['window'],record['min_trials'],record['margin'])
    if not isinstance(record['entries'],list) or len(record['entries'])>gate.capacity:raise ValueError('invalid transfer gate capacity')
    for row in record['entries']:
        key=row['key'];gains=row['gains'];state=row['state']
        if not isinstance(key,str) or not 0<len(key)<=512 or key in gate.entries or state not in ('trial','active','revoked'):
            raise ValueError('invalid transfer gate key/state')
        if not isinstance(gains,list) or not 1<=len(gains)<=gate.window or any(not math.isfinite(g) or abs(g)>1 for g in gains):
            raise ValueError('invalid transfer gate observations')
        if state=='active' and len(gains)<gate.min_trials:raise ValueError('unsupported transfer approval')
        gate.entries[key]=dict(state=state,gains=deque(gains,maxlen=gate.window))
    return gate
