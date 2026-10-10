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
