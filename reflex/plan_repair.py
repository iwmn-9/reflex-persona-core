"""Local schedule repair under an unchanged goal and owner contract.

Choices come from modeled public observations. A patch is one shared sequence,
not a different hindsight action for each model branch. Fresh search proposals
remain available; edit distance never overrides purpose or personality scores.
"""
from dataclasses import dataclass
from .core import identifier
from .plan_continuity import ArrivalPreference


def patches(retained, choices, *, limit=24):
    """Round-robin single-step substitutions, including an invalid first step.

    The caller must reevaluate each complete sequence on every model branch.
    Branch-illegal future steps use the existing observed feedback fallback.
    """
    if type(limit) is not int or not 0<=limit<=64:raise ValueError('bounded repair bank required')
    if not isinstance(retained,(tuple,list)) or not 1<=len(retained)<=16:
        raise ValueError('bounded remaining sequence required')
    for k in retained:identifier(k)
    if len(choices)!=len(retained):raise ValueError('public choice set for each remaining step required')
    pools=[]
    for pos,keys in enumerate(choices):
        if not isinstance(keys,(set,tuple,list)):raise ValueError('explicit public choice sets required')
        for k in keys:identifier(k)
        pools.append(iter(sorted(set(keys)-{retained[pos]})))
    output=[];active=list(range(len(pools)))
    while active and len(output)<limit:
        next_round=[]
        for pos in active:
            if len(output)>=limit:break
            key=next(pools[pos],None)
            if key is None:continue
            sequence=list(retained);sequence[pos]=key
            output.append(tuple(sequence));next_round.append(pos)
        active=next_round
    return tuple(output)


def edits(old,new):
    return sum(a!=b for a,b in zip(old,new))+abs(len(old)-len(new))


@dataclass(frozen=True)
class RevisionPreference(ArrivalPreference):
    distances: dict

    def validate(self,forecast):
        super().validate(forecast)
        if set(self.distances)!=set(forecast.roots) or any(type(v) is not int or not 0<=v<=16 for v in self.distances.values()):
            raise ValueError('complete bounded schedule edit distances required')

    def resolve_tie(self,names,eligible,subjective,purpose,selected):
        # The old selection remains primary. Change only an exact tie in BOTH
        # subjective score and purpose; do not impose a switching cost.
        ties=[i for i in range(len(names)) if eligible[i] and
              abs(subjective[i]-subjective[selected])<=1e-12 and
              abs(purpose[i]-purpose[selected])<=1e-12]
        return min(ties,key=lambda i:(self.distances[names[i]],names[i]))
