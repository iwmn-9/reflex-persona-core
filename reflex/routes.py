"""Finite game-proposed victory routes scored by the unchanged personality policy.

No route name/genre/Big Five-to-victory lookup lives here. Games provide bounded
effects and known availability. Hysteresis is explicit, not a learned win model.
Route memory is separate from per-action intent so their costs are not confused.
"""
from dataclasses import dataclass
import copy
import math
from .core import Policy, compile_batch


@dataclass(frozen=True)
class RouteState:
    chosen: str | None = None
    switches: int = 0


def choose_route(c,state=None,uncertainty=0.,switch_margin=.06,stochastic=False,policy=None):
    state=state or RouteState()
    if not isinstance(state,RouteState): raise ValueError('route memory required')
    if not 0<=uncertainty<=1 or not 0<=switch_margin<=1: raise ValueError('bounded switching controls required')
    scoped=copy.deepcopy(c)
    # Route switching is paid HERE; per-action intent belongs to a different
    # choice space. Reusing that memory would silently double-charge switches.
    scoped['state']['intent_action']=None
    batch=compile_batch([scoped]); numeric=(policy or Policy()).decide(batch,stochastic=stochastic)
    decision=numeric.records(batch)[0]; new=decision['action_id']; reason='initial_preference'
    scores={name:float(numeric.scores[0,i]) for i,name in enumerate(batch.ids[0])}
    eligible={name:bool(numeric.eligible[0,i]) for i,name in enumerate(batch.ids[0])}
    if state.chosen is not None and new!=state.chosen:
        if not eligible.get(state.chosen,False): reason='old_route_unavailable_or_rejected'
        elif scores[new]-scores[state.chosen]<=switch_margin+.1*uncertainty:
            new=state.chosen; reason='retain_axis_below_switch_margin'
        else: reason='material_persona_and_opportunity_gain'
    elif new==state.chosen: reason='same_route'
    switched=state.chosen is not None and state.chosen!=new
    return RouteState(new,state.switches+int(switched)),dict(chosen=new,reason=reason,
        switched=switched,scores=scores,eligible=eligible,context=scoped,
        candidate=decision['action_id'],required_gain=switch_margin+.1*uncertainty)
