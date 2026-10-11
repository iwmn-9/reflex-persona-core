"""Owned hypothetical observers reconstructed only from witnessed events.

No actual counterpart memory/state/profile is accepted. The factory declares
how a hypothetical observer learns. State projections/visibility are supplied
by the game; the builder never invents private events or a private belief.
"""
import copy
from .core import digest


class PublicPerspectives:
    def __init__(self,game,scope,owner,actors,factory):
        actors=tuple(actors)
        if not actors or len(actors)>256 or len(set(actors))!=len(actors) or owner not in actors:
            raise ValueError('unique finite public participants including the owner required')
        self.game=game;self.scope=scope;self.owner=owner;self.actors=actors
        self._views={a:factory(a) for a in actors};self.delivered=0

    def observe(self,state,actor,action,event_id,witnesses):
        witnesses=tuple(witnesses)
        if actor not in self._views or len(set(witnesses))!=len(witnesses) or not set(witnesses)<=set(self.actors) or self.owner not in witnesses:
            raise ValueError('only an event witnessed by this owner may enter its models')
        staged=copy.deepcopy(self._views)
        for viewer in witnesses:
            if viewer!=actor:staged[viewer].observe(state,actor,action,event_id)
        self._views=staged;self.delivered+=1

    def snapshot(self,game,scope,owner):
        if (game,scope,owner)!=(self.game,self.scope,self.owner):
            raise ValueError('hypothetical public history belongs to another owner/world')
        return copy.deepcopy(self._views)

    def receipt(self):
        return dict(game=self.game,scope=self.scope,owner=self.owner,actors=list(self.actors),
            delivered_events=self.delivered,belief_digest=digest({str(a):v.record() for a,v in self._views.items()}),
            source='hypothetical observers regenerated from witnessed events under declared learners; no actual counterpart state copied')
