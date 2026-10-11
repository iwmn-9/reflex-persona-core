"""Scoped, finite public behavior evidence for an unsigned teammate.

Reuses HypothesisTracker. A model forecast supplies no observation. Only a later
witnessed action updates beliefs; winning/losing never identifies a personality.
"""
import copy
from .core import digest,identifier
from .opponent_beliefs import HypothesisTracker,distributions


class PartnerMemory:
    def __init__(self,game,episode,group,actors,names=('hold','independent')):
        for x in (game,episode,group):identifier(x)
        actors=tuple(actors)
        if not 1<=len(actors)<=64 or len(set(actors))!=len(actors):raise ValueError('bounded distinct observed partners')
        if any(type(a) is not int or a<0 for a in actors):raise ValueError('declared public partner IDs')
        self.scope=(game,episode,group);self.actors=actors;self.names=tuple(names)
        self.trackers={a:HypothesisTracker(names) for a in actors};self.pending=None;self.last_tick=-1

    def weights(self,actor):
        s=self.trackers[actor].snapshot()
        # Bounded model-type posterior, not certainty about future actions.
        # Keep a uniform type component, including after repeated observations.
        return tuple(.92*w+.08/len(s.names) for w in s.weights)

    def begin(self,scope,tick,models):
        if tuple(scope)!=self.scope or type(tick) is not int or tick<=self.last_tick or self.pending is not None:
            raise ValueError('fresh tick in the same cooperation episode required')
        if not set(models)<=set(self.actors):raise ValueError('undeclared partner evidence')
        for actor,m in models.items():distributions(m,self.names,self.trackers[actor].smoothing)
        self.pending=dict(tick=tick,models=copy.deepcopy(models))
        return {a:self.weights(a) for a in models}

    def observe(self,scope,tick,actions,*,witnessed):
        if tuple(scope)!=self.scope or self.pending is None or tick!=self.pending['tick']:
            raise ValueError('matching unresolved public evidence ticket required')
        if type(witnessed) is not bool or not witnessed:raise ValueError('only witnessed actual teammate actions')
        if set(actions)!=set(self.pending['models']):raise ValueError('one revealed action per pending partner')
        staged=copy.deepcopy(self.trackers);receipts={}
        for actor,action in actions.items():
            event=digest([self.scope,tick,actor,'revealed-partner'])
            receipts[str(actor)]=staged[actor].observe(self.pending['models'][actor],action,event)
        self.trackers=staged;self.last_tick=tick;self.pending=None
        return receipts

    def receipt(self):
        return dict(scope=list(self.scope),actors=list(self.actors),last_tick=self.last_tick,
            beliefs={str(a):t.snapshot().record() for a,t in self.trackers.items()},
            weights={str(a):list(self.weights(a)) for a in self.actors},
            source='revealed public actions under declared hypotheses; not other private memory or calibrated future certainty')

    def abandon(self,scope,tick):
        if tuple(scope)!=self.scope or self.pending is None or tick!=self.pending['tick']:
            raise ValueError('matching missing-observation ticket required')
        self.last_tick=tick;self.pending=None
