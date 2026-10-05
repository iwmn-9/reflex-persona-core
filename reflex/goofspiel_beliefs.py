"""Game-owned hypothesis distributions and revealed-bid observations only."""
from dataclasses import dataclass
from .opponent_beliefs import HypothesisTracker

NAMES=('uniform','reserve','high','low')
CONTEXT_NAMES=NAMES+('pressure_attack','pressure_conserve')


def under_pressure(s,actor):
    """Game-defined material deficit, not a universal intelligence/personality axis."""
    return max(s.scores)-s.scores[actor]>=max(s.prizes)*.5


def hypotheses(s,actor,names=NAMES):
    h=s.hands[actor]; prize=s.prizes[s.round]
    distances=[abs(c-prize) for c in h]; best=min(distances)
    reserve=[c for c,d in zip(h,distances) if d<=best+1]
    result=dict(uniform={f'BID:{b}':1/len(h) for b in h},
                reserve={f'BID:{b}':(1/len(reserve) if b in reserve else 0.) for b in h},
                high={f'BID:{b}':float(b==max(h)) for b in h},
                low={f'BID:{b}':float(b==min(h)) for b in h})
    if any(n not in NAMES for n in names):
        behind=under_pressure(s,actor)
        result['pressure_attack']=result['high'] if behind else result['reserve']
        result['pressure_conserve']=result['reserve'] if behind else result['high']
    return {n:result[n] for n in names}


@dataclass(frozen=True)
class Forecast:
    viewer: int
    players: tuple

    def probabilities(self,s,actor): return self.players[actor].predict(hypotheses(s,actor,self.players[actor].names))

    @property
    def confidence(self):
        values=[p.confidence for i,p in enumerate(self.players) if i!=self.viewer]
        return sum(values)/len(values)

    @property
    def strongest_confidence(self):
        return max(p.confidence for i,p in enumerate(self.players) if i!=self.viewer)

    @property
    def nonuniform_mass(self):
        # Upper bound on total variation from uniform in ANY future hand/prize
        # for this finite-basis model. A certain uniform opponent needs no rerun.
        return max(p.confidence*(1-p.smoothing)*(1-p.weights[p.names.index('uniform')])
                   for i,p in enumerate(self.players) if i!=self.viewer)

    def record(self):
        return {f'player-{i}':p.record() for i,p in enumerate(self.players) if i!=self.viewer}


class PublicBidBeliefs:
    """One observer's isolated trackers; actual controller names never accepted."""
    def __init__(self,players,viewer,contextual=True,responsive=True):
        if type(players) is not int or not 2<=players<=10 or type(viewer) is not int or not 0<=viewer<players:
            raise ValueError('valid public observer required')
        if type(contextual) is not bool: raise ValueError('contextual must be boolean')
        names=CONTEXT_NAMES if contextual else NAMES
        self.viewer=viewer; self.trackers=tuple(HypothesisTracker(names,responsive=responsive) for _ in range(players)); self.last_round=-1

    def snapshot(self): return Forecast(self.viewer,tuple(t.snapshot() for t in self.trackers))

    def reveal(self,before,joint_bids):
        if before.terminal or len(before.hands)!=len(self.trackers) or len(joint_bids)!=len(self.trackers):
            raise ValueError('matching pre-reveal state required')
        if before.round<=self.last_round: raise ValueError('public observations must advance monotonically')
        if any(type(b) is not int or b not in h for b,h in zip(joint_bids,before.hands)):
            raise ValueError('invalid public joint bids')
        diagnostics={}
        for actor,bid in enumerate(joint_bids):
            if actor==self.viewer: continue
            tracker=self.trackers[actor]
            diagnostics[f'player-{actor}']=tracker.observe(hypotheses(before,actor,tracker.names),f'BID:{bid}',f'round-{before.round}')
        self.last_round=before.round
        return diagnostics
