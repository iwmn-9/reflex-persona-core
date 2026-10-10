"""Finite support for learned predictions, separate from personality.

An adapter declares public opportunity classes before observing actions. A bank
cannot borrow observations from another class. Unknown classes fail explicitly;
there is no implicit unbounded context creation or fallback to unrelated data.
This mechanism does not infer game rules or psychological identities.
"""
from .strong_search import PublicMemory
from collections import deque
from .prediction_support import SupportBanks


class SupportedPublicMemory:
    """Tabletop adapter: future draws versus current-card settlement.

No Thanks' last card may still involve bargaining and owner redecisions. The
class certifies only the absence of later draws, not a forced terminal action.
Goofspiel remains a one-bank negative control: its last bid is already forced.
Opponent IDs and actual private personality/controller state never enter keys.
"""
    def __init__(self,game,viewer,players=4):
        # Construct first to reuse all existing observer/game validation.
        first=PublicMemory(game,viewer,players)
        self.game,self.viewer,self.players=game,viewer,players
        classes=('future_draws','current_card_only') if game=='no_thanks' else ('bidding',)
        self.support=SupportBanks(classes,lambda:PublicMemory(game,viewer,players))
        self.support._banks[classes[0]]=first
        self.ids=[deque(maxlen=64) for _ in range(players)]

    def opportunity(self,s):
        return 'bidding' if self.game=='goofspiel' else ('future_draws' if s.remaining>0 else 'current_card_only')

    def bank(self,s):return self.support.select(self.opportunity(s))

    def models(self,s,actor,use_history=True):return self.bank(s).models(s,actor,use_history)
    def predict(self,s,actor,adaptive=True,models=None):return self.bank(s).predict(s,actor,adaptive,models)
    def features(self,s,actor):return self.bank(s).features(s,actor)
    def forecast_weights(self,s,actor,adaptive=True):return self.bank(s).weights(actor,adaptive)
    def forecast_coefficients(self,s,actor,adaptive=True):
        bank=self.bank(s)
        return bank.coefficients[actor] if adaptive else bank.initial_coefficients

    def rollout_banks(self,adaptive=True):return [self.support.select(k).rollout_banks(adaptive)[0] for k in self.support.classes]
    def rollout_bank_indices(self,remaining):return (remaining==0).astype(int) if self.game=='no_thanks' else remaining*0

    def observe(self,s,actor,revealed,observation_id):
        if actor!=self.viewer and observation_id in self.ids[actor]:raise ValueError('duplicate public observation across support banks')
        record=self.bank(s).observe(s,actor,revealed,observation_id)
        if record is not None:
            self.ids[actor].append(observation_id)
            record['opportunity_class']=self.opportunity(s)
        return record

    def record(self):
        banks={k:self.support.select(k).record() for k in self.support.classes}
        return [dict(support_version='public-opportunity-v1',opportunity_classes=list(self.support.classes),
                     banks={k:v[a] for k,v in banks.items()}) for a in range(self.players)]
