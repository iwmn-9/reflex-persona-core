"""Diagnostic common income; no hint about preparation enters the controller."""
from reflex.core import digest
from reflex.examples import effect
from tools.preparation_world import PreparationChain


class BackgroundChain(PreparationChain):
    def __init__(self, background, delay=5, payout=14., turns=24, profile=None):
        super().__init__(delay, payout=payout, turns=turns, profile=profile)
        self.background=background
        self.model_id=digest(['background-preparation-diagnosis-v1',self.model_id,background])

    def transition(self,world,key,symbol):
        after,_=super().transition(world,key,symbol)
        after['cash']+=self.background
        gain=max(-1.,min(1.,(after['cash']-world['cash'])/12.))
        return after,effect(gain,needs={'growth':gain,'safety':0.},values={'achievement':gain})
