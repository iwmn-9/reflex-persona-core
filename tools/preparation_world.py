"""Diagnostic public delayed returns; opaque actions and no controller hints."""
import copy
from reflex.adaptive_dynamics import PublicEvent
from reflex.core import digest
from reflex.examples import effect
from reflex.laboratory import PROFILES
from tools.continuing_games import ContinuingGame


class DelayedYield(ContinuingGame):
    def __init__(self,delay,payout=6.,turns=24,profile=None):
        super().__init__('gambling',PROFILES[0] if profile is None else profile,turns)
        self.delay=delay;self.payout=payout
        self.model_id=digest(['delayed-yield-v1',delay,turns])

    def start(self):
        w=super().start();w['payout']=self.payout
        return w

    def keys(self,w):
        if self.terminal(w):return ()
        return ('x',)+(('y',) if w['investment']==0 and w['cash']>=3 else ())

    def needs(self,w):return dict(growth=.3,safety=0.)
    def models(self,c):return {'0':{'0':1.},'1':{'0':1.}}
    def events(self,w,key):return [PublicEvent('0',*self.transition(w,key,'0'))]

    def actor(self,c,**kwargs):
        own=super().actor(c,**kwargs)
        own.predictor.legal_for=lambda c:('0',)
        return own

    def transition(self,w,key,symbol):
        if key not in self.keys(w):raise ValueError('illegal diagnostic action')
        n=copy.deepcopy(w);n.update(tick=w['tick']+1,signal='0',last=key)
        if key=='x':n['cash']+=1.
        else:n['cash']-=3.;n['investment']=self.delay
        if w['investment']:
            n['investment']-=1
            if n['investment']==0:n['cash']+=w['payout']
        gain=(n['cash']-w['cash'])/12.
        return n,effect(gain,needs={'growth':gain,'safety':0.},values={'achievement':gain})
