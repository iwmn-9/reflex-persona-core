"""Exercise unobserved counterfactuals and terminal branches explicitly."""
import copy
import math
import unittest
from reflex.adaptive_dynamics import PublicEvent
from reflex.event_reader import EventReader
from reflex.core import digest
from reflex.examples import effect
from reflex.laboratory import PROFILES
from tools.continuing_games import ContinuingGame
from tests.test_relative_gain import evaluate


class FactoredReturns(ContinuingGame):
    def __init__(self,early=False):
        super().__init__('gambling',PROFILES[0],turns=8)
        self.early=early
        self.model_id=digest(['factored-returns-v1',self.model_id,early])

    def start(self):
        w=super().start();w.update(quiet=not self.early,ended=False);return w

    def terminal(self,w):return super().terminal(w) or w['ended']
    def keys(self,w):return () if self.terminal(w) else ('p','q')
    def legal(self,c):return ('0',) if self.world(c)['quiet'] else ('0','1')
    def models(self,c):
        legal=self.legal(c)
        return {n:{k:1/len(legal) for k in legal} for n in ('0','1')}
    def needs(self,w):return dict(growth=.3,safety=0.)

    def actor(self,c,**options):
        actor=super().actor(c,**options)
        actor.predictor=EventReader(c['scope'],('0','1'),self.models,self.outcome,
            legal_for=self.legal,model_id=self.model_id,known_conditionals=True,
            factors=('mode',),features_for=lambda c:('quiet' if self.world(c)['quiet'] else 'noisy',),
            learning='segmented')
        return actor

    def events(self,w,key):
        return [PublicEvent(z,*self.transition(w,key,z)) for z in (('0',) if w['quiet'] else ('0','1'))]

    def transition(self,w,key,symbol):
        if key not in self.keys(w):raise ValueError('legal action required')
        n=copy.deepcopy(w)
        gain=1. if key=='p' else 2. if w['quiet'] else 0. if symbol=='0' else 8.
        n.update(tick=w['tick']+1,cash=w['cash']+gain,quiet=key=='p',signal=symbol,last=key,
            ended=self.early and not w['quiet'] and symbol=='1')
        return n,effect(gain/12.,needs={'growth':gain/12.,'safety':0.},values={'achievement':gain/12.})


class PreparationInformationTests(unittest.TestCase):
    def test_same_retry_history_cannot_condition_on_unobserved_ordinary_events(self):
        game=FactoredReturns()
        _,old=evaluate(game,'ordinary',6,seeds=tuple(range(701,709)),tail=False)
        _,new=evaluate(game,'expected',6,seeds=tuple(range(701,709)),tail=False)
        name='candidate-000-retry-until-gain'
        old_rows=old['modeled'][name];new_rows=new['modeled'][name]
        # Every retry sees only event0 while p keeps it in the quiet mode.
        # Ordinary q moves to a noisy mode; its unrealized responses vary.
        ordinary=old['modeled']['candidate-000-reflex']
        self.assertGreater(len({tuple(p['world']['signal'] for p in r['prefixes']) for r in ordinary}),1)
        self.assertGreater(len({r['retry_termination_step'] for r in old_rows}),1)
        self.assertEqual(len({r['retry_termination_step'] for r in new_rows}),1)
        for r in new_rows:
            stop=r['retry_termination_step'] or len(r['prefixes'])
            self.assertEqual(r['actions'][:stop],['p']*stop)
            self.assertTrue(all(p['world']['signal']=='0' for p in r['prefixes'][:stop]))

    def test_early_terminal_goals_are_held_in_the_root_schedule(self):
        _,cap=evaluate(FactoredReturns(early=True),'expected',6,seeds=(701,702),tail=False)
        policies=cap['forecast'].audit['continuation_portfolio']['policies']
        for name,policy in policies.items():
            if not name.endswith('-until-gain'):continue
            root=name.split('-retry')[0]+'-reflex'
            rows=cap['modeled'][root];branches=cap['paths'][root]
            self.assertGreater(len({len(r['prefixes']) for r in rows}),1)
            self.assertTrue(any(r['terminal'] and len(r['prefixes'])==1 for r in rows))
            mass=math.fsum(b.probability for b in branches)
            expected=tuple(math.fsum(b.probability*r['prefixes'][min(t,len(r['prefixes'])-1)]['assessment']['value']
                for b,r in zip(branches,rows))/mass for t in range(6))
            self.assertEqual(tuple(policy['reference_goal_curve']),expected)


if __name__=='__main__':unittest.main()
