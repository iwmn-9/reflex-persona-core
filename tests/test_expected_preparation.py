"""A future switch must not read another possible world's realized outcome."""
import math
import unittest
from reflex.laboratory import PROFILES
from tools.background_preparation import BackgroundChain
from tools.continuing_games import ContinuingGame
from tests.test_continuation_persona import IntrinsicTradeoff
from tests.test_relative_gain import evaluate


class ExpectedPreparationTests(unittest.TestCase):
    def test_one_root_known_schedule_for_all_stochastic_paths(self):
        # The paths include different first responses and future model seeds.
        # Every path must use the same precomputed schedule, including when
        # its paired ordinary realization differs from that expectation.
        for family in ('auction','resources'):
            with self.subTest(family=family):
                game=ContinuingGame(family,PROFILES[0],turns=128)
                _,cap=evaluate(game,'expected',4,seeds=(701,702))
                policies=cap['forecast'].audit['continuation_portfolio']['policies']
                varied=False
                for name,policy in policies.items():
                    if not name.endswith('-until-gain'):continue
                    root=name.split('-retry')[0]+'-reflex'
                    rows=cap['modeled'][root];branches=cap['paths'][root]
                    mass=math.fsum(b.probability for b in branches)
                    expected=tuple(math.fsum(b.probability*r['prefixes'][min(t,len(r['prefixes'])-1)]['assessment']['value']
                        for b,r in zip(branches,rows))/mass for t in range(4))
                    self.assertEqual(tuple(policy['reference_goal_curve']),expected)
                    varied |= any(tuple(p['assessment']['value'] for p in r['prefixes'])!=expected for r in rows)
                    for row in cap['modeled'][name]:
                        crossed=[t+1 for t,p in enumerate(row['prefixes']) if p['assessment']['value']>expected[t]+1e-12]
                        self.assertEqual(row['retry_termination_step'],crossed[0] if crossed else None)
                self.assertTrue(varied,'fixture must distinguish a paired realization from its mean')

    def test_deterministic_reference_and_original_paths_are_preserved(self):
        game=BackgroundChain(1.125)
        a,old=evaluate(game,'ordinary');b,new=evaluate(game,'expected')
        self.assertEqual(a['decision']['action_id'],b['decision']['action_id'])
        self.assertEqual(old['modeled'],new['modeled'])
        self.assertEqual(old['paths'],new['paths'])

    def test_persona_tradeoff_need_override_and_opaque_action_ids(self):
        for tail in (False,True):
            for profile,mode,names,expected in ((0,'principle',('x','y'),'x'),(2,'principle',('x','y'),'y'),
                    (2,'need',('x','y'),'x'),(2,'principle',('zeta','alpha'),'alpha')):
                with self.subTest(tail=tail,profile=profile,mode=mode,names=names):
                    game=IntrinsicTradeoff(PROFILES[profile],names,mode=mode)
                    result,_=evaluate(game,'expected',tail=tail)
                    self.assertEqual(result['decision']['action_id'],expected)


if __name__=='__main__':unittest.main()
