import unittest
from tools.diagnose_auction_continuity import reference
from tools.continuing_games import ContinuingGame
from reflex.laboratory import PROFILES


class ContinuityDiagnosticTests(unittest.TestCase):
    def test_evaluator_only_reference_matches_full_public_tree(self):
        for variant in (0,1):
            g=ContinuingGame('auction',PROFILES[0],turns=3,variant=variant)
            laws=((.8,.1,.1),(.1,.1,.8),(.1,.8,.1))
            fast=reference(3,variant,lambda t:laws[t])
            def q(w,k):
                return sum(p*v(g.transition(w,k,str(z))[0]) for z,p in enumerate(laws[w['tick']]))
            def v(w):
                return g.goal(w).value if g.terminal(w) else max(q(w,k) for k in g.keys(w))
            w=g.start()
            for key,value in fast(0,8).items():self.assertAlmostEqual(value,q(w,key),places=13)


if __name__=='__main__':unittest.main()
