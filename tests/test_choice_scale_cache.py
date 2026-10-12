import unittest
from reflex.integrated_trials import life
from reflex.public_responses import PublicResponses
from reflex.laboratory import profiles
from tools.run_public_peer_reliability import mixed


class ChoiceScaleCacheTests(unittest.TestCase):
    def test_pair_reuses_only_identical_model_inputs_and_preserves_whole_records(self):
        for public in (False,True):
            factory=(lambda c:PublicResponses(c['scope'],profiles())) if public else None
            cache={}
            for gain in (False,True):
                args=dict(turns=4,horizon=2,peer_model_factory=factory,world_setup=mixed,objective_gain=gain)
                a=life(7200,'exchange','balanced4','owned',**args)
                b=life(7200,'exchange','balanced4','owned',forecast_cache=cache,**args)
                self.assertEqual(a,b)
            n=len(cache)
            life(7201,'exchange','balanced4','owned',turns=4,horizon=2,peer_model_factory=factory,world_setup=mixed,forecast_cache=cache)
            self.assertGreater(len(cache),n)

    def test_uncached_default_and_explicit_empty_cache_are_identical(self):
        a=life(7200,'watch','balanced4','owned',turns=3,horizon=2)
        b=life(7200,'watch','balanced4','owned',turns=3,horizon=2,forecast_cache={})
        self.assertEqual(a,b)


if __name__=='__main__':unittest.main()
