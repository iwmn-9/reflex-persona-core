"""Exact tile equivalence, persistent actor state and invalid update isolation."""
from dataclasses import fields,replace
import copy
import unittest
import numpy as np
from reflex.core import Batch,Decisions,Policy,compile_batch
from reflex.examples import families
from reflex.runtime import Population
from reflex.scaling import TiledPolicy,array_bytes
from reflex.scale_experiment import contexts,combat_run


class ScalingTests(unittest.TestCase):
    def same(self,left,right,kind):
        for f in fields(kind):
            a,b=getattr(left,f.name),getattr(right,f.name)
            if isinstance(a,np.ndarray): np.testing.assert_array_equal(a,b,err_msg=f.name)
            else: self.assertEqual(a,b)

    def test_all_semantic_cases_both_modes_residual_and_tile_boundaries(self):
        cs=[row['context'] for row in families()]
        batch=compile_batch(cs)
        for residual in (None,np.linspace(-.04,.04,8)):
            for stochastic in (False,True):
                expected=Policy(residual).decide(batch,stochastic)
                for tile in (1,3,len(cs)-1,len(cs),len(cs)+1):
                    actual=TiledPolicy(tile,residual).decide(batch,stochastic)
                    self.same(expected,actual,Decisions)
                    self.assertEqual(expected.records(batch),actual.records(batch))

    def test_dynamic_steps_keep_every_state_field_and_counter_identical(self):
        cs=contexts(37,12,2); reference=Population(cs); tiled=Population(cs,TiledPolicy(7))
        traits=reference.batch.traits.copy(); values=reference.batch.values.copy()
        for tick in range(9):
            needs=np.clip(reference.batch.needs+.02*(tick%3-1),0,1)
            effects=reference.batch.effects.copy(); effects[:,:,:,0]*=-1
            legal=reference.batch.legal.copy(); legal[:,tick%10]=False
            probability=reference.batch.probability.copy(); probability[:,:,0]=.25;probability[:,:,1]=.75
            kwargs=dict(needs=needs,effects=effects,legal=legal,probability=probability)
            a=reference.step(**kwargs); b=tiled.step(**kwargs)
            self.same(a,b,Decisions); self.same(reference.batch,tiled.batch,Batch)
            np.testing.assert_array_equal(reference.ticks,tiled.ticks)
            self.assertEqual(reference.action_ids(a),tiled.action_ids(b))
        np.testing.assert_array_equal(tiled.batch.traits,traits)
        np.testing.assert_array_equal(tiled.batch.values,values)

    def test_tile_borrowed_inputs_frozen_and_no_output_alias(self):
        b=compile_batch(contexts(19)); before=b.effects.copy()
        d=TiledPolicy(3).decide(b); d.scores[:]=0
        np.testing.assert_array_equal(before,b.effects)
        self.assertFalse(b.effects.flags.writeable)
        self.assertFalse(np.shares_memory(d.features,b.effects))
        self.assertEqual(array_bytes(b),sum(getattr(b,f.name).nbytes for f in fields(Batch) if isinstance(getattr(b,f.name),np.ndarray)))

    def test_last_tile_failure_does_not_commit_partial_population(self):
        pop=Population(contexts(11),TiledPolicy(3)); before=pop.batch;ticks=pop.ticks.copy()
        legal=before.legal.copy();legal[-1]=False
        with self.assertRaisesRegex(ValueError,'viable'): pop.step(legal=legal)
        self.assertIs(pop.batch,before);np.testing.assert_array_equal(pop.ticks,ticks)

    def test_reordered_actors_have_same_decisions(self):
        cs=contexts(31); b=compile_batch(cs); order=np.random.default_rng(5).permutation(len(cs))
        a=TiledPolicy(6).decide(b); reordered=TiledPolicy(6).decide(b.take(order))
        for f in fields(Decisions): np.testing.assert_array_equal(getattr(a,f.name)[order],getattr(reordered,f.name))

    def test_public_combat_worlds_match_across_tiles(self):
        a=combat_run(3,0,ticks=3);b=combat_run(3,4,ticks=3)
        self.assertEqual(a['state_hashes'],b['state_hashes'])
        self.assertEqual(a['decisions'],b['decisions'])
        old=combat_run(3,0,ticks=3,geometry_cache=False)
        self.assertEqual(a['state_hashes'],old['state_hashes'])

    def test_terrain_cache_matches_uncached_bfs_on_all_map_cells(self):
        from reflex.combat import MAPS,WIDTH,HEIGHT,terrain_zone_distance
        for geometry in MAPS.values():
            for x in range(WIDTH):
                for y in range(HEIGHT):
                    args=(geometry['walls'],(x,y))
                    self.assertEqual(terrain_zone_distance(*args),terrain_zone_distance.__wrapped__(*args))
        self.assertEqual(terrain_zone_distance.cache_info().maxsize,512)

    def test_tile_size_contract(self):
        for invalid in (0,-1,1.5,True,'8'):
            with self.assertRaises(ValueError): TiledPolicy(invalid)


if __name__=='__main__':unittest.main()
