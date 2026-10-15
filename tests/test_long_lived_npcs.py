import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from reflex.core import compile_batch
from reflex.encounters import EncounterWorld, after_action, run
from reflex.social_withdrawal import WithdrawalSocialPopulation
from tools.long_lived_npcs import inputs, need_state, comparison_outcome, run_one


class LongLivedTests(unittest.TestCase):
    def test_extracted_transition_preserves_prior_rules(self):
        rng=np.random.default_rng(17)
        for _ in range(100):
            before=dict(energy=float(rng.random()),health=float(rng.random()*8),stock=float(rng.random()*8),score=float(rng.random()*5))
            gain=float(rng.uniform(-.4,.8));health=float(rng.uniform(-.5,.3));upkeep=float(rng.choice([.04,.1]))
            for kind in ('cooperate','claim','alone','rest','finished'):
                expected=copy.deepcopy(before)
                if kind!='finished':
                    expected['energy']=min(1,expected['energy']+.4) if kind=='rest' else max(0,expected['energy']-.18)
                    expected['energy']=max(0,expected['energy']-.025)
                    expected['stock']=max(0,expected['stock']+gain-.08)
                    expected['health']=max(0,min(8,expected['health']+health)-upkeep)
                    expected['score']+=gain
                self.assertEqual(expected,after_action(before,kind,gain,health,upkeep))

    def test_net_need_effects_include_maintenance_and_normalization(self):
        for domain in ('exchange','watch'):
            w=EncounterWorld(91000,domain,'balanced4','public')
            own=w.actors[0];w.health[own]=3.;w.stock[own]=2.
            before=need_state(w.state(own));cs,_=inputs(w,net_effects=True)
            a=next(a for a in cs[0]['actions'] if a['id']=='alone')
            choices={x:'alone' for x in w.actors};rows,_=w.resolve(choices)
            after=need_state(rows[0]['after'])
            for name in before:self.assertAlmostEqual(a['outcomes'][0]['needs'][name],before[name]-after[name])
            if domain=='watch':self.assertLess(a['outcomes'][0]['needs']['safety'],0)

    def test_projection_changes_no_legality_probability_or_fixed_axes(self):
        for domain in ('exchange','watch'):
            w=EncounterWorld(91000,domain,'balanced8','free');a,_=inputs(w);b,_=inputs(w,net_effects=True)
            aa,bb=compile_batch(a),compile_batch(b)
            for field in ('legal','probability','traits','values','needs','rng'):
                np.testing.assert_array_equal(getattr(aa,field),getattr(bb,field))
            keep=[i for i in range(22) if i not in (1,2)]
            np.testing.assert_array_equal(aa.effects[:,:,:,keep],bb.effects[:,:,:,keep])

    def test_memory_ablation_changes_only_predictions_and_does_not_touch_world(self):
        w=EncounterWorld(91000,'watch','balanced4','public');own=w.actors[0];other=w.available_partner(own)
        w.history[other].extend([('claim',own)]*8)
        saved=copy.deepcopy(w.__dict__)
        learned,_=inputs(w);forgot,_=inputs(w,forget=True)
        x=compile_batch(learned);y=compile_batch(forgot)
        self.assertFalse(np.array_equal(x.probability,y.probability))
        for field in ('legal','effects','traits','values','needs','rng'):
            np.testing.assert_array_equal(getattr(x,field),getattr(y,field))
        self.assertEqual(w.history,saved['history']);self.assertEqual(w.state(own),{k:saved[k][own] for k in ('energy','stock','health','score')})

    def test_counterfactual_evaluator_matches_actual_execution_without_mutation(self):
        for seed in range(20):
            w=EncounterWorld(seed,'watch','balanced4','public')
            joint={a:'cooperate:'+w.available_partner(a) for a in w.actors}
            tick=w.tick;saved=[w.state(a) for a in w.actors]
            predicted=[comparison_outcome(w,a,joint[a],joint) for a in w.actors]
            self.assertEqual(w.tick,tick);self.assertEqual(saved,[w.state(a) for a in w.actors])
            rows,_=w.resolve(joint);self.assertEqual(predicted,[r['after'] for r in rows])

    def test_long_runner_preserves_legacy_social_actions_including_probe_ticks(self):
        original,trace=run(91000,'exchange','social4','coarse',33,contact='public',population_factory=WithdrawalSocialPopulation)
        with tempfile.TemporaryDirectory() as tmp:
            r=run_one((tmp,91000,'exchange','social4','public','social',33))
            saved=json.loads(gzip.decompress((Path(tmp)/r['trajectory']).read_bytes()))
            self.assertEqual([[x['action'] for x in t['rows']] for t in trace],
                [[x['action'] for x in t['rows']] for t in saved['trace']])
            self.assertGreater(saved['run']['probes']['count'],0)
            for a,b in zip(original['actors'],r['actors']):
                for key in ('score','energy','stock','health'):self.assertEqual(a[key],b[key])


if __name__=='__main__':unittest.main()
