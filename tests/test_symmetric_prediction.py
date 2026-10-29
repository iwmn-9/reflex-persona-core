import copy
import itertools
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from tools.symmetric_prediction import SymmetricEventLearner, complete_templates
from reflex.segmented_prediction import SegmentedEventLearner


class SymmetricPredictionTests(unittest.TestCase):
    arm = 'symmetric'
    def test_label_equivariance_and_unknown_template_transfer(self):
        labels = ('a', 'b', 'c')
        models = {'given': dict(zip(labels, (.8,.1,.1)))}
        learner = SymmetricEventLearner(owner='one', model_id='law', names=('given',))
        memory = learner.start()
        np.testing.assert_allclose(list(learner.predict(memory, (), labels, models).values()), [1/3]*3, atol=1e-14)
        for z in ('c','c','a','c','b','c'):
            memory = learner.observe(memory, (), labels, models, z)
        p = learner.predict(memory, (), labels, models)
        self.assertGreater(p['c'], p['a'])
        for order in itertools.permutations(labels):
            mapping = dict(zip(labels, order))
            transformed = {'given': {mapping[z]: q for z,q in models['given'].items()}}
            other = learner.start()
            for z in ('c','c','a','c','b','c'):
                other = learner.observe(other, (), labels, transformed, mapping[z])
            q = learner.predict(other, (), labels, transformed)
            for z in labels:
                self.assertAlmostEqual(p[z], q[mapping[z]], places=13)

    def test_exact_small_posterior_and_bounded_completion(self):
        from tests.test_segmented_prediction import enumerated
        labels = ('a','b','c')
        models = {'given': dict(zip(labels, (.7,.2,.1)))}
        learner = SymmetricEventLearner(owner='one', model_id='law', names=('given',))
        memory = learner.start()
        sequence = (2,0,2,1)
        for z in sequence:
            memory = learner.observe(memory, (), labels, models, labels[z])
        expected = enumerated(sequence, complete_templates(((.7,.2,.1),),3), 3)
        np.testing.assert_allclose(list(learner.predict(memory, (), labels, models).values()), expected, atol=1e-14)
        for n in (5,8):
            labels = tuple(map(str, range(n)))
            models = {'given': dict(zip(labels, map(float, np.arange(1,n+1)/sum(range(1,n+1)))))}
            learners = [klass(owner='one',model_id='law',names=('given',)) for klass in (SegmentedEventLearner,SymmetricEventLearner)]
            memories = [l.start() for l in learners]
            for z in (labels[-1],labels[0],labels[-1]):
                memories = [l.observe(m,(),labels,models,z) for l,m in zip(learners,memories)]
            self.assertEqual(*[l.predict(m,(),labels,models) for l,m in zip(learners,memories)])

    def test_predictive_branches_save_restore_and_no_factual_contamination(self):
        from tools.run_adverse_recovery import actor_for, game_for
        from reflex.decision_loop import DecisionLoop
        spec = dict(game='gambling',profile=1,turns=10,seed=123,mechanism='world')
        g = game_for(spec)
        c = g.context(g.start())
        actor = actor_for(g,c,self.arm)
        reader = actor.predictor
        before = reader.record()
        forecast = reader.predictive_continuation(steps=3)
        mass = 0.
        for path in itertools.product(('0','1','2'), repeat=3):
            branch = copy.deepcopy(reader)
            probability = 1.
            context = copy.deepcopy(c)
            for z in path:
                p = forecast(branch,context)
                self.assertAlmostEqual(sum(p.values()),1.)
                probability *= p[z]
                branch,_ = branch.updated_event(context,{'revealed_action':z},'hypothetical')
                context['tick'] += 1
            restored = reader.restored(json.loads(json.dumps(branch.record())))
            self.assertEqual(restored.record(),branch.record())
            mass += probability
        self.assertAlmostEqual(mass,1.,places=13)
        self.assertEqual(reader.record(),before)
        with self.assertRaises(ValueError):
            actor_for(g,c,'stable').predictor.restored(reader.record())
        rebuilt = DecisionLoop.from_record(c,json.loads(json.dumps(actor.record())),predictor=actor.predictor)
        self.assertEqual(actor.record(),rebuilt.record())

    def test_full_and_resumed_game_equality_across_rule_families(self):
        from tools.run_adverse_recovery import run
        for game in ('auction','combat','gambling','resources'):
            spec = dict(id=f'{game}-transfer-check',game=game,profile=1,turns=3,seed=123,
                        mechanism='world',regime='outside')
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                full=run(root/'full',spec,self.arm)
                self.assertIsNone(run(root/'resume',spec,self.arm,max_decisions=1))
                resumed=run(root/'resume',spec,self.arm)
                for row in (full,resumed): row.pop('seconds')
                self.assertEqual(full,resumed)
                self.assertEqual(full['replayed_actual_steps'],3)
                self.assertEqual(full['checks'],dict(fixed_axes=3,owner_parity=3,restored=3))


if __name__ == '__main__':
    unittest.main()
