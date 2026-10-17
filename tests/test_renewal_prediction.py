import itertools,json,unittest
import numpy as np
from reflex.renewal_prediction import categorical_renewal
from reflex.rollout_sampling import stratified_uniforms
from reflex.adaptive_dynamics import EventLearner


def exhaustive(seq,k,h):
    # Independent oracle: enumerate all partitions, integrate each segment's
    # Dirichlet likelihood, then average its next-symbol posterior.
    weighted=np.zeros(k);total=0.
    for cuts in itertools.product((False,True),repeat=max(0,len(seq)-1)):
        mass=1.;counts=np.full(k,.5)
        for i,z in enumerate(seq):
            if i:
                mass*=h if cuts[i-1] else 1-h
                if cuts[i-1]:counts=np.full(k,.5)
            mass*=counts[z]/sum(counts);counts[z]+=1
        weighted+=mass*(h/k+(1-h)*counts/sum(counts));total+=mass
    return weighted/total


class RenewalPredictionTests(unittest.TestCase):
    def test_conditioned_learning_is_equivariant_to_factor_and_symbol_names(self):
        left=EventLearner(owner='npc',model_id='one',factors=('x','y'),renewal=True)
        right=EventLearner(owner='npc',model_id='two',factors=('renamed-y','renamed-x'),renewal=True)
        a=left.start();b=right.start();mapping={'a':'q','b':'z','c':'m'}
        for i in range(60):
            context=(str(i%2),str((i//2)%3));symbol='c' if i%2 else 'a'
            a=left.observe(a,context,('a','b','c'),{},symbol)
            b=right.observe(b,tuple(reversed(context)),('m','q','z'),{},mapping[symbol])
        p=left.predict(a,('1','2'),('a','b','c'),{})
        q=right.predict(b,('2','1'),('m','q','z'),{})
        self.assertGreater(p['c'],.75)
        for k,v in p.items():self.assertAlmostEqual(v,q[mapping[k]],places=13)

    def test_matches_independent_exact_partition_enumeration(self):
        for seq in ((),(0,),(0,0,2),(1,1,1,0,2,2)):
            for h in (.03,.4):
                np.testing.assert_allclose(categorical_renewal(seq,3,h),exhaustive(seq,3,h),atol=1e-14)

    def test_innovations_after_change_and_no_forced_change_clock(self):
        before=categorical_renewal((0,)*80,3,1/32)
        after=categorical_renewal((0,)*80+(2,)*8,3,1/32)
        outlier=categorical_renewal((0,)*80+(2,),3,1/32)
        self.assertGreater(before[0],.9);self.assertGreater(after[2],.8)
        self.assertGreater(outlier[0],outlier[2])

    def test_symbol_equivariance(self):
        seq=(0,1,1,2,0,2,2,2);mapping=(2,0,1)
        p=categorical_renewal(seq,3,.1)
        q=categorical_renewal(tuple(mapping[x] for x in seq),3,.1)
        for i in range(3):self.assertAlmostEqual(p[i],q[mapping[i]])

    def test_owner_checkpoint_and_legacy_identity(self):
        old=EventLearner(owner='a',model_id='rules')
        explicit=EventLearner(owner='a',model_id='rules',renewal=False)
        self.assertEqual(old.record(old.start()),explicit.record(explicit.start()))
        m=EventLearner(owner='a',model_id='rules',renewal=True);s=m.start()
        for z in ('c',)*12:s=m.observe(s,(),('a','b','c'),{},z)
        saved=json.loads(json.dumps(m.record(s)))
        self.assertEqual(m.restore(saved),s)
        self.assertEqual(m.start().observations,0)
        for wrong in (old,EventLearner(owner='b',model_id='rules',renewal=True)):
            with self.assertRaises(ValueError):wrong.restore(saved)
        # A different legal alphabet is not evidence for this stratum.
        self.assertEqual(m.predict(s,(),('x','y'),{}),dict(x=.5,y=.5))

    def test_sampler_strata_step_independence_and_bank_marginals(self):
        seeds=tuple(range(12));u=stratified_uniforms(seeds,4)
        self.assertEqual(sorted(int(x*12) for x in u),list(range(12)))
        self.assertEqual(u,stratified_uniforms(seeds,4))
        self.assertNotEqual(tuple(np.argsort(u)),tuple(np.argsort(stratified_uniforms(seeds,5))))
        means=np.mean([stratified_uniforms(tuple(range(i,i+3)),7) for i in range(300)],axis=0)
        self.assertTrue(np.all(np.abs(means-.5)<.05))

    def test_invalid_limits(self):
        for args in [((3,),3,.1),((),0,.1),((),3,0),((0,)*513,3,.1)]:
            with self.assertRaises(ValueError):categorical_renewal(*args)
        with self.assertRaises(ValueError):stratified_uniforms((1,1),0)


if __name__=='__main__':unittest.main()
