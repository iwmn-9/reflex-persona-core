import itertools
import unittest
import numpy as np
from reflex.segmented_prediction import HAZARDS, SegmentedEventLearner
from tools.template_transfer_prediction import TemplateTransferLearner, transfer_prediction
from tests.test_symmetric_prediction import SymmetricPredictionTests


class TemplateTransferTests(unittest.TestCase):
    arm = 'transfer'
    test_predictive_branches_save_restore_and_no_factual_contamination = SymmetricPredictionTests.test_predictive_branches_save_restore_and_no_factual_contamination
    test_full_and_resumed_game_equality_across_rule_families = SymmetricPredictionTests.test_full_and_resumed_game_equality_across_rule_families

    def test_initial_labeled_prior_and_bounded_fallback_are_retained(self):
        for n in (2,3,4,5):
            labels=tuple(map(str,range(n)))
            models={'known':dict(zip(labels,map(float,np.arange(1,n+1)/sum(range(1,n+1)))))}
            learners=[cls(owner='public',model_id='one',names=('known',)) for cls in (SegmentedEventLearner,TemplateTransferLearner)]
            memories=[l.start() for l in learners]
            p=[l.predict(m,(),labels,models) for l,m in zip(learners,memories)]
            np.testing.assert_allclose(list(p[0].values()),list(p[1].values()),atol=1e-14)
            for z in (labels[-1],labels[-1],labels[0]):
                memories=[l.observe(m,(),labels,models,z) for l,m in zip(learners,memories)]
            if n>4:
                self.assertEqual(*[l.predict(m,(),labels,models) for l,m in zip(learners,memories)])
        baseline=SegmentedEventLearner(owner='public',model_id='one')
        candidate=TemplateTransferLearner(owner='public',model_id='one')
        a,b=baseline.start(),candidate.start()
        for z in ('b','b','a'):
            a=baseline.observe(a,(),('a','b'),{},z)
            b=candidate.observe(b,(),('a','b'),{},z)
        self.assertEqual(baseline.predict(a,(),('a','b'),{}),candidate.predict(b,(),('a','b'),{}))

    def test_weighted_posterior_matches_independent_change_partition_enumeration(self):
        sequence=(2,0,2,1)
        library=((.7,.2,.1),(.1,.6,.3))
        orders=tuple(itertools.permutations(range(3)))
        templates=tuple(tuple(row[i] for i in perm) for row in library for perm in orders)
        hypotheses=(*library,*templates)
        prior=np.array([.5/len(library)]*len(library)+[.25/len(templates)]*len(templates)+[.25])
        def evidence_prediction(sequence):
            counts=np.full(3,.5);likelihood=1.
            for z in sequence:
                likelihood*=counts[z]/sum(counts);counts[z]+=1.
            weight=prior*np.array([np.prod([row[z] for z in sequence]) for row in hypotheses]+[likelihood])
            return weight.sum(),(weight@np.array([*hypotheses,tuple(counts/sum(counts))]))/weight.sum()
        initial=evidence_prediction(())[1];total=0.;weighted=np.zeros(3)
        for hazard in HAZARDS:
            for cuts in itertools.product((0,1),repeat=3):
                probability=1.;start=0
                for index,cut in enumerate(cuts,1):
                    probability*=hazard if cut else 1-hazard
                    if cut:
                        probability*=evidence_prediction(sequence[start:index])[0];start=index
                likelihood,p=evidence_prediction(sequence[start:]);probability*=likelihood
                weighted+=probability*(hazard*initial+(1-hazard)*p);total+=probability
        observed,_,_=transfer_prediction(sequence,(library,)*len(sequence),library,3)
        np.testing.assert_allclose(observed,weighted/total,atol=1e-14)


if __name__=='__main__':unittest.main()
