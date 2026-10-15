import unittest
from reflex.core import digest,FEATURES
from reflex.examples import context,action,effect
from reflex.decision_loop import DecisionLoop,Request
from reflex.loop_predictor import CategoricalReader,SupportedCategoricalReader,ValidatedSupportedCategoricalReader,GuardedSupportedCategoricalReader
from reflex.judgment import Binding,ReadControl
from reflex.planning import vector


class ResponseComparisonTests(unittest.TestCase):
    def fixture(self,kind,cap):
        c=context('comparison',[action('work',effect(.2))]);c['facts']['opportunity']='one'
        args=(c['scope'],('a','b'),lambda cc:{'a':{'A':.9,'B':.1},'b':{'A':.1,'B':.9}},lambda *args:effect(.2))
        reader=kind(*args,known_conditionals=True,classes=('one','two'),
            opportunity_for=lambda cc:cc['facts']['opportunity'],support_key='public-space-v1')
        a=DecisionLoop(c,predictor=reader,read_control=ReadControl(max_nodes=cap))
        return c,a

    def test_wrapped_predictors_compare_live_prior_and_candidate_even_without_reading(self):
        for cls in (SupportedCategoricalReader,ValidatedSupportedCategoricalReader,GuardedSupportedCategoricalReader):
            for cap in (0,64):
                c,a=self.fixture(cls,cap)
                for tick in range(12):
                    c['tick']=tick;c['facts']['opportunity']='one' if tick<8 else 'two'
                    prior,candidate=a.predictor.forecasts(c)
                    r=a.decide(Request(c,{'work':Binding('work','same',FEATURES)},{'work':()},threatened=True),False)
                    trial=a.pending['response_trial'];self.assertIsNotNone(trial)
                    self.assertEqual(trial['prior'],prior);self.assertEqual(trial['candidate'],candidate)
                    if a.pending['read'] is not None:self.assertEqual(trial['key'],a.pending['read']['key'])
                    event='A' if tick<6 else 'B'
                    result=a.observe(r['ticket'],vector(effect(.2)),event={'revealed_action':event})
                    expected=sum((prior[k]-(k==event))**2-(candidate[k]-(k==event))**2 for k in prior)/len(prior)
                    if prior!=candidate:self.assertAlmostEqual(result['reading']['gain'],expected)
                    else:self.assertFalse(result['reading']['scored'])

    def test_unknown_conditional_readers_do_not_claim_known_response_comparison(self):
        c,a=self.fixture(SupportedCategoricalReader,0)
        for bank in a.predictor.support._banks.values():bank.known_conditionals=False
        r=a.decide(Request(c,{'work':Binding('work','same',FEATURES)},{'work':()}),False)
        self.assertIsNone(a.pending['response_trial']);a.abandon(r['ticket'])

    def test_old_uniform_gate_is_not_reused_as_candidate_validation_on_restore(self):
        for cls in (SupportedCategoricalReader,ValidatedSupportedCategoricalReader,GuardedSupportedCategoricalReader):
            c,a=self.fixture(cls,0)
            key=a.predictor.support.select('one').key
            if cls is ValidatedSupportedCategoricalReader:key+=':local'
            if cls is GuardedSupportedCategoricalReader:key+=':shared-fallback'
            old='response:'+digest(key)
            for _ in range(4):a.evidence._update(old,.5,0.)
            a=DecisionLoop.from_record(c,a.record(),predictor=a.predictor)
            r=a.decide(Request(c,{'work':Binding('work','same',FEATURES)},{'work':()}),False)
            new=a.pending['response_trial']['key']
            self.assertNotEqual(new,old);self.assertTrue(a.evidence.accepted(old));self.assertFalse(a.evidence.accepted(new))
            a.abandon(r['ticket'])


if __name__=='__main__':unittest.main()
