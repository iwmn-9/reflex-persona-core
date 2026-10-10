import copy,json,unittest
import numpy as np
from reflex.paired_value import fit,predict,calibrate,arbitrate


class PairedValueTests(unittest.TestCase):
    def cases(self):
        return [(np.array([[s,0],[s,1.]]),np.array([[.3-.1*s,.4],[.3+.1*s,.7]])) for s in (-1.,-.5,.5,1.)]

    def test_matched_effect_can_change_sign_with_context(self):
        model=fit(self.cases(),alpha=.01)
        for s in (-.8,.8):
            p,support=predict(model,[[s,0],[s,1]])
            self.assertTrue(support);self.assertGreater((p[1,0]-p[0,0])*s,0)
            self.assertGreater(p[1,1]-p[0,1],.25)

    def test_constant_case_reward_does_not_create_action_advantage(self):
        cases=[(x,np.full_like(y,.2 if i%2 else .8)) for i,(x,y) in enumerate(self.cases())]
        p,_=predict(fit(cases),[[.4,0],[.4,1]])
        np.testing.assert_array_equal(p,np.zeros((2,2)))

    def test_case_level_nuisance_and_candidate_order_do_not_change_difference(self):
        original=self.cases();shifted=[(x,y+.15) for x,y in original]
        a=fit(original);b=fit(shifted)
        np.testing.assert_allclose(a['coef'],b['coef'],atol=1e-12)
        x=np.array([[.6,0],[.6,1.]])
        p,_=predict(a,x);q,_=predict(a,x[::-1]);np.testing.assert_allclose(p,q[::-1],atol=1e-12)

    def test_factual_singleton_and_malformed_data_are_rejected(self):
        for cases in ([],[([[1]],[[.5]])],[([[1],[2]],[[0],[2]])],[([[1],[np.nan]],[[0],[1]])],[([[1],[2]],[[],[]])]):
            with self.assertRaises(ValueError):fit(cases)
        model=fit(self.cases());model['scale'][0]=0
        with self.assertRaises(ValueError):predict(model,[[.5,0],[.5,1]])

    def test_json_roundtrip_and_support_boundary(self):
        model=fit(self.cases());restored=json.loads(json.dumps(model));x=[[.6,0],[.6,1]]
        p,a=predict(model,x);q,b=predict(restored,x);np.testing.assert_array_equal(p,q);self.assertEqual(a,b)
        _,support=predict(restored,[[100,0],[100,1]]);self.assertFalse(support)

    def test_calibration_measures_heldout_pair_errors(self):
        model=fit(self.cases(),alpha=.01);before=copy.deepcopy(model)
        bound=calibrate(model,self.cases());self.assertEqual(model,before)
        self.assertEqual(len(bound),2);self.assertTrue(all(0<=b<.02 for b in bound))
        with self.assertRaises(ValueError):calibrate(model,[])

    def test_incumbent_changes_only_for_supported_clear_purpose_gain(self):
        model=fit(self.cases(),alpha=.01)
        chosen,g=arbitrate(model,[[.8,0],[.8,1]],0,[0.,0.],max_regret=0)
        self.assertEqual(chosen,1);json.dumps(g)
        chosen,g=arbitrate(model,[[.8,0],[.8,1]],0,[1.,0.],max_regret=0)
        self.assertEqual(chosen,0)
        chosen,g=arbitrate(model,[[100,0],[100,1]],0,[0.,0.],max_regret=0)
        self.assertEqual(chosen,0);self.assertFalse(g['supported'])

    def test_weights_and_incumbent_contract(self):
        for weights in ([1],[-1]*4,[np.nan]*4):
            with self.assertRaises(ValueError):fit(self.cases(),case_weights=weights)
        model=fit(self.cases())
        for index,errors in ((-1,[0.,0.]),(0,[-1.,0.]),(0,[0.])):
            with self.assertRaises(ValueError):arbitrate(model,[[.8,0],[.8,1]],index,errors)


if __name__=='__main__':unittest.main()
