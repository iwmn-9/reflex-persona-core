import copy
from dataclasses import asdict
import unittest
import numpy as np
from tools.probe_outcome_value import fit, predict, features
from reflex.goofspiel import Position, make_context
from reflex.laboratory import PROFILES


class OutcomeValueProbeTests(unittest.TestCase):
    def test_episode_weights_also_control_normalization(self):
        x=np.array([[0.,1.],[1.,0.],[.5,.2]]);y=np.array([0.,1.,.6]);w=np.ones(3)
        a=fit(x,y,w)
        b=fit(np.repeat(x,3,axis=0),np.repeat(y,3),np.repeat(w/3,3))
        np.testing.assert_allclose(predict(a,x),predict(b,x),atol=1e-13)

    def test_fitted_model_is_reproducible_and_bounded_for_extrapolation(self):
        x=np.linspace(0,1,40)[:,None];y=x[:,0];w=np.ones(40)
        a=fit(x,y,w);b=fit(x,y,w)
        for key in a:np.testing.assert_array_equal(a[key],b[key])
        p=predict(a,np.array([[-20.],[20.]]));self.assertTrue(np.all((p>=.02)&(p<=.98)))
        self.assertGreater(predict(a,np.array([[.8]]))[0],predict(a,np.array([[.2]]))[0])

    def test_malformed_or_unobserved_training_data_are_rejected(self):
        for x,y,w in (([[np.nan]],[0],[1]),([[1]],[2],[1]),([[1]],[0],[0]),([[1],[2]],[0],[1])):
            with self.assertRaises(ValueError):fit(x,y,w)

    def test_features_ignore_rival_identity_hidden_future_and_dictionary_order(self):
        s=Position.start(4,3);b=asdict(s);c=make_context(s,0,PROFILES[0],'win_share',0,0,'probe')
        a=features('goofspiel',b,0,'BID:2',c)
        alternate=copy.deepcopy(c);alternate['personality']=dict(reversed(list(c['personality'].items())))
        b['hidden_future']='not an input';b['rival_profiles']=['imaginary']*4
        np.testing.assert_array_equal(a,features('goofspiel',b,0,'BID:2',alternate))


if __name__=='__main__':unittest.main()
