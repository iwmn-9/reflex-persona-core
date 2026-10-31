import copy
import unittest
from unittest.mock import patch

import numpy as np

from reflex.core import FEATURES, NEEDS, VALUES, TRAITS, number, _effect_values, compile_batch
from reflex.examples import context, action, effect
from reflex.decision_loop import EvidenceGate, THRESHOLDS, _cdf_signature
from reflex.planning import vector, from_vector


def original_signature(rows, names):
    samples=[(r['p'],vector(r)) for r in rows]
    return tuple(float(sum(p*(v[FEATURES.index(f)]<=t) for p,v in samples))
                 for f in names for t in THRESHOLDS)


class ExactEvaluationRuntimeTests(unittest.TestCase):
    def test_sparse_effects_keep_supplied_value_validation_and_frozen_batch_bytes(self):
        rng=np.random.default_rng(20261102)
        def original(mapping,names):return [number(mapping.get(key,0),-1,1) for key in names]
        for tick in range(40):
            def fields(names):return {k:float(rng.uniform(-1,1)) for k in names if rng.random()<.3}
            c=context('sparse-effects',[action(str(j),effect(float(rng.uniform(-1,1)),
                needs=fields(NEEDS),values=fields(VALUES),style=fields(TRAITS))) for j in range(5)])
            with patch('reflex.core._effect_values',original):expected=compile_batch([c])
            actual=compile_batch([c])
            for name in actual.__dataclass_fields__:
                a,b=getattr(actual,name),getattr(expected,name)
                if isinstance(a,np.ndarray):
                    self.assertEqual(a.tobytes(),b.tobytes(),name)
                    self.assertFalse(a.flags.writeable)
                else:self.assertEqual(a,b,name)
        for invalid in (True,None,2.,float('inf'),float('nan')):
            with self.assertRaises(ValueError):_effect_values({'growth':invalid},NEEDS)
        class Mapping(dict):
            def get(self,key,default):return -.5
        self.assertEqual(_effect_values(Mapping(),NEEDS),[-.5]*len(NEEDS))

    def test_scalar_validation_keeps_values_and_rejection_contract(self):
        def original(value,low,high):
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value):
                raise ValueError('finite numeric value required')
            if not low<=value<=high:
                raise ValueError(f'value outside [{low}, {high}]')
            return float(value)
        values=[False,True,None,'0',0,1,-1,2**100,0.,-0.,1.,-1.,float('nan'),float('inf'),-float('inf'),
                float(np.nextafter(0.,1.)),float(np.nextafter(1.,2.)),np.float32(.5),np.float64(.5)]
        values += list(map(float,np.random.default_rng(20261031).uniform(-2,2,1000)))
        for value in values:
            for low,high in ((-1,1),(0,1)):
                try: expected=original(value,low,high)
                except Exception as error:
                    with self.assertRaises(type(error)) as result:number(value,low,high)
                    self.assertEqual(str(result.exception),str(error))
                else:self.assertEqual(number(value,low,high).hex(),expected.hex())

    def test_signature_preserves_every_bit_at_threshold_boundaries(self):
        rng=np.random.default_rng(20261031)
        edges=np.concatenate((THRESHOLDS,np.nextafter(THRESHOLDS,-np.inf),
                              np.nextafter(THRESHOLDS,np.inf),[-1.,1.,0.,-0.]))
        for i in range(600):
            n=int(rng.integers(1,9)); probabilities=rng.dirichlet(np.ones(n))
            values=rng.choice(edges,size=(n,len(FEATURES))) if i%2 else rng.uniform(-1,1,(n,len(FEATURES)))
            rows=[from_vector(v,float(p)) for v,p in zip(values,probabilities)]
            names=FEATURES if i%3 else tuple(rng.choice(FEATURES,size=7,replace=True))
            actual=_cdf_signature(rows,names); expected=original_signature(rows,names)
            self.assertEqual(np.asarray(actual).tobytes(),np.asarray(expected).tobytes())

    def test_nonstandard_probability_types_preserve_reference_semantics(self):
        for p in (0,1,np.float32(.5),np.float64(.5)):
            rows=[from_vector(np.zeros(len(FEATURES)),.5)]
            rows[0]['p']=p
            self.assertEqual(_cdf_signature(rows,FEATURES),original_signature(rows,FEATURES))
        self.assertEqual(_cdf_signature([],FEATURES),original_signature([],FEATURES))

    def test_adoption_revocation_and_history_match_over_long_feedback(self):
        rng=np.random.default_rng(20261101)
        old,new=EvidenceGate(),EvidenceGate()
        for tick in range(256):
            prior=[from_vector(rng.uniform(-1,1,len(FEATURES)),.5) for _ in range(2)]
            candidate=copy.deepcopy(prior) if tick%5==0 else [from_vector(rng.uniform(-1,1,len(FEATURES)),.5) for _ in range(2)]
            observed=rng.uniform(-1,1,len(FEATURES)); key=str(tick%3)
            with patch('reflex.decision_loop._cdf_signature',original_signature):
                expected=old.compare(key,prior,candidate,observed,FEATURES)
            actual=new.compare(key,prior,candidate,observed,FEATURES)
            self.assertEqual(actual,expected)
            self.assertEqual(new.entries,old.entries)


if __name__=='__main__':unittest.main()
