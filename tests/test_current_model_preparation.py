import json
from pathlib import Path
import tempfile
import unittest

from tools.run_continuation_portfolio import execution
from tools.run_current_model_preparation import audit


class CurrentPreparationTests(unittest.TestCase):
    def spec(self):
        return dict(id='chain-test',mechanism='chain',length=2,payout=7.,profile=2,
                    turns=3,changes=[[2,2.]],controller=dict(horizon_consensus=True,minimum_horizon=32,
                    coherent_cumulative_mean=True,purpose_estimator='conditional',tail_recovery=2,recovery_cache=64))

    def test_candidate_full_resumed_owner_and_factual_change_replay(self):
        spec=self.spec()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            full=execution(root/'full',spec,True,reader_selection='current_model')
            self.assertIsNone(execution(root/'split',spec,True,reader_selection='current_model',max_decisions=1))
            with self.assertRaises(ValueError):execution(root/'split',spec,True)
            split=execution(root/'split',spec,True,reader_selection='current_model')
            for key in ('trace','final','score','checks','configuration'):
                self.assertEqual(full[key],split[key],key)
            self.assertTrue(audit(root/'split',spec,'current_model')['complete'])
            self.assertEqual(split['final']['payout'],2.)

    def test_default_reference_and_explicit_advice_remain_identical(self):
        spec=self.spec()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            default=execution(root/'default',spec,True)
            explicit=execution(root/'explicit',spec,True,reader_selection='advice')
            self.assertEqual(default['trace'],explicit['trace'])
            self.assertEqual(default['configuration'],explicit['configuration'])
            self.assertTrue(audit(root/'explicit',spec,'advice')['complete'])
            with self.assertRaises(ValueError):execution(root/'illegal',spec,False,reader_selection='current_model')


if __name__=='__main__':unittest.main()
