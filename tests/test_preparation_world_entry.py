import copy
import tempfile
import unittest
from pathlib import Path

from reflex.core import character_key
from reflex.laboratory import PROFILES
from tools.continuing_games import ContinuingGame
from tools.run_adverse_recovery import decide, run
from tools.adverse_segment import audit


class PreparationWorldEntryTests(unittest.TestCase):
    def test_same_fixed_horizon_path_retains_original_proposals(self):
        for family in ('auction','resources','combat','gambling'):
            with self.subTest(family=family):
                game=ContinuingGame(family,PROFILES[1],turns=3)
                world=game.start();actor=game.actor(game.context(world),learning='segmented')
                a,stable=decide(copy.deepcopy(actor),game,world,'stable')
                b,new=decide(copy.deepcopy(actor),game,world,'expected')
                self.assertEqual(character_key(a['context']),character_key(b['context']))
                self.assertEqual(new['horizon_consensus']['minimum_horizon'],32)
                self.assertEqual(new['horizon_consensus']['maximum_horizon'],32)
                meta=new['forecast'].audit['continuation_portfolio']
                self.assertEqual(meta['seeds'],[701,702,703])
                self.assertNotIn('tail_purpose_reflex',meta)
                self.assertEqual(meta['retry_until_gain']['kind'],'public-purpose-above-root-expected-ordinary')
                for name in stable['paths']:
                    self.assertEqual(stable['paths'][name],new['paths'][name])
                    self.assertEqual(stable['modeled'][name],new['modeled'][name])

    def test_learning_resume_and_wrong_arm_rejection(self):
        spec=dict(id='entry-resume',mechanism='world',game='combat',profile=1,
                  seed=601,regime='outside',turns=3)
        registration=dict(plan_sha256='test-plan',source_sha256='test-source')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            full=run(root/'full',spec,'expected',registration=registration)
            self.assertIsNone(run(root/'split',spec,'expected',max_decisions=1,registration=registration))
            self.assertFalse(audit(root/'split',spec,'expected',registration)['complete'])
            with self.assertRaises(ValueError):run(root/'split',spec,'stable',registration=registration)
            resumed=run(root/'split',spec,'expected',registration=registration)
            for key in ('trace','final','final_owner','final_rng','configuration','checks'):
                self.assertEqual(full[key],resumed[key],key)
            self.assertTrue(audit(root/'split',spec,'expected',registration)['complete'])


if __name__=='__main__':unittest.main()
