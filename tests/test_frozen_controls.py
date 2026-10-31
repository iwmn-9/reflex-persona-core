import json
from pathlib import Path
import tempfile
import unittest

from tools.frozen_controls import load_controls, materialize_controls

ROOT = Path(__file__).resolve().parents[1]


def prospect():
    plan = json.loads((ROOT/'evidence/long_horizon/adverse_gain_tail/plan.json').read_bytes())
    receipt = json.loads((ROOT/'evidence/relative_gain/development_control_bank.json').read_bytes())
    plan['frozen_controls'] = dict(arm='stable',
        archive='evidence/relative_gain/development_control_bank.zip',
        archive_sha256=receipt['bank_sha256'], original_proof_sha256=receipt['original_proof_sha256'],
        source_ref=receipt['original_source'], registration=receipt['controls'][0]['original_registration'],
        runtime=receipt['native_runtime'])
    return plan


class FrozenControlTests(unittest.TestCase):
    def test_whole_original_cohort_and_bytes_are_preserved_on_reentry(self):
        plan = prospect()
        bank = load_controls(plan)
        self.assertEqual(len(bank['files']), 30)
        self.assertNotEqual(bank['registration']['plan_sha256'],
            plan['predecessor_plan_sha256'])
        with tempfile.TemporaryDirectory() as directory:
            materialize_controls(plan, directory)
            materialize_controls(plan, directory)
            for (identity, name), raw in bank['files'].items():
                self.assertEqual((Path(directory)/f'adverse-{identity}-stable'/name).read_bytes(),raw)
            target = Path(directory)/f'adverse-{plan["specs"][0]["id"]}-stable/result.json'
            target.write_bytes(b'{}')
            with self.assertRaises(AssertionError):
                materialize_controls(plan, directory)

    def test_changed_conditions_provenance_precision_or_subset_are_refused(self):
        variants = []
        for key in ('archive_sha256','original_proof_sha256','source_ref'):
            plan=prospect();plan['frozen_controls'][key]='wrong';variants.append(plan)
        plan=prospect();plan['specs']=plan['specs'][:-1];variants.append(plan)
        plan=prospect();plan['specs'][0]['seed']+=1;variants.append(plan)
        plan=prospect();plan['frozen_controls']['registration']['source_sha256']='wrong';variants.append(plan)
        plan=prospect();plan['arithmetic']['OPENBLAS_CORETYPE']='wrong';variants.append(plan)
        for plan in variants:
            with self.subTest(plan=plan['frozen_controls']):
                with self.assertRaises(AssertionError):
                    load_controls(plan)

    def test_existing_same_source_studies_do_not_implicitly_reuse_controls(self):
        plan=prospect();del plan['frozen_controls']
        self.assertIsNone(load_controls(plan))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                materialize_controls(plan,directory)
            self.assertEqual(list(Path(directory).iterdir()),[])


if __name__=='__main__':unittest.main()
