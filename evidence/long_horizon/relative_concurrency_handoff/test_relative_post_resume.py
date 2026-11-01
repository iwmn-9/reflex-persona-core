import unittest
from finish_relative_post_resumed import validate_regression, validate_finisher

class ResumeGuards(unittest.TestCase):
    def test_completed_regression(self):
        ok=dict(complete=True,passed=True,source_commit='493a685ab2837c999d13d832dc25a51068845fb5',failures=0,errors=0,unexpected_successes=[])
        validate_regression(ok)
        for change in (dict(complete=False),dict(passed=False),dict(source_commit='other'),dict(failures=1),dict(errors=1),dict(unexpected_successes=['bad'])):
            with self.subTest(change=change),self.assertRaises(ValueError):validate_regression(dict(ok,**change))
    def test_whole_finisher(self):
        ok=dict(complete=True,state='complete_requires_scientific_review',manifest_sha256='exact')
        validate_finisher(ok,'exact')
        for change in (dict(complete=False),dict(state='waiting'),dict(state='failed_requires_inspection'),dict(manifest_sha256='other')):
            with self.subTest(change=change),self.assertRaises(ValueError):validate_finisher(dict(ok,**change),'exact')

if __name__=='__main__':unittest.main()
