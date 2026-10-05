"""Preregistration uses repository paths independent of the host separator."""
from pathlib import PurePosixPath, PureWindowsPath
import unittest
from unittest.mock import patch
from reflex import goal_opponent_experiment, goal_gameplay_experiment


class ManifestPathTests(unittest.TestCase):
    def test_both_harnesses_serialize_posix_keys_on_both_path_flavours(self):
        for module in (goal_opponent_experiment, goal_gameplay_experiment):
            design = module.DESIGN.relative_to(module.ROOT).as_posix()
            results = []
            for cls, base in ((PurePosixPath, '/repo'), (PureWindowsPath, 'C:/repo')):
                with self.subTest(module=module.__name__, path_flavour=cls.__name__):
                    root = cls(base)
                    with patch.object(module, 'ROOT', root), patch.object(module, 'DESIGN', root/design), \
                         patch.object(module, 'sha', return_value='same-file-hash'), \
                         patch.object(cls, 'glob', return_value=[root/'tests/test_fixture.py'], create=True):
                        result = module.frozen_files()
                    self.assertIn(design, result)
                    self.assertIn('tests/test_fixture.py', result)
                    self.assertTrue(all('\\' not in key and ':' not in key for key in result))
                    results.append(result)
            self.assertEqual(results[0], results[1])


if __name__ == '__main__': unittest.main()
