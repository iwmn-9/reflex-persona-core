import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.archive_adverse_study import registered_sources
from tools.owned_checkpoint import sha


class RegisteredSourceArchiveTests(unittest.TestCase):
    def test_registered_tests_and_root_files_are_retained_with_exact_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            files={'.gitattributes':b'* -text\n','reflex/core.py':b'value=1\n','tests/test_contract.py':b'assert True\n',
                   '.github/workflows/test.yml':b'name: verify\n','pyproject.toml':b'[project]\nname="fixture"\n'}
            for name,data in files.items():
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            (root/'unregistered.txt').write_bytes(b'not part of the contract\n')
            subprocess.run(['git','init','--quiet'],cwd=root,check=True)
            subprocess.run(['git','-c','core.autocrlf=false','add','.'],cwd=root,check=True)
            subprocess.run(['git','-c','user.name=Test','-c','user.email=test@example.invalid',
                            'commit','--quiet','-m','frozen fixture'],cwd=root,check=True)
            expected={name:sha(data) for name,data in files.items()}
            self.assertEqual(registered_sources('HEAD',expected,root),files)
            corrupted=dict(expected);corrupted['tests/test_contract.py']='0'*64
            with self.assertRaises(AssertionError):registered_sources('HEAD',corrupted,root)
            missing=dict(expected);missing['tests/missing.py']=sha(b'missing')
            with self.assertRaises(KeyError):registered_sources('HEAD',missing,root)


if __name__=='__main__':unittest.main()
