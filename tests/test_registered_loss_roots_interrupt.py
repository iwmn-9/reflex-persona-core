"""A background-launched coordinator must still accept cooperative shutdown."""
from pathlib import Path
import subprocess
import sys
import unittest


class RegisteredRootInterruptionTests(unittest.TestCase):
    def test_ignored_launcher_signal_does_not_disable_cooperative_handoff(self):
        driver = Path(__file__).resolve().parents[1] / 'tools/run_registered_loss_roots.py'
        program = '''
import contextlib, io, runpy, signal, sys
signal.signal(signal.SIGINT, signal.SIG_IGN)
path = sys.argv[1]
sys.argv = [path, '--help']
with contextlib.redirect_stdout(io.StringIO()):
    try:
        runpy.run_path(path, run_name='__main__')
    except SystemExit as error:
        assert error.code == 0
try:
    signal.raise_signal(signal.SIGINT)
except KeyboardInterrupt:
    print('cooperative interruption received')
else:
    raise AssertionError('launcher silently disabled interruption')
'''
        result = subprocess.run([sys.executable, '-c', program, str(driver)],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'cooperative interruption received')


if __name__ == '__main__':
    unittest.main()
