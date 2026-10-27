"""Run the documented host against an isolated installed package, not repository imports."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

LAUNCH = '''import sys,runpy
sys.path.insert(0,sys.argv[1])
class NoResearch:
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='tools' or fullname.startswith('tools.'):
   raise ImportError('consumer may not import research tools')
sys.meta_path.insert(0,NoResearch())
import reflex
from pathlib import Path
assert Path(reflex.__file__).resolve().is_relative_to(Path(sys.argv[1]))
sys.argv=sys.argv[2:]
runpy.run_path(sys.argv[0],run_name='__main__')
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('installed', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    installed = args.installed.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    example = Path(__file__).resolve().parents[1]/'examples/stable_host.py'
    rows = []
    with tempfile.TemporaryDirectory(prefix='reflex-stable-consumer-') as folder:
        work = Path(folder)
        copied = work/'stable_host.py'
        shutil.copyfile(example, copied)
        for command in (['--save', 'full.json'], ['--save', 'prefix.json', '--stop-after', '3'],
                        ['--resume', 'prefix.json', '--save', 'resumed.json'],
                        ['--resume', 'full.json', '--save', 'terminal.json']):
            result = subprocess.run([sys.executable, '-I', '-c', LAUNCH, str(installed), str(copied), *command],
                                    cwd=work, text=True, encoding='utf8', capture_output=True)
            if result.returncode:
                (args.output/'failure.txt').write_text(result.stdout+result.stderr, encoding='utf8')
                raise RuntimeError(result.stderr)
            rows.append(json.loads(result.stdout))
        read = lambda name: json.loads((work/name).read_bytes())
        full = read('full.json')
        assert full == read('resumed.json') == read('terminal.json')
        assert rows[0]['complete'] and not rows[1]['complete'] and rows[1]['tick'] == 3
        (args.output/'final.json').write_text(json.dumps(full, sort_keys=True)+'\n', encoding='utf8')
    summary = dict(complete=True, isolated_from_repository=True, research_imports_blocked=True,
                   example_sha256=hashlib.sha256(example.read_bytes()).hexdigest(),
                   verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   python=sys.version, new_quality_trials=False, checks=rows,
                   host_world_rng_owner_and_trace_exact=True)
    (args.output/'verification.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
