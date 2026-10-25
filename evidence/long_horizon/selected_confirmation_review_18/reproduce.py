"""Reproduce the factual partial audit without running new planner trajectories."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,help='New temporary reproduction directory')
    args=parser.parse_args();output=Path(args.output).resolve()
    folder=Path(__file__).resolve().parent;repo=folder.parents[2]
    verification=json.loads((folder/'verification.json').read_bytes())
    digest=lambda raw:hashlib.sha256(raw).hexdigest()
    for name,expected in verification['artifacts'].items():
        assert digest((folder/name).read_bytes())==expected,name
    proof=json.loads((folder/'replay_verification.json').read_bytes())
    source=repo/proof['source_zip']
    assert digest(source.read_bytes())==proof['source_zip_sha256']
    output.mkdir(parents=True,exist_ok=False)
    (output/'plan.json').write_bytes((folder/'plan.json').read_bytes())
    for archive,target in ((folder/'inputs.zip',output/'trial'),(source,output/'trial/_source')):
        with zipfile.ZipFile(archive) as bundle:
            for name in bundle.namelist():
                assert (target/name).resolve().is_relative_to(target.resolve()),name
            bundle.extractall(target)
    subprocess.run([sys.executable,'-I',str(folder/'review.py'),'--root',str(output)],check=True)
    actual=json.loads((output/'review.json').read_bytes())
    expected=json.loads(gzip.decompress((folder/'review.json.gz').read_bytes()))
    assert {k:v for k,v in actual.items() if k!='at'}=={k:v for k,v in expected.items() if k!='at'}
    print(json.dumps(dict(complete=True,actual_decisions=actual['replayed_actual_decisions'],
        whole_trial_complete=False,adoption=False,exact_reproduction=True)),flush=True)


if __name__=='__main__':main()
