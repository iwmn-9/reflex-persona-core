"""Run a study from detached source bytes while the main checkout evolves."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def dispatch(root,script,arguments):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    base=Path(__file__).resolve().parents[1];snapshot=root/'_source'
    if snapshot.exists():raise FileExistsError('fresh experiment source directory required')
    files=list((base/'reflex').glob('*.py'))+list((base/'tools').glob('*.py'))
    files.append(base/'reflex/lab_viewer.html')
    fingerprints={}
    for p in files:
        relative=p.relative_to(base);target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,target);fingerprints[relative.as_posix()]=hashlib.sha256(target.read_bytes()).hexdigest()
    (root/'source_snapshot.json').write_text(json.dumps(fingerprints,indent=2)+'\n',encoding='utf-8')
    subprocess.run([sys.executable,str(snapshot/'tools'/script),'--root',str(root),'--frozen',*arguments],check=True)
