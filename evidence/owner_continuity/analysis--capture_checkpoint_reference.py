import hashlib,json,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
r=Path.cwd();sys.path.insert(0,str(r))
from tools.run_owned_value import run
p=r/'results/owner_checkpoint_equivalence';p.mkdir(exist_ok=False)
manifest={f.as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in [*Path('reflex').glob('*.py'),Path('tools/run_owned_value.py'),Path('tools/continuing_games.py')]}
(p/'original_manifest.json').write_text(json.dumps(manifest,indent=2))
jobs=[((g,2,41901,'shift-return',8,1 if arm=='receding_recovery' else 7,3,0),arm) for g in ('auction','resources','combat','gambling') for arm in ('endpoint_posterior','receding_recovery')]
jobs += [(('resources',0,41901,'shift-return',26,1,3,0),'reflex_segmented')]
(p/'registration.json').write_text(json.dumps(jobs,indent=2))
for index,(job,arm) in enumerate(jobs):
 row=run(job,arm);(p/f'original-{index}.json').write_text(json.dumps(row))
 print(json.dumps(dict(index=index,game=job[0],arm=arm,seconds=row['seconds'])),flush=True)
