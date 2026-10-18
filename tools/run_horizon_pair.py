"""Run two registered horizons on one host, with independent frozen owners.

This changes execution placement only. Each side retains its own immutable
source, world RNG, checkpoint, trajectory and original-platform audit. A failed
or partial side is never counted as a completed pair.
"""
import argparse,json,os,platform,shutil,subprocess,sys,uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.laboratory import PROFILES


def run_pair(root,jobs,stage,recovery_limit=16):
    if (not isinstance(jobs,list) or len(jobs)!=2 or
        any(not isinstance(j,list) or len(j)!=8 for j in jobs)):
        raise ValueError('two explicit horizon jobs required')
    left,right=jobs
    if left[:6]!=right[:6] or left[7:]!=right[7:] or left[6]==right[6]:
        raise ValueError('same world, person and schedule with two different horizons required')
    if any(type(j[6]) is not int or not 1<=j[6]<=127 for j in jobs):
        raise ValueError('bounded nonempty future required')
    if type(left[1]) is not int or not 0<=left[1]<len(PROFILES):
        raise ValueError('registered persona required')
    if type(recovery_limit) is not int or not 1<=recovery_limit<=256:
        raise ValueError('bounded common recovery budget required')
    root=Path(root).resolve()
    if root.exists():raise FileExistsError('fresh pair directory required; resume each frozen execution separately')
    root.mkdir(parents=True)
    header=dict(version='same-host-horizon-pair-v1',pair_id=uuid.uuid4().hex,jobs=jobs,
        profile=PROFILES[left[1]]['id'],stage=stage,recovery_limit=recovery_limit,
        python=sys.version,system=platform.system(),machine=platform.machine(),cpu_count=os.cpu_count(),
        semantics='Independent subprocesses on one host; no sharing of actor state, RNG, or model caches.')
    manifest=root/'pair_execution.json';manifest.write_text(json.dumps(header,indent=2)+'\n',encoding='utf8')
    children=[]
    for job in jobs:
        child=root/f'h{job[6]+1}';child.mkdir();shutil.copyfile(manifest,child/manifest.name)
        children.append(child)
    script=Path(__file__).with_name('run_horizon_study.py')
    def execute(item):
        child,job=item
        args=[sys.executable,str(script),'--root',str(child),'--job',json.dumps(job),
              '--stage',stage,'--recovery-limit',str(recovery_limit)]
        return subprocess.run(args,check=False).returncode
    with ThreadPoolExecutor(max_workers=2) as pool:
        codes=list(pool.map(execute,zip(children,jobs)))
    header['return_codes']=codes;header['complete']=codes==[0,0]
    if header['complete']:
        snapshots=[json.loads((p/'source_snapshot.json').read_text()) for p in children]
        assert snapshots[0]==snapshots[1],'paired source snapshots differ'
        for child,job in zip(children,jobs):
            row=json.loads((child/'trajectories.jsonl').read_text())
            assert row['profile']==header['profile'] and row['tail_steps']==job[6]
            assert json.loads((child/'audit.json').read_text())['complete']
        header['same_source']=True
    manifest.write_text(json.dumps(header,indent=2)+'\n',encoding='utf8')
    for child in children:shutil.copyfile(manifest,child/manifest.name)
    if not header['complete']:raise RuntimeError('incomplete pair retained with per-side checkpoints')
    return header


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--job',required=True)
    p.add_argument('--partner-job',required=True);p.add_argument('--stage',required=True)
    p.add_argument('--recovery-limit',type=int,default=16);a=p.parse_args()
    print(json.dumps(run_pair(a.root,[json.loads(a.job),json.loads(a.partner_job)],a.stage,a.recovery_limit)),flush=True)
