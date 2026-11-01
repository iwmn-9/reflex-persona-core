"""Import a completed exact-replay recovery, retaining the original evidence."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.owned_checkpoint import encoded,sha
from tools.review_adverse_recovery import summarize


def accept(root,recovered):
    root=Path(root);source=Path(recovered)
    recovery=json.loads((source/'runtime_recovery/recovery.json').read_bytes())
    status=json.loads((source/'status.json').read_bytes())
    saved=json.loads((source/'checkpoint.json').read_bytes())['payload']
    assert recovery['state']=='worker_finished' and recovery['worker_returncode']==0
    assert recovery['retained_prefix'] and recovery['original_worker_status']==status
    assert status['state']=='complete' and status['complete']
    plan=json.loads((ROOT/'evidence/long_horizon/adverse_recovery/plan.json').read_bytes())
    spec=next(s for s in plan['specs'] if s['id']==status['case'])
    assert status['actual_steps']==recovery['final_steps']==spec['turns']
    probes=json.loads((source/'runtime_recovery/attempts.json').read_bytes())
    matching=[p for p in probes if p['exact']]
    assert len(matching)==1 and matching[0]['returncode']==0
    match=matching[0]
    assert recovery['selected_mode']==dict(NPY_DISABLE_CPU_FEATURES=match['numpy_disabled'],OPENBLAS_CORETYPE=match['blas_core'])
    assert match['audit']['actual_steps']==recovery['original_steps']
    assert match['audit']['checkpoint_sha256']==recovery['original_checkpoint_sha256']
    name=f"adverse-{spec['id']}-{status['arm']}"
    dest=root/'latest'/name
    original=(dest/'checkpoint.json').read_bytes()
    assert sha(original)==recovery['original_checkpoint_sha256'],'original archive changed or recovery already imported'
    old=json.loads(original)['payload']
    assert saved['contract']==old['contract'],'source/configuration/runtime contract changed'
    assert saved['trace'][:len(old['trace'])]==old['trace'],'actual experience replaced'
    assert len(old['trace'])==recovery['original_steps']
    assert match['audit']['registration']==status['registration']
    # Validate complete result/envelope/physical steps before modifying latest.
    staging=root/'recovery_validation'/name
    if staging.exists():raise FileExistsError('fresh validation directory required')
    shutil.copytree(source,staging)
    summarize(staging.parent,dict(plan,specs=[spec]))
    receipt=dict(case=spec['id'],arm=status['arm'],recovery_source=str(source),
        original_checkpoint_sha256=sha(original),checkpoint_sha256=sha((source/'checkpoint.json').read_bytes()),
        recovery_sha256=sha((source/'runtime_recovery/recovery.json').read_bytes()),
        original_steps=len(old['trace']),final_steps=len(saved['trace']),exact_prefix=True,
        selected_mode=recovery['selected_mode'])
    receipts=root/'recovery_receipts';receipts.mkdir(exist_ok=True)
    (receipts/f"{name}.json").write_bytes(encoded(receipt))
    # The immutable downloaded source and original artifact folders stay intact.
    # Copy the complete status last so a live collector cannot see a false
    # complete state while the checkpoint/result are still being installed.
    for p in source.iterdir():
        if p.name=='status.json':continue
        target=dest/p.name
        if p.is_dir():shutil.copytree(p,target,dirs_exist_ok=True)
        else:shutil.copy2(p,target)
    shutil.copy2(source/'status.json',dest/'status.json')
    print(encoded(receipt).decode())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--recovered',required=True)
    a=p.parse_args();accept(a.root,a.recovered)
