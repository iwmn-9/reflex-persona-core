"""One preregistered confirmation after a completed development gate.

No scheduler, policy optimization, automatic adoption or replacement of the
original 40/8/8. The playing source is copied byte-for-byte from the gate study;
this separate management program is also frozen and hashed before execution.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


def encoded(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def sha(data):return hashlib.sha256(data).hexdigest()


def family_gate(rows):
    groups=[]
    for game in sorted({r['spec']['game'] for r in rows}):
        rs=[r for r in rows if r['spec']['game']==game]
        groups.append(dict(game=game,conditions=len(rs),mean_delta=sum(r['delta'] for r in rs)/len(rs),
            conditional_delta=sum(r['conditional_delta'] for r in rs)/len(rs),
            worse=sum(r['delta'] < -1e-12 for r in rs)))
    return bool(groups) and all(g['mean_delta']>=-1e-12 and g['conditional_delta']>=-1e-12 for g in groups),groups


def require_registered_rows(rows,specs):
    expected={s['id']:s for s in specs}
    assert len(expected)==len(specs)==len(rows), 'missing or repeated conditions'
    assert len({r['id'] for r in rows})==len(rows), 'repeated result'
    for row in rows:assert expected[row['id']]==row['spec'], 'changed condition'


def verified_evidence(folder,source,plan_sha256):
    proof=json.loads((folder/'verification.json').read_bytes())
    assert proof['complete'] and proof['source_verified'] and proof['actual_owner_replayed']
    for name,value in proof['artifacts'].items():assert sha((folder/name).read_bytes())==value
    summary=json.loads((folder/'summary.json').read_bytes())
    assert summary['complete'] and summary['completed_pairs']==summary['registered_pairs']==proof['pairs']
    plan=json.loads((folder/'plan.json').read_bytes())
    assert sha((folder/'plan.json').read_bytes())==plan_sha256
    assert plan['source']==source
    require_registered_rows(summary['rows'],plan['specs'])
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--workers',type=int,default=6)
    args=p.parse_args();root=Path(args.root).resolve();base=Path.cwd()
    registration=json.loads((root/'registration.json').read_bytes())
    assert sha(Path(__file__).read_bytes())==registration['manager_sha256']
    manifest=json.loads((root/'source_snapshot.json').read_bytes())
    assert sha(encoded(manifest))==registration['algorithm_source_sha256']
    for name,value in manifest.items():assert sha((root/'_source'/name).read_bytes())==value
    lock=root/'execution.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,encoded(dict(pid=os.getpid(),created=datetime.now(timezone.utc).isoformat())));os.close(fd)
    status=dict(complete=False,adoption=False,pid=os.getpid(),phase='waiting-for-registered-gate')
    def save(**values):
        status.update(values,updated=datetime.now(timezone.utc).isoformat())
        temp=root/'status.new';temp.write_bytes(encoded(status));temp.replace(root/'status.json')
    save()
    try:
        gate=base/registration['gate_root']
        while True:
            state=json.loads((gate/'status.json').read_bytes())
            if state.get('error'):raise RuntimeError('Development gate failed: '+state['error'])
            if state['complete']:break
            if state['phase']=='screen-regression-needs-analysis':
                save(phase='not-started-gate-rejected',reason='preparation regression');return
            time.sleep(45)
        assert state['source']==registration['algorithm_source_sha256']
        screen=verified_evidence(base/registration['gate_screen_evidence'],
            registration['algorithm_source_sha256'],registration['gate_screen_plan_sha256'])
        worlds=verified_evidence(base/registration['gate_world_evidence'],
            registration['algorithm_source_sha256'],registration['gate_world_plan_sha256'])
        accepted,groups=family_gate(worlds['rows'])
        if any(r['delta'] < -1e-12 for r in screen['rows']) or not accepted:
            save(phase='not-started-gate-rejected',families=groups);return
        plan=dict(specs=registration['specs'],source=registration['algorithm_source_sha256'],
            registration_sha256=sha((root/'registration.json').read_bytes()),
            purpose='Fresh independent confirmation, all registered conditions retained; no automatic adoption',
            criteria=registration['criteria'])
        target=root/'plan.json'
        if target.exists():assert json.loads(target.read_bytes())==plan
        else:target.write_bytes(encoded(plan))
        sys.path.insert(0,str(root/'_source'))
        from tools.run_continuation_portfolio import pair
        assert Path(sys.modules[pair.__module__].__file__).resolve().is_relative_to(root/'_source')
        save(phase='confirmation',gate_families=groups)
        rows=[]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(pair,root,spec) for spec in plan['specs']]
            for future in as_completed(futures):
                rows.append(future.result())
                temp=root/'summary.new';temp.write_bytes(encoded(dict(complete=len(rows)==len(plan['specs']),
                    completed_pairs=len(rows),registered_pairs=len(plan['specs']),rows=sorted(rows,key=lambda r:r['id']))))
                temp.replace(root/'summary.json')
                save(completed_pairs=len(rows),registered_pairs=len(plan['specs']))
        require_registered_rows(rows,plan['specs'])
        save(phase='auditing')
        subprocess.run([sys.executable,str(root/'_source/tools/audit_continuation_portfolio.py'),
            '--root',str(root),'--publish',str(base/registration['evidence'])],check=True,cwd=base)
        passed,groups=family_gate(rows)
        save(complete=True,phase='audited-needs-review',family_gate_passed=passed,families=groups)
    except Exception as error:
        save(error=repr(error));raise
    finally:
        lock.unlink()


if __name__=='__main__':main()
