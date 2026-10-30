"""Complete-only source/history audit and archival of one frozen study.

Optional finite waiting only observes an existing collector. It never launches
calculations, changes gates, adopts a candidate or creates a recurring task.
"""
import argparse
from datetime import datetime,timezone
import gzip
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reflex.core import digest,character_key
from tools.owned_checkpoint import encoded,sha
from tools.run_adverse_recovery import game_for
from tools.run_owned_value import law,sample
from tools.review_adverse_recovery import summarize
from tools.analyze_adverse_results import analyze


def registered_sources(source_ref, expected, root=ROOT):
    """Read every registered path from the frozen commit, including tests.

    Archive roots come from the registration rather than a fixed directory
    list. Only the individually registered and hash-checked files are retained.
    """
    if not expected:raise ValueError('nonempty source registration required')
    roots=sorted({name.split('/')[0] for name in expected})
    raw=subprocess.check_output(['git','archive','--format=zip',source_ref,'--',*roots],cwd=root)
    source={}
    with zipfile.ZipFile(io.BytesIO(raw)) as frozen:
        for name,checksum in expected.items():
            data=frozen.read(name)
            assert sha(data)==checksum,(name,'source mismatch')
            source[name]=data
    return source


def archive(root,plan_path,source_ref,output):
    root=Path(root);plan_path=Path(plan_path);out=Path(output)
    plan=json.loads(plan_path.read_bytes());proof=json.loads((plan_path.parent/'registration.json').read_bytes())
    assert sha(plan_path.read_bytes())==proof['plan_sha256']
    expected=dict(plan_sha256=proof['plan_sha256'],source_sha256=sha(encoded(proof['sources'])))
    summary=summarize(root/'latest',plan)
    assert summary['complete'] and summary['pairs_complete']==len(plan['specs'])
    if out.exists():raise FileExistsError('do not replace an existing complete audit')
    source=registered_sources(source_ref,proof['sources'])
    rows=[];audits=[];saved_files={}
    for spec in plan['specs']:
        matched=[]
        for arm in plan['arms']:
            folder=root/'latest'/f"adverse-{spec['id']}-{arm}"
            files={name:(folder/name).read_bytes() for name in ('checkpoint.json','result.json','status.json')}
            envelope=json.loads(files['checkpoint.json']);p=envelope['payload'];r=json.loads(files['result.json']);status=json.loads(files['status.json'])
            assert sha(encoded(p))==envelope['sha256']
            cfg=p['contract']['configuration'];trace=r['trace']
            assert cfg==r['configuration'] and cfg['registration']==status['registration']==expected
            assert cfg['runner_sha256']==proof['sources']['tools/run_adverse_recovery.py']
            assert all(h==proof['sources'][n] for n,h in p['contract']['sources'].items())
            assert all(h==proof['sources']['tools/'+n] for n,h in cfg['adapters'].items())
            assert trace==p['trace'] and r['final_owner']==digest(p['owner'])
            assert r['final_rng']==p['rng'] and status['checkpoint_sha256']==sha(files['checkpoint.json'])
            assert len(trace)==spec['turns']==r['replayed_actual_steps']==status['actual_steps']
            g=game_for(spec);w=g.start();rng=np.random.default_rng(spec['seed']);axes=character_key(g.context(w))
            prior=None;increments=[]
            for t in trace:
                assert t['before']==w and t['tick']==w['tick']
                if prior is not None:assert t['before_owner']==prior
                truth=law(w['tick'],spec['turns'],spec['regime']);assert truth==t['truth']
                response=sample(truth,rng.random());assert response==t['response']
                increment=sum(prob*(g.goal(g.transition(w,t['action'],z)[0]).value-g.goal(w).value) for z,prob in truth.items())
                assert increment==t['conditional_increment'];increments.append(increment)
                w,_=g.transition(w,t['action'],response);assert w==t['after']
                assert character_key(g.context(w))==axes;prior=t['owner']
            assert g.terminal(w) and w==r['final'] and prior==r['final_owner']
            # Preserve the frozen worker's built-in sum reduction. Sequential
            # += has different float rounding on Python 3.12; no tolerance or
            # change to any recorded increment is needed.
            assert rng.bit_generator.state==p['rng'] and sum(increments)==r['conditional_increment']
            assert r['score']==g.goal(w).value and p['owner']['last_tick']+1==len(trace)
            rows.append(dict(spec=spec,arm=arm,result=r))
            audits.append(dict(id=spec['id'],arm=arm,steps=len(trace),hashes={n:sha(b) for n,b in files.items()},
                persona=axes,source_bound=True,physical_and_rng_replayed=True,
                owner_hash_chain=True,worker_exact_owner_replay=r['replayed_actual_steps']))
            matched.append([(t['truth'],t['response']) for t in trace])
            for name,b in files.items():saved_files[f"{spec['id']}/{arm}/{name}"]=b
        assert all(v==matched[0] for v in matched[1:]),'paired actual random events changed'
    review=analyze(root/'latest',plan)
    out.mkdir(parents=True)
    (out/'summary.json').write_bytes(encoded(summary));(out/'outcome_review.json').write_bytes(encoded(review))
    (out/'plan.json').write_bytes(plan_path.read_bytes());(out/'registration.json').write_bytes(encoded(proof))
    (out/'runs.jsonl.gz').write_bytes(gzip.compress(b''.join(encoded(row)+b'\n' for row in rows),mtime=0))
    for name,contents in (('source.zip',source),('checkpoints.zip',saved_files)):
        with zipfile.ZipFile(out/name,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for key,data in contents.items():z.writestr(key,data)
    verification=dict(complete=True,at=datetime.now(timezone.utc).isoformat(),source_ref=source_ref,
        pairs=len(plan['specs']),executions=len(rows),actual_steps=sum(r['steps'] for r in audits),
        registration=expected,workers=audits,development_gate_passed=summary['development_gate_passed'],
        adoption=False,independent_confirmation='not started',
        artifacts={p.name:sha(p.read_bytes()) for p in out.iterdir()},auditor_sha256=sha(Path(__file__).read_bytes()),
        limitations=['Exact owner replay was performed by each frozen execution worker and exact arithmetic recovery. Local audit verifies their receipts, full hash chain, source and saved final owner; it does not claim cross-platform reexecution of every planning search.',
            'These are selected development conditions, not independent confirmation. All individual adverse results remain.'])
    (out/'verification.json').write_bytes(encoded(verification))
    print(encoded({k:verification[k] for k in ('complete','pairs','actual_steps','development_gate_passed','adoption')}).decode(),flush=True)
    return verification


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--plan',required=True)
    p.add_argument('--source-ref',required=True);p.add_argument('--output',required=True);p.add_argument('--wait-minutes',type=int,default=0)
    a=p.parse_args();assert 0<=a.wait_minutes<=240
    deadline=time.monotonic()+60*a.wait_minutes;ready=0
    plan=json.loads(Path(a.plan).read_bytes());status_path=Path(a.root)/'archive_status.json'
    while True:
        report=summarize(Path(a.root)/'latest',plan)
        ready=ready+1 if report['complete'] else 0
        status_path.write_bytes(encoded(dict(state='ready' if ready else 'waiting_for_existing_collection',
            completed_pairs=report['pairs_complete'],registered_pairs=len(plan['specs']),
            updated=datetime.now(timezone.utc).isoformat(),pid=__import__('os').getpid())))
        if ready>=(2 if a.wait_minutes else 1):break
        if time.monotonic()>=deadline:raise RuntimeError('finite archive wait ended before full evidence was available')
        time.sleep(60)
    try:
        v=archive(a.root,a.plan,a.source_ref,a.output)
        status_path.write_bytes(encoded(dict(state='complete_audited',output=a.output,development_gate_passed=v['development_gate_passed'],adoption=False)))
    except BaseException as e:
        status_path.write_bytes(encoded(dict(state='audit_failed',error=repr(e))))
        raise
