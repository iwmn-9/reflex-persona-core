"""Audit preserved prefixes AFTER an explicitly authorized study retirement.

This tool never stops or resumes processes. The caller must first stop the
selected finite coordinator/finisher/workers, preserving their original files.
An incomplete study remains incomplete and cannot qualify adoption.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile
from snapshot_registered_queues import argv, canonical, processes

def sha(data):return hashlib.sha256(data).hexdigest()
def encoded(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def inactive(manifest,powershell):
    for row in processes(powershell):
        # This read-only auditor also takes --manifest. It is not a writer.
        if row['ProcessId']==os.getpid():continue
        args=argv(row.get('CommandLine') or '')
        if '--manifest' in args:
            index=args.index('--manifest')
            if index+1<len(args) and canonical(args[index+1])==canonical(manifest):
                raise ValueError('A registered coordinator/worker is still active; no checkpoint will be opened')

def audit(manifest,powershell,output,reason):
    manifest,output=Path(manifest).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve previous audit')
    if not reason.strip():raise ValueError('Explicit retirement reason required')
    raw=manifest.read_bytes();registration=json.loads(raw)
    jobs=registration['jobs']
    assert len({canonical(j['source']) for j in jobs})==1
    assert len({canonical(j['plan']) for j in jobs})==1
    source=Path(jobs[0]['source'])
    inactive(manifest,powershell);inactive(manifest,powershell)
    os.environ.update(jobs[0]['arithmetic'])
    os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    sys.path.insert(0,str(source))
    from tools.adverse_segment import audit as audit_one
    from tools.run_adverse_recovery import verify_registration
    original_registration=verify_registration(Path(jobs[0]['plan']))
    assert all(j['registration']==original_registration for j in jobs)
    output.mkdir(parents=True)
    records=[];inputs={}
    with zipfile.ZipFile(output/'preserved_histories.zip','x',zipfile.ZIP_DEFLATED) as packed:
        packed.writestr('manifest.json',raw)
        for index,job in enumerate(jobs):
            inactive(manifest,powershell)
            directory=Path(job['output']);checkpoint=directory/'checkpoint.json'
            if not checkpoint.exists():
                assert not (directory/'result.json').exists()
                row=dict(index=index,id=job['spec']['id'],arm=job['arm'],complete=False,
                    actual_steps=0,state='not_started_no_history')
            else:
                before=checkpoint.read_bytes()
                facts=audit_one(directory,job['spec'],job['arm'],job['registration'])
                assert checkpoint.read_bytes()==before
                row=dict(index=index,id=job['spec']['id'],arm=job['arm'],**facts,
                    state='complete_original_execution' if facts['complete'] else 'retired_saved_partial')
            records.append(row)
            for filename in ('checkpoint.json','result.json','status.json','execution.stdout.log','execution.stderr.log'):
                file=directory/filename
                if file.exists():
                    data=file.read_bytes();name=f'job-{index:03d}/{filename}'
                    packed.writestr(name,data);inputs[name]=sha(data)
            print(json.dumps(row),flush=True)
        for filename in ('manifest.json','status.json','verification.json'):
            file=manifest.parent/filename
            if file.exists():
                data=file.read_bytes();name='original_queue/'+filename
                packed.writestr(name,data);inputs[name]=sha(data)
        packed.writestr('audit_retired_study.py',Path(__file__).read_bytes())
    inactive(manifest,powershell)
    assert manifest.read_bytes()==raw
    assert verify_registration(Path(jobs[0]['plan']))==original_registration
    with zipfile.ZipFile(output/'preserved_histories.zip') as packed:
        assert all(sha(packed.read(name))==value for name,value in inputs.items())
    proof=dict(audit_complete=True,study_complete=all(r['complete'] for r in records),
        adoption=False,reason=reason,
        at=datetime.now(timezone.utc).isoformat(),manifest=str(manifest),
        manifest_sha256=sha(raw),source_ref=registration['source_ref'],registration=original_registration,
        registered_executions=len(jobs),original_completed=sum(r['complete'] for r in records),
        retained_actual_steps=sum(r['actual_steps'] for r in records),rows=records,
        required_steps=sum(j['spec']['turns'] for j in jobs),source_unchanged=True,
        history_archive_sha256=sha((output/'preserved_histories.zip').read_bytes()),inputs=inputs,
        limitations=['No missing execution or partial history is completed, substituted, or counted successful.',
            'Only factual action/learning/personality/world/RNG replay was performed; no forecast or new world.',
            'Retirement reason is supplied by the caller; this read-only audit does not itself establish user authorization.',
            'Original files and logs remain unchanged; original queue status may describe its state before retirement.',
            'This audit is not the full-cohort scientific result or a loss-diagnostic archive.'])
    (output/'verification.json').write_bytes(encoded(proof))
    print(json.dumps({k:proof[k] for k in ('audit_complete','study_complete','original_completed','retained_actual_steps','required_steps')}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','powershell','output','reason'):parser.add_argument('--'+name,required=True)
    audit(**vars(parser.parse_args()))
