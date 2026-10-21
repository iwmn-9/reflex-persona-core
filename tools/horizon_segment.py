"""Operational handoff of frozen worlds; a handoff never certifies completion.

Only the Linux Actions worker uses a wall-clock budget. Its frozen child is
stopped before the platform timeout; existing atomic checkpoints, actual
history and registered source are then independently verified. Real errors,
missing progress, or invalid evidence retain a failed workflow conclusion.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

PAUSED = 75
RECEIPT = 'segment_handoff.json'


class SegmentPaused(Exception):
    pass


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path=Path(path);temporary=path.with_suffix('.new')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    temporary.replace(path)


def bounded_command(args, root, deadline, stage):
    """Stop only our own process group; stderr/abnormal exits cannot be pauses."""
    if os.name != 'posix':
        raise RuntimeError('bounded execution is supported on the Linux worker only')
    root=Path(root);stdout=root/f'{stage}.stdout.log';stderr=root/f'{stage}.stderr.log'
    stop=dict(version='horizon-interruption-v1',stage=stage,deadline_monotonic=deadline,
              requested_at=datetime.now(timezone.utc).isoformat(),termination=None)
    remaining=deadline-time.monotonic()
    if remaining<=0:
        stop.update(termination='budget_exhausted_before_launch',child_return_code=None)
    else:
        with stdout.open('wb') as out,stderr.open('wb') as err:
            process=subprocess.Popen(args,stdout=out,stderr=err,start_new_session=True)
            try:
                code=process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                # poll again: an already reported failure is never reclassified.
                code=process.poll()
                if code is None:
                    try:os.killpg(process.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:code=process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL);process.wait()
                        raise RuntimeError('worker refused planned stop; evidence retained')
                    if code == -signal.SIGTERM:
                        stop.update(termination='requested_sigterm',child_return_code=code)
        if stop['termination'] is None:
            if code:
                print(stderr.read_text(errors='replace'),file=sys.stderr,flush=True)
                raise subprocess.CalledProcessError(code,args)
            return
        # Python exceptions racing the signal are conservative failures, too.
        if stderr.stat().st_size:
            raise RuntimeError('stderr during planned stop; inspect before continuing')
        stop.update(stdout_sha256=sha(stdout),stderr_sha256=sha(stderr))
    save(root/'segment_interruption.json',stop)
    raise SegmentPaused('segment budget exhausted; completion remains unverified')


def handoff_facts(directories, expected_jobs, *, failed_peer=False):
    """Derive proof from files, rather than trusting a success/complete flag."""
    from tools.resume_horizon_pair import inspect,encoded,sha as bytes_sha
    from tools.audit_recorded_forecasts import probability_differences,verify_receipt
    directories=list(map(Path,directories));assert len(directories)==len(expected_jobs)==2
    headers=[read(d/'contract_pair.json') for d in directories]
    assert headers[0]==headers[1],'handoff pair mismatch'
    header=headers[0]
    assert header['complete'] is False,'completed pair is not a handoff'
    assert header['version']=='segmented-same-host-value-contract-v1'
    assert header.get('jobs',dict(baseline=header['job'],candidate=header['job']))==dict(zip(('baseline','candidate'),expected_jobs))
    codes=header['return_codes']
    assert len(codes)==2 and PAUSED in codes
    if failed_peer:assert 1 in codes and all(c in (1,PAUSED) for c in codes),'explicit failed peer required'
    else:assert all(c in (0,PAUSED) for c in codes),'real failure is not a handoff'
    start=header['continuation']['boundary'];arms=[]
    assert min(len(read(d/'checkpoint.json')['payload']['trace']) for d in directories)>start,'no resolved progress; planned stop is not a valid handoff'
    for side,(directory,job,code) in enumerate(zip(directories,expected_jobs,codes)):
        item=inspect(directory,sha(directory/'source_snapshot.json'),sha(directory/'checkpoint.json'),job,pair_file='contract_pair.json')
        assert {k:v for k,v in item['files'].items() if k.endswith('.py')}==header['sources']
        saved=item['saved'];trace=saved['trace'];ticks=len(trace)
        boundary=read(directory/'boundary_provenance.json')
        assert boundary['resume_at']==start<=ticks
        assert bytes_sha(encoded(trace[:start]))==boundary['retained_prefix_sha256'],'handoff changed original prefix'
        assert boundary['original_checkpoint_sha256']==header['continuation']['inputs'][side]['checkpoint_sha256']
        assert item['manifest_sha256']==header['continuation']['inputs'][side]['manifest_sha256']
        hashes={f:sha(directory/f) for f in ('source_snapshot.json','checkpoint.json','registration.json','boundary_provenance.json','execution_status.json')}
        state=read(directory/'execution_status.json')
        assert state['return_code']==code and state.get('ended_at')
        if header.get('retained_history_required'):
            retained=directory/'retained_execution.json';raw=read(retained)
            assert sha(retained)==header['continuation']['inputs'][side]['retained_checkpoint_sha256']
            assert bytes_sha(encoded(raw['payload']))==raw['sha256']
            assert raw['payload']['contract']==saved['contract']
            old=raw['payload']['trace'];common=min(ticks,len(old))
            assert trace[:common]==old[:common],'handoff replaced retained experience'
            replay=read(directory/'retained_history_replay.json')
            assert replay['start']==start and replay['checked_until']>=common
            assert replay['retained_checkpoint_sha256']==sha(retained)
            for n in replay['normalizations']:
                assert start<=n['tick']<len(old)
                assert probability_differences(old[n['tick']]['law'],n['fresh'])==n['differences']
            hashes.update({f:sha(directory/f) for f in ('retained_execution.json','retained_history_replay.json')})
        if code==PAUSED:
            stop=read(directory/'segment_interruption.json')
            assert stop['version']=='horizon-interruption-v1'
            if stop['termination']=='requested_sigterm':
                assert stop['child_return_code']==-signal.SIGTERM
                err=directory/f"{stop['stage']}.stderr.log"
                assert not err.read_bytes() and sha(err)==stop['stderr_sha256']
                assert sha(directory/f"{stop['stage']}.stdout.log")==stop['stdout_sha256']
                hashes.update({f:sha(directory/f) for f in (err.name,f"{stop['stage']}.stdout.log")})
            else:
                assert stop['termination']=='budget_exhausted_before_launch' and stop['child_return_code'] is None
            hashes['segment_interruption.json']=sha(directory/'segment_interruption.json')
        elif code==0:
            audit=read(directory/'audit.json');row=read(directory/'trajectories.jsonl')
            assert audit['complete'] and audit['source_freeze_verified']
            assert audit['trajectory_sha256']==sha(directory/'trajectories.jsonl')
            assert row['executed']==ticks and row['final']==saved['world']
            for key in ('world_rng_replay','restored_action_feedback','physical_replay','owner_parity','fixed_axes'):
                assert audit['checks'][key]==ticks
            assert [{k:v for k,v in t.items() if k not in ('forecast_error','realized_at_forecast_end')} for t in row['trace']]==trace
            verify_receipt(directory,audit,row)
            provenance=read(directory/'resume_provenance.json')
            assert provenance['complete'] and provenance['original_prefix_unchanged']
            hashes.update({f:sha(directory/f) for f in ('audit.json','trajectories.jsonl','resume_provenance.json')})
            if (directory/'recorded_forecast_replay.json').exists():hashes['recorded_forecast_replay.json']=sha(directory/'recorded_forecast_replay.json')
        arms.append(dict(side=('baseline','candidate')[side],ticks=ticks,return_code=code,files=hashes))
    end=min(a['ticks'] for a in arms)
    assert end>start,'no resolved progress; planned stop is not a valid handoff'
    assert end<=expected_jobs[0][4]
    return dict(execution_id=header['execution_id'],plan_sha256=header['plan_sha256'],
                from_tick=start,to_tick=end,arms=arms,contract_pair_sha256=sha(directories[0]/'contract_pair.json'))


def write_handoff(root, jobs):
    root=Path(root);dirs=[root/n for n in ('baseline','candidate')]
    facts=handoff_facts(dirs,jobs)
    record=dict(version='horizon-handoff-v1',state='handoff_pending',experiment_complete=False,
        workflow_run=os.environ.get('GITHUB_RUN_ID'),workflow_attempt=os.environ.get('GITHUB_RUN_ATTEMPT'),
        at=datetime.now(timezone.utc).isoformat(),**facts)
    for directory in [root,*dirs]:save(directory/RECEIPT,record)
    return record


def verify_handoff(directories,jobs,*,run=None,plan_sha=None,sources=None):
    directories=list(map(Path,directories))
    records=[read(d/RECEIPT) for d in directories];record=records[0]
    assert records[0]==records[1],'different handoff receipts'
    assert record['version']=='horizon-handoff-v1' and record['state']=='handoff_pending'
    assert record['experiment_complete'] is False
    if run is not None:assert str(run)==record['workflow_run'],'wrong handoff workflow'
    if plan_sha is not None:assert record['plan_sha256']==plan_sha
    if sources is not None:assert read(directories[0]/'contract_pair.json')['sources']==sources
    for k,v in handoff_facts(directories,jobs).items():assert record[k]==v,('handoff proof changed',k)
    return record
