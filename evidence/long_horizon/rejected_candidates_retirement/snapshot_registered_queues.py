"""Read-only finite-queue observation; never open a live worker's JSON.

Match workers by manifest + job, including children of retired coordinators.
An append notice is saved progress, never evidence of a successful worker exit.
Only an inactive worker's final, registration-matched audited status can count.
This is monitoring evidence, not a replacement for the cohort/archive audit.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone


def canonical(path):
    return os.path.normcase(str(Path(path).resolve()))


def argv(command):
    argc = ctypes.c_int()
    api = ctypes.windll.shell32.CommandLineToArgvW
    api.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
    api.restype = ctypes.POINTER(ctypes.c_wchar_p)
    pointer = api(command, ctypes.byref(argc))
    if not pointer:
        raise ctypes.WinError()
    try:
        return [pointer[index] for index in range(argc.value)]
    finally:
        free = ctypes.windll.kernel32.LocalFree
        free.argtypes = [ctypes.c_void_p]
        free.restype = ctypes.c_void_p
        free(pointer)


def processes(powershell):
    command = ("@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
               "Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine) | "
               "ConvertTo-Json -Compress -Depth 3")
    output = subprocess.check_output([powershell, '-NoProfile', '-Command', command],
                                    encoding='utf-8-sig', timeout=60)
    rows = json.loads(output) or []
    return rows if isinstance(rows, list) else [rows]


def match_workers(rows, manifest):
    found = {}
    for row in rows:
        command = row.get('CommandLine') or ''
        args = argv(command) if command else []
        if '--manifest' not in args:
            continue
        index = args.index('--manifest')
        if index+1 >= len(args):
            raise ValueError('Malformed manifest argument')
        if canonical(args[index+1]) != canonical(manifest):
            continue
        if '--job' not in args:
            continue  # Coordinator, not a writer of a particular worker status.
        job = args.index('--job')
        number = int(args[job+1])
        found.setdefault(number, []).append(dict(pid=row['ProcessId'],
            parent=row['ParentProcessId'], created=row['CreationDate']))
    return found


def saved_notice(path, spec, arm):
    if not path.exists():
        return dict(saved_turn=0, terminal_notice=False)
    with path.open('rb') as stream:
        end = stream.seek(0, 2)
        start = max(0, end-65536)
        stream.seek(start)
        raw = stream.read()
    lines = raw.splitlines()
    if start:
        lines = lines[1:]
    if raw and not raw.endswith(b'\n'):
        lines = lines[:-1]
    turn, terminal = 0, False
    for line in lines:
        try:
            notice = json.loads(line)
        except (ValueError, UnicodeError):
            continue
        if not isinstance(notice, dict) or notice.get('id') != spec['id'] or notice.get('arm') != arm:
            continue
        tick = notice.get('turn')
        if type(tick) is int and 0 <= tick <= spec['turns'] and tick >= turn:
            turn = tick
            terminal = tick == spec['turns'] and notice.get('complete') is True
    return dict(saved_turn=turn, terminal_notice=terminal)


def classify(job, notice, before, after, load_status):
    if before or after:
        return dict(state='running_or_exiting', workers=before or after,
                    audited_complete=False, **notice)
    if not notice['terminal_notice']:
        return dict(state='not_running_saved_partial' if notice['saved_turn'] else 'not_started',
                    workers=[], audited_complete=False, **notice)
    try:
        status, status_sha = load_status()
    except (OSError, ValueError) as error:
        return dict(state='inactive_needs_review', workers=[], audited_complete=False,
                    status_error=repr(error), **notice)
    if not isinstance(status, dict):
        return dict(state='inactive_needs_review', workers=[], audited_complete=False,
                    status_error='Worker status is not an object', **notice)
    valid = (status.get('case') == job['spec']['id'] and status.get('arm') == job['arm']
             and status.get('registration') == job['registration']
             and status.get('state') == 'complete' and status.get('complete') is True
             and status.get('actual_steps') == job['spec']['turns'])
    return dict(state='audited_worker_complete' if valid else 'inactive_needs_review',
                workers=[], audited_complete=valid, status_sha256=status_sha,
                worker_recorded_state=status.get('state'), worker_error=status.get('error'),
                exit_code='not observed by this monitor', **notice)


def snapshot(manifests, powershell):
    prepared = []
    first = processes(powershell)
    for manifest_path in manifests:
        path = Path(manifest_path).resolve()
        data = path.read_bytes()  # Frozen registration; no live queue state.
        manifest = json.loads(data)
        jobs = manifest['jobs']
        workers = match_workers(first, path)
        if any(index not in range(len(jobs)) for index in workers):
            raise ValueError('Active worker is outside its registration')
        notices = [saved_notice(Path(job['output'])/'execution.stdout.log', job['spec'], job['arm']) for job in jobs]
        prepared.append((path, data, jobs, workers, notices))
    second = processes(powershell)
    studies = []
    for path, data, jobs, before, notices in prepared:
        after = match_workers(second, path)
        if any(index not in range(len(jobs)) for index in after):
            raise ValueError('Active worker is outside its registration')
        observations = []
        for index, (job, notice) in enumerate(zip(jobs, notices)):
            def load_status(job=job):
                raw = (Path(job['output'])/'status.json').read_bytes()
                return json.loads(raw), hashlib.sha256(raw).hexdigest()
            row = classify(job, notice, before.get(index, []), after.get(index, []), load_status)
            row.update(index=index, case=job['spec']['id'], arm=job['arm'])
            observations.append(row)
        studies.append(dict(manifest=str(path), manifest_sha256=hashlib.sha256(data).hexdigest(),
            registered=len(jobs), audited_workers=sum(row['audited_complete'] for row in observations),
            saved_turns=sum(row['saved_turn'] for row in observations), jobs=observations))
    return dict(recorded_at=datetime.now(timezone.utc).isoformat(), studies=studies,
                cohort_audit=False, adoption=False, limitations=[
                    'No live worker/queue/checkpoint JSON is opened; inactive final worker status is read only after two process snapshots.',
                    'Does not observe exit codes or replace the coordinator, complete-history audit, loss analysis, or acceptance gates.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', action='append', required=True)
    parser.add_argument('--powershell', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = snapshot(args.manifest, args.powershell)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps([dict(manifest=row['manifest'], registered=row['registered'],
        audited_workers=row['audited_workers'], saved_turns=row['saved_turns']) for row in result['studies']]))
