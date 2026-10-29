"""Finite local execution of immutable registered studies; never adopts a result.

The platform handoff starts new paired replications. Remote histories are
retained separately, never transplanted into a different runtime contract.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    path = Path(path)
    temp = path.with_suffix('.new')
    temp.write_bytes(encoded(value))
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def exclusive(path):
    """An OS lock survives stale PID files and releases on process death."""
    import msvcrt
    with Path(path).open('a+b') as handle:
        handle.seek(0)
        handle.write(b'0')
        handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def progress(folder):
    folder = Path(folder)
    status = read(folder/'status.json') if (folder/'status.json').exists() else {}
    checkpoint = folder/'checkpoint.json'
    n = 0
    modified = None
    if checkpoint.exists():
        envelope = read(checkpoint)
        payload = envelope['payload']
        assert sha(encoded(payload)) == envelope['sha256']
        n = len(payload['trace'])
        assert n == payload['world']['tick'] == payload['owner']['last_tick']+1
        modified = checkpoint.stat().st_mtime
    return dict(state=status.get('state', 'pending'), actual_steps=n,
                checkpoint_modified=modified, complete=status.get('complete', False),
                pid=status.get('pid'), error=status.get('error'))


class ExistingWorker:
    """Observe our locked worker after a coordinator restart; never relaunch it."""
    def __init__(self, folder, pid):
        self.folder, self.pid = Path(folder), pid

    def poll(self):
        try:
            with exclusive(self.folder/'worker.lock'):
                state = progress(self.folder)
        except OSError:
            return None
        return 0 if state['complete'] and state['state'] == 'complete' else 1


def worker(manifest, job_index):
    job = manifest['jobs'][job_index]
    folder = Path(job['output'])
    folder.mkdir(parents=True, exist_ok=True)
    with exclusive(folder/'worker.lock'):
        for key, value in job['arithmetic'].items():
            assert os.environ.get(key) == value
        sys.path.insert(0, job['source'])
        from tools.run_adverse_recovery import run, verify_registration
        from tools.adverse_segment import audit
        registration = verify_registration(job['plan'])
        assert registration == job['registration']
        old = read(folder/'checkpoint.json')['payload']['trace'] if (folder/'checkpoint.json').exists() else []
        started = time.monotonic()
        state = dict(case=job['spec']['id'], arm=job['arm'], registration=registration,
                     complete=False, state='running', actual_steps=len(old), pid=os.getpid(),
                     started=now(), from_tick=len(old), retained_prefix_sha256=sha(encoded(old)))
        save(folder/'status.json', state)
        try:
            run(folder, job['spec'], job['arm'], registration=registration)
            facts = audit(folder, job['spec'], job['arm'], registration)
            trace = read(folder/'checkpoint.json')['payload']['trace']
            assert trace[:len(old)] == old, 'actual history replaced'
            assert facts['complete'] and facts['actual_steps'] == job['spec']['turns']
            state.update(facts, state='complete', updated=now(), seconds=time.monotonic()-started)
            save(folder/'status.json', state)
        except BaseException as error:
            state.update(state='failed', error=repr(error), updated=now(), complete=False)
            save(folder/'status.json', state)
            raise


def coordinate(path, workers):
    path = Path(path).resolve()
    root = path.parent
    manifest = read(path)
    assert manifest['driver_sha256'] == sha(Path(__file__).read_bytes())
    import numpy as np
    runtime = dict(python=sys.version, numpy=np.__version__, system=platform.system(), machine=platform.machine())
    assert runtime == manifest['runtime']
    with exclusive(root/'queue.lock'):
        active = {}
        # Recover our live workers rather than duplicate or terminate them.
        for i, job in enumerate(manifest['jobs']):
            folder = Path(job['output'])
            folder.mkdir(parents=True, exist_ok=True)
            try:
                with exclusive(folder/'worker.lock'):
                    pass
            except OSError:
                status = read(folder/'status.json')
                assert status['registration'] == job['registration']
                assert status['case'] == job['spec']['id'] and status['arm'] == job['arm']
                active[i] = ExistingWorker(folder, status['pid']), []
        done = set()
        failures = {}
        while True:
            snapshot = []
            for i, job in enumerate(manifest['jobs']):
                p = progress(job['output'])
                snapshot.append(dict(index=i, study=job['study'], case=job['spec']['id'], arm=job['arm'], **p))
                if p['complete']:
                    assert p['state'] == 'complete' and p['actual_steps'] == job['spec']['turns']
                    done.add(i)
            for i, (child, logs) in list(active.items()):
                code = child.poll()
                if code is None:
                    continue
                for handle in logs:
                    handle.close()
                del active[i]
                p = progress(manifest['jobs'][i]['output'])
                if code or not p['complete']:
                    failures[i] = dict(returncode=code, progress=p)
                else:
                    done.add(i)
            # Independent workers finish after a peer failure. No blind retry.
            if not failures:
                for i, job in enumerate(manifest['jobs']):
                    if len(active) >= workers:
                        break
                    if i in done or i in active:
                        continue
                    folder = Path(job['output'])
                    logs = [(folder/name).open('ab') for name in ('execution.stdout.log', 'execution.stderr.log')]
                    env = os.environ.copy()
                    env.update(job['arithmetic'])
                    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
                    child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                        '--manifest', str(path), '--job', str(i)], cwd=job['source'], env=env,
                        stdout=logs[0], stderr=logs[1], creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
                    active[i] = child, logs
            state = dict(updated=now(), pid=os.getpid(), manifest_sha256=sha(path.read_bytes()),
                complete=len(done) == len(manifest['jobs']), executions_complete=len(done),
                executions_registered=len(manifest['jobs']), active={i:p.pid for i,(p,_) in active.items()},
                failures=failures, progress=snapshot, adoption=False)
            save(root/'state.json', state)
            if not active and (failures or len(done) == len(manifest['jobs'])):
                if failures:
                    raise RuntimeError('local worker failed; evidence retained, inspect before restart')
                return
            time.sleep(15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--job', type=int)
    args = parser.parse_args()
    if args.job is None:
        if not 1 <= args.workers <= 8:
            raise ValueError('bounded local concurrency required')
        coordinate(args.manifest, args.workers)
    else:
        worker(read(args.manifest), args.job)
