"""Finite execution of resolved loss diagnostics, never an actual-world study.

The manifest binds this driver, the probe, a fully audited root plan and the
original frozen worker. A prior interruption needs an explicit receipt hash
after the operator has checked that no child is still running.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock
import time


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix('.new')
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding='utf-8')
    deadline = time.monotonic() + 2
    while True:
        try:
            temporary.replace(path)
            return
        except OSError as error:
            if getattr(error, 'winerror', None) not in (5, 32) or time.monotonic() >= deadline:
                raise
            time.sleep(.05)


@contextmanager
def exclusive(path):
    with path.open('ab+') as stream:
        if os.name == 'nt':
            import msvcrt
            stream.seek(0); stream.write(b'0'); stream.flush(); stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def validate(job, result, original, registration, probe_hash):
    saved = original['trace'][job['tick']]
    seeds = list(range(job['start'], job['start'] + job['samples']))
    assert result['id'] == job['id'] and result['tick'] == job['tick']
    assert result['arm'] == original['configuration']['arm']
    assert result['seeds'] == seeds
    assert result['factual_owner'] == job['factual_owner'] == saved['before_owner']
    assert result['input_sha256'] == job['input_sha256']
    assert result['registration'] == registration and result['diagnostic_source_sha256'] == probe_hash
    assert result['runtime'] == original['runtime'] and result['exact_factual_prefix_steps'] == job['tick']
    control = seeds == original['configuration']['seeds']
    assert result['exact_registered_forecast_reproduced'] == control
    if control:
        assert (result['action'], result['selected_purpose']) == (saved['action'], saved['selected_purpose'])
        assert result['deliberation']['selected_plan'] == saved['selected_plan']
        assert result['purpose_by_plan'] == saved['purpose_by_plan']
    return control


def main(root, acknowledge_interruption=None, workers=1):
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError('one to eight currently available local CPU slots required')
    root = Path(root).resolve()
    manifest = read(root / 'manifest.json')
    assert sha(__file__) == manifest['driver_sha256']
    assert sha(root / 'plan.json') == manifest['plan_sha256']
    plan = read(root / 'plan.json')
    assert plan['complete_archive'] and not plan['adoption'] and plan['prior_diagnosis'] is None
    assert plan['source_ref'] == manifest['source_ref']
    probe = root / 'probe_registered_owner.py'
    assert sha(probe) == manifest['probe_sha256']
    worker = Path(manifest['worker'])
    source_plan = worker / manifest['source_plan']
    os.environ.update(read(source_plan)['arithmetic'])
    os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    sys.path.insert(0, str(worker))
    from tools.run_adverse_recovery import verify_registration
    registration = verify_registration(source_plan)
    assert registration == plan['registration']
    jobs = plan['jobs']
    assert all(not job['reuse'] for job in jobs)
    assert len(jobs) == plan['required_evaluations'] == plan['pending_evaluations']
    status_path = root / 'status.json'
    with exclusive(root / 'queue.lock'):
        if status_path.exists() and read(status_path)['state'] == 'complete':
            receipt = read(root / 'verification.json')
            assert receipt['complete'] and receipt['sources'] == manifest
            assert [row['job'] for row in receipt['rows']] == jobs
            assert all(sha(root / row['file']) == row['sha256'] for row in receipt['rows'])
            return
        if status_path.exists():
            if acknowledge_interruption != sha(status_path):
                raise RuntimeError('inspect prior parent/child and acknowledge the exact interruption receipt before resuming')
        status = dict(complete=False, adoption=False, pid=os.getpid(), evaluations_required=len(jobs),
            evaluations_complete=0, root_plan_sha256=manifest['plan_sha256'], source_registration=registration,
            local_workers=workers)
        rows, controls, active = {}, 0, {}
        active_lock = Lock()

        def evaluate(index, job):
            source = root / 'inputs' / job['input_archive_member']
            assert sha(source) == job['input_sha256']
            original = read(source)
            filename = f'root-{index:03}.json'; output = root / filename
            if not output.exists():
                with (root / 'execution.stdout.log').open('ab') as out, (root / 'execution.stderr.log').open('ab') as err:
                    child = subprocess.Popen([sys.executable, str(probe), '--worker', str(worker),
                        '--plan', str(source_plan), '--input', str(source), '--input-sha256', job['input_sha256'],
                        '--tick', str(job['tick']), '--start', str(job['start']), '--samples', str(job['samples']),
                        '--output', str(output)], stdout=out, stderr=err,
                        creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
                                      if os.name == 'nt' else 0)
                    with active_lock:
                        active[index] = dict(child_pid=child.pid, job=job)
                    code = child.wait()
                    with active_lock:
                        active.pop(index)
                    if code:
                        raise subprocess.CalledProcessError(code, child.args)
            control = validate(job, read(output), original, registration, manifest['probe_sha256'])
            return index, dict(job=job, file=filename, sha256=sha(output)), control

        pool = ThreadPoolExecutor(max_workers=workers)
        pending = set()
        try:
            status.update(state='starting', active={}, updated=datetime.now(timezone.utc).isoformat())
            save(status_path, status)
            next_index = 0
            while next_index < len(jobs) or pending:
                while next_index < len(jobs) and len(pending) < workers:
                    pending.add(pool.submit(evaluate, next_index, jobs[next_index]))
                    next_index += 1
                completed, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in completed:
                    index, row, control = future.result()
                    rows[index] = row; controls += control
                with active_lock:
                    current = dict(active)
                status.update(state='running', active=current, evaluations_complete=len(rows),
                              updated=datetime.now(timezone.utc).isoformat())
                save(status_path, status)
            pool.shutdown(wait=True)
            assert set(rows) == set(range(len(jobs)))
            assert verify_registration(source_plan) == registration
            assert controls == len({(j['id'], j['tick']) for j in jobs})
            save(root / 'verification.json', dict(complete=True, adoption=False, evaluations=len(rows),
                rows=[rows[i] for i in range(len(jobs))],
                root_plan_sha256=manifest['plan_sha256'], sources=manifest, exact_factual_roots=True,
                exact_original_forecast_controls=controls))
            status.update(state='complete', complete=True, evaluations_complete=len(rows), active={},
                          updated=datetime.now(timezone.utc).isoformat())
            save(status_path, status)
        except BaseException as error:
            # Do not dispatch remaining roots after a failure or kill a peer's
            # valid calculation. Finished files remain available for audit.
            for future in pending:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
            status.update(state='failed', error=repr(error), complete=False, updated=datetime.now(timezone.utc).isoformat())
            save(status_path, status)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--acknowledge-interruption')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    main(args.root, args.acknowledge_interruption, args.workers)
