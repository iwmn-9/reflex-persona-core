"""Finite Colab CPU run: consecutive H32/native-learning fidelity, not quality adoption."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
CASES = ((0, 'auction', 0), (5, 'resources', 1), (10, 'combat', 2), (15, 'gambling', 3))
TIMEOUT = 1800
ERROR_LIMIT = 2e-12
ERROR_FIELDS = ('max_prediction_error', 'max_state_error', 'max_score_error')


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False), encoding='utf8')
    temporary.replace(path)


def hashes(binary):
    files = [*sorted((ROOT / 'reflex').glob('*.py')), *sorted((ROOT / 'rust').glob('reflex-*/**/*.rs')),
             *sorted((ROOT / 'rust').glob('reflex-*/Cargo.toml')), ROOT / 'rust/Cargo.toml',
             ROOT / 'rust/Cargo.lock', binary, Path(__file__)]
    files += [ROOT / 'tools' / name for name in (
        'verify_rust_owned.py', 'verify_rust_runtime.py', 'verify_rust_immediate.py',
        'verify_rust_policy_probe.py', 'continuing_games.py', 'run_owned_value.py',
        'asymmetric_games.py', 'open_model_games.py')]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def stop_group(process):
    """The verifier and its native child share our private Linux process group."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Linux/Colab required for bounded child process-group cleanup')
    binary = ROOT / 'rust/target/debug/examples/owned_verify'
    before = hashes(binary)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    plan = dict(cases=[dict(index=i, game=g, profile=p) for i, g, p in CASES],
                steps=187, horizon_at=153, horizon_steps=3, workers=args.workers,
                timeout_seconds=TIMEOUT, error_limit=ERROR_LIMIT, sources=before,
                scope='Consecutive selection/learning/restore fidelity; no intelligence-quality adoption')
    write(out / 'inputs.json', plan)
    started = time.monotonic()
    state = dict(complete=False, phase='running', cases={str(i): dict(index=i, game=g, profile=p, state='queued') for i, g, p in CASES})
    lock, cancelled, active = threading.Lock(), threading.Event(), {}

    def update(index=None, **changes):
        with lock:
            if index is not None:
                state['cases'][str(index)].update(changes)
            state['updated_utc'] = datetime.now(timezone.utc).isoformat()
            state['seconds'] = time.monotonic() - started
            write(out / 'status.json', state)

    def run(case):
        index, _, _ = case
        destination = out / f'case-{index:02d}'
        process = None
        began = time.monotonic()
        try:
            if cancelled.is_set():
                raise RuntimeError('batch cancelled before launch')
            command = [sys.executable, str(ROOT / 'tools/verify_rust_owned.py'), '--output', str(destination),
                       '--steps', '187', '--profiles', '4', '--case-index', str(index),
                       '--horizon-at', '153', '--horizon-steps', '3']
            environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            with (out / f'case-{index:02d}.log').open('w', encoding='utf8') as log:
                process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                           stderr=subprocess.STDOUT, env=environment, start_new_session=True)
                with lock:
                    active[index] = process
                update(index, state='running', pid=process.pid)
                while process.poll() is None:
                    if cancelled.is_set():
                        raise RuntimeError('batch cancelled')
                    if time.monotonic() - began >= TIMEOUT:
                        raise TimeoutError(f'case exceeded {TIMEOUT} seconds')
                    try:
                        progress = json.loads((destination / 'progress.json').read_bytes())
                        update(index, progress=progress)
                    except (FileNotFoundError, json.JSONDecodeError):
                        pass  # The existing child writes progress in place; retain the last full record.
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
            if process.returncode != 0:
                raise RuntimeError(f'verifier exited {process.returncode}; see case log')
            summary = json.loads((destination / 'summary.json').read_bytes())
            if not (summary.get('complete') is True and summary.get('decisions') == 187
                    and summary.get('saves') == 187 and summary.get('horizon_decisions') == 3):
                raise ValueError('incomplete case or wrong decision/save/H32 counts')
            if not all(type(summary.get(k)) in (int, float) and math.isfinite(summary[k])
                       and 0 <= summary[k] <= ERROR_LIMIT for k in ERROR_FIELDS):
                raise ValueError('existing numerical fidelity limit not met')
            update(index, state='complete', summary=summary)
        except Exception as error:
            update(index, state='failed', error=f'{type(error).__name__}: {error}')
        finally:
            try:
                if process is not None:
                    stop_group(process)
                with lock:
                    active.pop(index, None)
                update(index, pid=None, elapsed_seconds=time.monotonic() - began)
            except Exception as error:
                update(index, state='failed', cleanup_error=f'{type(error).__name__}: {error}')

    update()
    pool = ThreadPoolExecutor(max_workers=args.workers)
    interrupted = None
    def terminate(*_):
        raise KeyboardInterrupt('SIGTERM')
    previous_handler = signal.signal(signal.SIGTERM, terminate)
    try:
        futures = [pool.submit(run, case) for case in CASES]
        for future in as_completed(futures):
            future.result()
    except BaseException as error:
        interrupted = f'{type(error).__name__}: {error}'
    finally:
        cancelled.set()
        with lock:
            processes = list(active.values())
        for process in processes:
            try:
                stop_group(process)
            except Exception as error:
                interrupted = f'Cleanup failure: {type(error).__name__}: {error}'
        pool.shutdown(wait=True, cancel_futures=True)
        signal.signal(signal.SIGTERM, previous_handler)

    try:
        source_unchanged = hashes(binary) == before
    except OSError as error:
        source_unchanged = False
        interrupted = f'Source audit failure: {type(error).__name__}: {error}'
    for case in state['cases'].values():
        if case['state'] == 'queued':
            case.update(state='not_run', error='batch cancelled before launch')
    completed = [case['summary'] for case in state['cases'].values() if case['state'] == 'complete']
    counts = {key: sum(item[key] for item in completed) for key in ('decisions', 'saves', 'horizon_decisions')}
    good = (interrupted is None and source_unchanged and len(completed) == 4
            and counts == dict(decisions=748, saves=748, horizon_decisions=12))
    summary = dict(complete=good, completed_cases=len(completed), registered_cases=4,
                   sources_unchanged=source_unchanged, interrupted=interrupted, **counts,
                   **{key: max((item[key] for item in completed), default=None) for key in ERROR_FIELDS},
                   seconds=time.monotonic() - started)
    write(out / 'summary.json', summary)
    state.update(complete=good, phase='complete' if good else 'failed', summary=summary)
    update()
    print(json.dumps(summary, allow_nan=False), flush=True)
    if not good:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
