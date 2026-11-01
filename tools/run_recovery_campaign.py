"""One finite shared-continuation development campaign; no automatic adoption."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.adverse_segment import audit
from tools.owned_checkpoint import encoded, sha
from tools.recovery_benefit import run_branch, save
from tools.run_adverse_recovery import run, verify_registration
from tools.run_continuation_portfolio import game_for


def retain_origin(checkpoint, origin, spec, boundary):
    if not checkpoint.exists():
        return
    raw = checkpoint.read_bytes(); envelope = json.loads(raw); payload = envelope['payload']
    assert sha(encoded(payload)) == envelope['sha256']
    tick = payload['world']['tick']
    if tick == boundary and not game_for(spec).terminal(payload['world']):
        if origin.exists():
            assert origin.read_bytes() == raw, 'origin replaced'
        else:
            temp = origin.with_suffix('.new'); temp.write_bytes(raw); temp.replace(origin)
    if tick > boundary and not origin.exists():
        raise ValueError('missing actual boundary snapshot; no substituted origin')


def execute(plan_path, output, task):
    plan_path = Path(plan_path); output = Path(output)
    registration = verify_registration(plan_path); plan = json.loads(plan_path.read_bytes())
    assert sha((ROOT/'docs/COMPLETION.md').read_bytes()) == plan['completion_contract_sha256']
    spec, kind, mode, experience = task
    if kind == 'episode':
        folder = output/'episodes'/spec['id']; folder.mkdir(parents=True, exist_ok=True)
        origin = output/'origins'/(spec['id']+'.json')
        # Recover the copy after a process exit between checkpoint and origin writes.
        retain_origin(folder/'checkpoint.json', origin, spec, plan['boundary'])
        result = run(folder, spec, 'expected', registration=registration,
                     max_decisions=plan['slice_decisions'], continuation_scope='shared')
        proof = audit(folder, spec, 'expected', registration)
        count = proof['actual_steps']
        retain_origin(folder/'checkpoint.json', origin, spec, plan['boundary'])
        if result is not None:
            assert proof['complete'] and result['complete']
        save(folder/'status.json', dict(**proof, state='complete' if proof['complete'] else 'handoff',
            result_sha256=sha((folder/'result.json').read_bytes()) if result is not None else None))
        return dict(task=task, finished=proof['complete'], steps=count,
                    origin=origin.exists(), unavailable=proof['complete'] and not origin.exists())
    folder = output/'branches'/f"{spec['id']}--{mode}--{experience}"
    result = run_branch(folder, spec, output/'origins'/(spec['id']+'.json'), mode, experience,
        registration, boundary=plan['boundary'], window=plan['window'],
        max_decisions=plan['slice_decisions'])
    progress = json.loads((folder/'progress.json').read_bytes())
    return dict(task=task, finished=result is not None, steps=progress['actual_steps'])


def main(plan_path, output, workers):
    plan_path = Path(plan_path).resolve(); output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True); (output/'origins').mkdir(exist_ok=True)
    lock = (output/'coordinator.lock').open('a+b')
    if os.name == 'nt':
        import msvcrt
        lock.write(b'0'); lock.flush(); lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_registration(plan_path); plan = json.loads(plan_path.read_bytes())
    assert plan['boundary'] % plan['slice_decisions'] == 0
    # Round-robin slices avoid assigning every worker to the first rule family.
    queue = deque((s, 'episode', None, None) for s in plan['specs'])
    queued_origins = set(); finished = {}; failures = []; unavailable = []
    def key(task):
        s, kind, mode, experience = task
        return '|'.join((s['id'],kind,str(mode),str(experience)))
    def status(active):
        save(output/'status.json', dict(updated=time.time(), pid=os.getpid(),
            finished=len(finished), expected=len(plan['specs'])*5,
            active=[key(t) for t in active], queued=len(queue), failures=failures,
            unavailable=unavailable, execution_finished=not queue and not active and not failures,
            all_registered_windows_available=not unavailable,
            adoption=False))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        active = {}
        while queue or active:
            while queue and len(active) < workers:
                task = queue.popleft()
                active[pool.submit(execute, str(plan_path), str(output), task)] = task
            status(active.values())
            ready, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in ready:
                task = active.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    failures.append(dict(task=key(task),error=repr(error)))
                    continue
                if result['finished']:
                    finished[key(task)] = result
                else:
                    queue.append(task)
                spec = task[0]
                if task[1] == 'episode' and result['origin'] and spec['id'] not in queued_origins:
                    queued_origins.add(spec['id'])
                    queue.extend((spec,'branch',mode,experience) for mode in ('changed','steady')
                                 for experience in ('learning','frozen'))
                if result.get('unavailable'):
                    unavailable.append(spec['id'])
                    save(output/'origins'/(spec['id']+'.unavailable.json'),
                         dict(case=spec['id'],reason='actual terminal before intervention boundary'))
        status(())
    verify_registration(plan_path)
    if failures:
        raise RuntimeError('campaign execution errors retained; no success or automatic retry')
    save(output/'done.json',dict(execution_finished=True,unavailable=unavailable,
        registered_branches=len(plan['specs'])*4,unavailable_branches=len(unavailable)*4,adoption=False,
        interpretation='Execution finished; comparison to parent controls and C3 audit still required.'))
    lock.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True); p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=8); args = p.parse_args()
    if not 1 <= args.workers <= 8:
        p.error('1..8 workers required')
    main(args.plan,args.output,args.workers)
