"""One finite research slice; actual errors are not budget handoffs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.horizon_segment import bounded_command, SegmentPaused
from reflex.core import digest
from tools.owned_checkpoint import encoded, sha
from tools.run_adverse_recovery import replay, verify_registration, game_for


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    path = Path(path)
    temp = path.with_suffix('.new')
    temp.write_bytes(encoded(value))
    temp.replace(path)


def audit(directory, spec, arm, registration):
    directory = Path(directory)
    envelope = read(directory/'checkpoint.json')
    saved = envelope['payload']
    assert sha(encoded(saved)) == envelope['sha256']
    cfg = saved['contract']['configuration']
    assert cfg['spec'] == spec and cfg['arm'] == arm and cfg['registration'] == registration
    trace = saved['trace']
    n = replay(game_for(spec), spec, trace, saved['owner'], saved['rng'])
    assert n == saved['world']['tick'] == saved['owner']['last_tick']+1
    assert saved['world'] == trace[-1]['after']
    assert saved['checks'] == dict(fixed_axes=n, owner_parity=n, restored=n)
    complete = game_for(spec).terminal(saved['world'])
    if (directory/'result.json').exists():
        result = read(directory/'result.json')
        assert complete and result['complete'] and result['trace'] == trace
        assert result['configuration'] == cfg and result['replayed_actual_steps'] == n
        assert result['final_owner'] == digest(saved['owner'])
    elif complete:
        # A process can stop between the last atomic checkpoint and final audit.
        complete = False
    return dict(complete=complete, actual_steps=n, checkpoint_sha256=sha((directory/'checkpoint.json').read_bytes()),
        runtime=saved['contract']['runtime'], registration=registration)


def segment(plan, case, arm, output, minutes):
    plan, output = Path(plan).resolve(), Path(output).resolve()
    registration = verify_registration(plan)
    spec = next(s for s in read(plan)['specs'] if s['id'] == case)
    output.mkdir(parents=True, exist_ok=True)
    start_trace = []
    if (output/'checkpoint.json').exists():
        start_trace = read(output/'checkpoint.json')['payload']['trace']
        if audit(output, spec, arm, registration)['complete']:
            return
    started = time.monotonic()
    status = dict(case=case, arm=arm, state='running', complete=False,
        from_tick=len(start_trace), actual_steps=len(start_trace), registration=registration)
    def status_write(**kw):
        status.update(kw, updated=datetime.now(timezone.utc).isoformat())
        write(output/'status.json', status)
    status_write()
    args = [sys.executable, '-u', str(ROOT/'tools/run_adverse_recovery.py'),
        '--plan', str(plan), '--case', case, '--arm', arm, '--output', str(output)]
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(bounded_command, args, output, started+minutes*60, 'execution')
            while not future.done():
                if (output/'checkpoint.json').exists():
                    n = len(read(output/'checkpoint.json')['payload']['trace'])
                    status_write(actual_steps=n)
                    print(json.dumps(dict(case=case, arm=arm, actual_steps=n,
                        planned_steps=spec['turns'], complete=False)), flush=True)
                time.sleep(15)
            paused = False
            try:
                future.result()
            except SegmentPaused:
                paused = True
        facts = audit(output, spec, arm, registration)
        trace = read(output/'checkpoint.json')['payload']['trace']
        assert trace[:len(start_trace)] == start_trace, 'actual history replaced'
        assert facts['complete'] or (paused and len(trace)>len(start_trace)), 'no valid progress'
        status_write(**facts, state='complete' if facts['complete'] else 'handoff_pending',
            retained_prefix_sha256=sha(encoded(start_trace)), seconds=time.monotonic()-started)
        print(json.dumps(status), flush=True)
    except BaseException as error:
        status_write(state='failed', error=repr(error))
        raise


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--case',required=True)
    p.add_argument('--arm',required=True,choices=('stable','stratified'))
    p.add_argument('--output',required=True);p.add_argument('--minutes',type=float,default=325)
    a=p.parse_args()
    if not 0<a.minutes<=330:raise ValueError('explicit bounded execution required')
    segment(a.plan,a.case,a.arm,a.output,a.minutes)
