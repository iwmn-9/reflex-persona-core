"""Run unchanged frozen continuations and publish independently readable progress.

The check output is telemetry, never a completed experiment or adoption result.
Final full artifacts and existing audits remain authoritative. No model imports.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from urllib.request import Request, urlopen


def utc():
    return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.new')
    tmp.write_text(json.dumps(value, indent=2)+'\n', encoding='utf8')
    tmp.replace(path)


def checkpoint_summary(directory, expected_job, expected_manifest):
    """Read one atomically replaced checkpoint, without modifying its source."""
    directory = Path(directory)
    checkpoint = directory/'checkpoint.json'
    if not checkpoint.exists():
        return None
    manifest = (directory/'source_snapshot.json').read_bytes()
    assert digest(manifest) == expected_manifest, 'progress source manifest drift'
    raw = checkpoint.read_bytes()
    envelope = json.loads(raw)
    p = envelope['payload']
    assert digest(encoded(p)) == envelope['sha256'], 'progress checksum mismatch'
    assert p['contract']['job'] == expected_job
    assert all(json.loads(manifest)[k] == v for k,v in p['contract']['sources'].items())
    trace = p['trace']; ticks = len(trace)
    assert 0 < ticks <= expected_job[4]
    assert p['world']['tick'] == ticks == p['owner']['last_tick']+1
    assert trace[-1]['after'] == p['world']
    owner = json.dumps(p['owner'], ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    assert trace[-1]['owner'] == digest(owner), 'progress owner mismatch'
    assert all(t['tick'] == i for i,t in enumerate(trace))
    return dict(ticks=ticks, requested_turns=expected_job[4], checkpoint_sha256=digest(raw),
                source_manifest_sha256=expected_manifest, elapsed_seconds=p['elapsed'],
                checkpoint_mtime_utc=datetime.fromtimestamp(checkpoint.stat().st_mtime,timezone.utc).isoformat())


def process_table(proc=Path('/proc')):
    """Linux observer; no environment variables, command lines or secrets read."""
    rows = {}
    if not proc.exists():
        return rows
    hz = os.sysconf('SC_CLK_TCK')
    for path in proc.glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            rows[int(path.parent.name)] = dict(parent=int(fields[1]), state=fields[0],
                cpu=(int(fields[11])+int(fields[12]))/hz, born=fields[19])
        except (OSError, ValueError, IndexError):
            continue  # Process exited during the read.
    return rows


def descendants(pid, table):
    found = {pid} if pid in table else set()
    while True:
        extra = {p for p,v in table.items() if v['parent'] in found} - found
        if not extra:
            return found
        found |= extra


class Progress:
    def __init__(self):
        self.previous = {}
        self.cpu = {}

    def observe(self, key, checkpoint, execution, table, now, monotonic):
        seen = self.cpu.setdefault(key, {})
        for pid in descendants(execution.get('pid'), table):
            item = table[pid]
            identity = (pid, item['born'])
            seen[identity] = max(seen.get(identity,0), item['cpu'])
        total = sum(seen.values())
        ticks = checkpoint['ticks'] if checkpoint else 0
        old = self.previous.get(key)
        advanced = old is not None and ticks > old['ticks']
        assert old is None or ticks >= old['ticks'], 'saved progress regressed'
        last_advance = now if advanced else (old['last_advance'] if old else None)
        busy = old is not None and total > old['cpu'] + .001
        last_activity = monotonic if advanced or busy or old is None else old['activity']
        code = execution.get('return_code')
        if code is not None:
            phase = 'finished' if code == 0 else 'failed'
        elif checkpoint is None:
            phase = 'preparing'
        elif not table:
            phase = 'cpu_unavailable'
        elif monotonic-last_activity >= 900:
            phase = 'suspected_stall_no_cpu_or_checkpoint_15m'
        elif advanced:
            phase = 'saved_advance'
        elif busy:
            phase = 'computing_no_new_checkpoint'
        else:
            phase = 'awaiting_next_observation'
        self.previous[key] = dict(ticks=ticks,cpu=total,last_advance=last_advance,activity=last_activity)
        return dict(phase=phase,last_observed_advance_at=last_advance,
                    cpu_seconds_observed=total,cpu_delta=None if old is None else total-old['cpu'],
                    no_activity_seconds=monotonic-last_activity,return_code=code)


class CheckPublisher:
    def __init__(self, label):
        self.repo = os.environ['GITHUB_REPOSITORY']
        self.run = int(os.environ['GITHUB_RUN_ID'])
        self.attempt = int(os.environ['GITHUB_RUN_ATTEMPT'])
        self.head = os.environ['GITHUB_SHA']
        self.external = f"horizon-progress:{self.run}:{self.attempt}:{label}"
        self.id = None

    def request(self, method, path, body):
        req = Request('https://api.github.com/repos/'+self.repo+'/'+path,
            data=json.dumps(body).encode(),method=method,
            headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],
                     'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'})
        with urlopen(req,timeout=25) as response:
            return json.load(response)

    def publish(self, snapshot, *, finished=False):
        output = dict(title='Frozen continuation: saved decisions and process activity',
                      summary='Progress only. CPU activity is not proof of decision quality; full artifacts and audits are required.',
                      text=json.dumps(snapshot,separators=(',',':')))
        body = dict(status='completed' if finished else 'in_progress',output=output)
        if finished:
            body.update(conclusion='neutral',completed_at=utc())
        if self.id is None:
            body.update(name='Horizon progress / '+self.external,head_sha=self.head,
                        external_id=self.external,started_at=utc(),
                        details_url=f'https://github.com/{self.repo}/actions/runs/{self.run}')
            self.id = self.request('POST','check-runs',body)['id']
        else:
            self.request('PATCH',f'check-runs/{self.id}',body)


def launch(index, pair, root):
    args=[sys.executable,'tools/resume_horizon_pair.py','--value-contract',
          '--root',str(root/f'pair{index}'),
          '--inputs',json.dumps([f'results/input/p{index}/a{s}' for s in range(2)]),
          '--jobs',json.dumps(pair['jobs']),
          '--manifests',json.dumps([x['manifest'] for x in pair['inputs']]),
          '--checkpoints',json.dumps([x['checkpoint'] for x in pair['inputs']])]
    if pair.get('recorded_predictions'):args+=['--recorded-predictions']
    if pair.get('guard_history'):args+=['--guard-history']
    return subprocess.Popen(args)


def main():
    pairs=json.loads(os.environ['PAIRS']); assert 1<=len(pairs)<=2
    root=Path('results/value-contract'); progress=Progress()
    publisher=CheckPublisher(pairs[0]['keys'][0])
    if '--finalize' in sys.argv:
        receipt=root/'telemetry_check.json'
        if not receipt.exists():return
        previous=read(receipt)
        assert previous['external_id']==publisher.external and previous['head_sha']==publisher.head
        publisher.id=previous['check_id']
        path=root/'progress.json'
        snapshot=read(path) if path.exists() else dict(previous,arms=[])
        snapshot.update(segment_ended=True,finalized_at=utc())
        publisher.publish(snapshot,finished=True)
        save(path,snapshot)
        return
    identity=dict(version='horizon-progress-v1',run=publisher.run,attempt=publisher.attempt,
                  head_sha=publisher.head,external_id=publisher.external)
    # Check permission/transport before starting computation. An ambiguous POST
    # is not automatically repeated; it cannot silently duplicate an experiment.
    initial=dict(identity,observed_at=utc(),arms=[],phase='starting')
    publisher.publish(initial)
    save(root/'telemetry_check.json',dict(check_id=publisher.id,**identity))
    processes=[launch(i,p,root) for i,p in enumerate(pairs)]
    while True:
        now=utc();table=process_table();arms=[]
        for i,pair in enumerate(pairs):
            for side,name in enumerate(('baseline','candidate')):
                directory=root/f'pair{i}'/name; key=pair['keys'][side]
                try:
                    cp=checkpoint_summary(directory,pair['jobs'][side],pair['inputs'][side]['manifest'])
                    status=read(directory/'execution_status.json') if (directory/'execution_status.json').exists() else {}
                    boundary=read(directory/'boundary_provenance.json') if (directory/'boundary_provenance.json').exists() else {}
                    stats=progress.observe(key,cp,status,table,now,time.monotonic())
                    arms.append(dict(key=key,checkpoint=cp,execution=status,**stats,
                        input_boundary=boundary.get('resume_at'),prior_turns=boundary.get('original_turns'),
                        segment_advance=None if cp is None else cp['ticks']-boundary['resume_at'],
                        new_experience=None if cp is None else max(0,cp['ticks']-boundary['original_turns'])))
                except Exception as error:
                    arms.append(dict(key=key,phase='observation_error',error=repr(error)))
        codes=[p.poll() for p in processes];finished=all(c is not None for c in codes)
        snapshot=dict(identity,observed_at=now,arms=arms,pair_return_codes=codes,finished=finished)
        save(root/'progress.json',snapshot)
        try:
            publisher.publish(snapshot,finished=finished)
        except Exception as error:
            # Model computation is independent of monitoring transport.
            print('Progress publication failed: '+type(error).__name__,flush=True)
            save(root/'telemetry_error.json',dict(at=utc(),error=type(error).__name__))
        if finished:
            if any(codes):raise SystemExit('Incomplete continuations retained with original source and experience')
            return
        time.sleep(60)


if __name__=='__main__':main()
