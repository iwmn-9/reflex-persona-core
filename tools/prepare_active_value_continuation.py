"""Prepare selected finished single-pair jobs while other registered jobs run.

Operational helper only: no dispatch, no reset, no modification of inputs.
Unlike the whole-run helper, explicitly selected cases must each identify a
completed study job. Missing/lost-host cases are not silently skipped.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from tools.resume_horizon_pair import inspect
from tools.value_contract_config import jobs
from tools.horizon_segment import verify_handoff,handoff_facts


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def api(path):
    return json.loads(subprocess.check_output(
        ['gh', 'api', 'repos/iwmn-9/reflex-persona-core/' + path],
        cwd=R, encoding='utf8'))


def main():
    parser = argparse.ArgumentParser()
    for key in ('run', 'root', 'plan', 'out', 'reason'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--keys', nargs='+', required=True)
    parser.add_argument('--recorded-predictions', action='store_true',
                        help='Audit-only numerical replay after an observed exact owner replay failure.')
    parser.add_argument('--guard-history', action='store_true')
    parser.add_argument('--handoff', action='store_true')
    parser.add_argument('--failed-pause', action='store_true')
    args = parser.parse_args()
    assert not (args.handoff and (args.failed_pause or args.recorded_predictions))
    root, out, plan_path = R / args.root, R / args.out, R / args.plan
    assert not out.exists(), 'fresh preparation directory required'
    plan = json.loads(plan_path.read_text())
    specs = {s['key']: s for s in plan['cases']}
    assert len(set(args.keys)) == len(args.keys) and set(args.keys) <= set(specs)
    state = api(f'actions/runs/{args.run}')
    assert state['head_sha'] == json.loads((root / 'workflow.json').read_text())['head_sha']
    job_rows = api(f'actions/runs/{args.run}/jobs?per_page=100')['jobs']
    artifacts, page = [], 1
    while True:
        batch = api(f'actions/runs/{args.run}/artifacts?per_page=100&page={page}')['artifacts']
        artifacts.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    names = {a['name']: a for a in artifacts}
    selected_jobs, records, batches, already_complete = [], [], [], []
    for key in args.keys:
        matching = [j for j in job_rows if j['name'].startswith(
            (f'study ({key},', f'study ({key}-baseline,'))]
        assert len(matching) == 1 and matching[0]['status'] == 'completed', (key, 'case job must be finished')
        job = matching[0]
        selected_jobs.append({k: job[k] for k in ('id', 'name', 'status', 'conclusion', 'started_at', 'completed_at')})
        spec = specs[key]
        dirs = [root / f'value-contract-{key}-{side}' for side in ('baseline', 'candidate')]
        assert all(d.exists() and d.name in names and not names[d.name]['expired'] for d in dirs), (key, 'both immutable artifacts required')
        proof = [json.loads((d / 'contract_pair.json').read_text()) for d in dirs]
        assert proof[0] == proof[1]
        proof = proof[0]
        assert proof['plan_sha256'] == sha(plan_path) and proof['sources'] == plan['sources']
        assert proof['job'] == spec['job']
        if proof['complete']:
            already_complete.append(key)
            continue
        side_jobs = jobs(spec, plan)
        if args.handoff:
            assert job['conclusion']=='success'
            verify_handoff(dirs,[side_jobs[s] for s in ('baseline','candidate')],run=args.run,plan_sha=sha(plan_path),sources=plan['sources'])
        else:assert job['conclusion'] != 'success', (key, 'incomplete artifacts in successful job')
        if args.failed_pause:
            assert args.recorded_predictions and args.guard_history
            handoff_facts(dirs,[side_jobs[s] for s in ('baseline','candidate')],failed_peer=True)
        inputs, inspected, keys = [], [], []
        for side, directory in zip(('baseline', 'candidate'), dirs):
            manifest = sha(directory / 'source_snapshot.json')
            checkpoint = sha(directory / 'checkpoint.json')
            inspected.append(inspect(directory, manifest, checkpoint, side_jobs[side], pair_file='contract_pair.json'))
            keys.append(key + '-' + side)
            inputs.append(dict(run=int(args.run), artifact=directory.name, manifest=manifest, checkpoint=checkpoint))
        boundary = min(len(x['saved']['trace']) for x in inspected)
        assert 0 < boundary <= spec['job'][4]
        pair=dict(keys=keys, jobs=[side_jobs[s] for s in ('baseline', 'candidate')], inputs=inputs)
        if proof.get('recorded_prediction_replay_permitted'):
            pair['recorded_predictions']=True
        if args.guard_history:pair['guard_history']=True
        if args.recorded_predictions:
            assert boundary==spec['job'][4] or (
                args.guard_history and max(len(x['saved']['trace']) for x in inspected)==spec['job'][4]
            ), 'mixed replay requires a completed arm and guarded retained history'
            pair['recorded_predictions']=True
        batches.append(dict(pairs=[pair]))
        records.append(dict(key=key, job_id=job['id'], turns=[len(x['saved']['trace']) for x in inspected], common_boundary=boundary))
    result = dict(created=datetime.now(timezone.utc).isoformat(), original_run=int(args.run),
                  original_head=state['head_sha'], parent_status=state['status'], plan=args.plan,
                  plan_sha256=sha(plan_path), reason=args.reason, selected_jobs=selected_jobs,
                  pairs=records, already_complete=already_complete,
                  semantics='Only explicitly selected completed jobs. Original frozen source, owner, world and RNG retained; common resolved boundary. Other registered cases remain required.')
    if batches:
        logs = {} if args.handoff else {j['id']: subprocess.check_output(['gh', 'api', '--allow-escape-sequences',
                f'repos/iwmn-9/reflex-persona-core/actions/jobs/{j["id"]}/logs'], cwd=R)
                for j in selected_jobs if any(r['job_id'] == j['id'] for r in records)}
        if args.recorded_predictions:
            assert all(b'stored actual action/feedback replay drift' in log for log in logs.values())
            if boundary<spec['job'][4]:
                assert args.guard_history and (args.failed_pause or all(b'has timed out after 345 minutes' in log for log in logs.values()))
            result['recorded_predictions']='Observed strict replay failure; unchanged eight-ULP probability check and exact owner/physics audits. If paired with timeout, all retained actions are freshly chosen and must exactly reproduce the original actions and learner states.'
        result['guard_history']=args.guard_history
        out.mkdir()
        for job_id, data in logs.items():
            (out / f'{job_id}.log').write_bytes(data)
        (out / 'batches.json').write_text(json.dumps(batches, indent=2) + '\n', encoding='utf8')
        (out / 'preparation.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
