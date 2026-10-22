"""Read-only execution check: distinguish audited completion and saved progress.

Never dispatches, cancels, edits the ledger or changes a frozen experiment.
An Actions in_progress flag is not proof of resolved-decision advancement.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.resume_horizon_pair import inspect
from tools.value_contract_config import jobs


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SavedAuditCache:
    """Reuse a compact audit only while every audited input byte is unchanged.

    Old segments are immutable, but their owner timelines were being decoded
    and hashed every minute. Content hashes, including all frozen source files,
    are still checked on every visit. This is telemetry, not final acceptance.
    """
    def __init__(self):
        self.items={};self.lock=threading.Lock()

    def summary(self,directory,job):
        directory=Path(directory).resolve()
        paths=[directory/name for name in ('source_snapshot.json','checkpoint.json',
                                          'registration.json','contract_pair.json')]
        hashes=tuple(sha(path) for path in paths)
        source=directory/'_source'
        for name,fingerprint in read(paths[0]).items():
            target=(source/name).resolve()
            assert target.is_relative_to(source) and target.is_file(),'invalid frozen source path'
            assert sha(target)==fingerprint,('source changed',name)
        key=(hashes,tuple(job))
        with self.lock:cached=self.items.get(directory)
        if cached is not None and cached[0]==key:
            return json.loads(cached[1])
        verified=inspect(directory,hashes[0],hashes[1],job,pair_file='contract_pair.json')
        assert tuple(sha(path) for path in paths)==hashes,'audit inputs changed during inspection'
        payload=verified['saved']
        result=dict(pair=verified['pair'],ticks=len(payload['trace']),
                    elapsed=payload['elapsed'],checkpoint_sha256=hashes[1])
        with self.lock:
            if len(self.items)>=512 and directory not in self.items:self.items.pop(next(iter(self.items)))
            self.items[directory]=(key,json.dumps(result))
        return result


SAVED_AUDITS=SavedAuditCache()


def api(path):
    return json.loads(subprocess.check_output(
        ['gh', 'api', 'repos/iwmn-9/reflex-persona-core/' + path], cwd=ROOT,timeout=60))


def live_progress(checks, task, run):
    prefix=f"horizon-progress:{task['run']}:{run['run_attempt']}:"
    selected=[c for c in checks if (c.get('external_id') or '').startswith(prefix)]
    matches=[]
    for check in selected:
        assert check['head_sha']==task['head'] and check['app']['slug']=='github-actions'
        value=json.loads(check['output']['text'])
        assert value['version']=='horizon-progress-v1'
        assert value['run']==task['run'] and value['attempt']==run['run_attempt']
        assert value['head_sha']==task['head'] and value['external_id']==check['external_id']
        arms=[a for a in value['arms'] if a['key'] in (task['key']+'-baseline',task['key']+'-candidate')]
        if not arms:continue
        assert len(arms)==2 and len({a['key'] for a in arms})==2
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(value['observed_at'])).total_seconds()
        matches.append(dict(check_id=check['id'],url=check['html_url'],observed_at=value['observed_at'],
                            age_seconds=age,stale=age>300,arms=arms,
                            pair_return_codes=value.get('pair_return_codes',[]),
                            segment_outcome=value.get('segment_outcome'),
                            segment_ended=value.get('segment_ended',False)))
    assert len(matches)<=1, 'ambiguous live monitor'
    return matches[0] if matches else None


def check(task, checks_by_head):
    run = api(f"actions/runs/{task['run']}")
    assert run['head_sha'] == task['head'], 'remote source changed'
    all_jobs = api(f"actions/runs/{task['run']}/jobs?per_page=100")['jobs']
    remote = [j for j in all_jobs if j['name'].startswith((f"study ({task['key']},",f"study ({task['key']}-baseline,"))]
    assert len(remote) <= 1
    job = remote[0] if remote else None
    checks=checks_by_head[task['head']]
    live=live_progress(checks,task,run)
    plan_path = ROOT / task['plan']
    plan = read(plan_path)
    spec = next(c for c in plan['cases'] if c['key'] == task['key'])
    expected_jobs = jobs(spec, plan)
    saved = []
    for segment in [*task['history'], task]:
        pair = []
        for side, expected_job in expected_jobs.items():
            directory = ROOT / segment['root'] / f"value-contract-{task['key']}-{side}"
            checkpoint = directory / 'checkpoint.json'
            if not checkpoint.is_file():
                break
            verified = SAVED_AUDITS.summary(directory,expected_job)
            assert verified['pair']['plan_sha256'] == sha(plan_path)
            assert verified['pair']['sources'] == plan['sources']
            ticks = verified['ticks']
            pair.append(dict(side=side, path=checkpoint.relative_to(ROOT).as_posix(),
                             sha256=verified['checkpoint_sha256'], ticks=ticks,
                             elapsed_minutes=verified['elapsed']/60,
                             cumulative_minutes_per_decision=verified['elapsed']/60/ticks))
        if len(pair) == 2:
            saved.append(dict(run=segment['run'], common_boundary=min(p['ticks'] for p in pair),
                              checkpoints=pair))
    assert saved, 'no validated input checkpoints'
    prior = task['history'][-1]['boundary'] if task['history'] else 0
    current = next((s for s in saved if s['run'] == task['run']), None)
    collected_advance=None
    if len(saved)>=2:
        collected_advance=dict(from_tick=saved[-2]['common_boundary'],to_tick=saved[-1]['common_boundary'],
                               added=saved[-1]['common_boundary']-saved[-2]['common_boundary'])
    if live:
        for arm in live['arms']:
            cp=arm.get('checkpoint')
            if cp:
                side='baseline' if arm['key'].endswith('-baseline') else 'candidate'
                reference=next(c for c in saved[-1]['checkpoints'] if c['side']==side)
                reference_root=ROOT/reference['path']
                assert cp['source_manifest_sha256']==sha(reference_root.parent/'source_snapshot.json')
                assert prior<=cp['ticks']<=spec['job'][4]
    steps = job['steps'] if job else []
    active = [s for s in steps if s['status'] == 'in_progress']
    resume = next((s for s in steps if s['name'] == "Resume at each pair's common experience boundary"), None)
    deadline = None
    if resume and resume['started_at']:
        deadline = (datetime.fromisoformat(resume['started_at'].replace('Z', '+00:00'))
                    + timedelta(minutes=345)).isoformat()
    return dict(key=task['key'], plan=task['plan'], run=task['run'],
                url=run['html_url'], head_sha=run['head_sha'], attempt=run['run_attempt'],
                remote_status=run['status'], conclusion=run['conclusion'],
                job_status=job['status'] if job else None, job_conclusion=job['conclusion'] if job else None, active_steps=active,
                step_limit_at_utc=deadline,
                input_common_boundary=prior, requested_turns=spec['job'][4],
                last_verified_saved=saved[-1],
                current_run_saved_progress=current,
                current_run_resolved_advance=None if current is None else current['common_boundary']-prior,
                latest_collected_advance=collected_advance,
                live=live,
                proof_scope='Checksum, frozen source, registered job, owner timeline and world RNG of saved checkpoints; not final quality/audit acceptance.',
                live_progress='unknown_until_artifact' if current is None else 'saved_boundary_verified')


def display(output):
    lines=['# 長期検証の実行状態','',f"取得時刻: {output['checked_at']}",'',
           '完走数は監査済みの組数。途中手数・CPU稼働は採用の証拠には数えない。','']
    names={'confirmation':'独立確認','diagnostic':'従来診断','future_replanning':'新診断'}
    for name,value in output['counts'].items():
        lines.append(f"- {names.get(name,name)}: {value['completed_conditions']}/{value['registered_conditions']}")
    lines+=['','| 条件 | 回収・照合済み手数 | 回収で確認した前進 | 現在のクラウド報告（基準 / 変更） | 現区間で前進を観測した時刻 | 状態 |',
            '|---|---:|---|---|---|---|']
    for row in output['pending']:
        live=row['live'];tick='未取得';last='未確認';phase='実行中の内部進捗は未確認'
        if live:
            def arm_label(a):
                cp=a.get('checkpoint')
                return str(cp['ticks']) if cp else '?'
            arms=sorted(live['arms'],key=lambda a:a['key'])
            tick=' / '.join(arm_label(a) for a in arms)
            last=' / '.join(a.get('last_observed_advance_at') or '初回観測後の増加なし' for a in arms)
            phase=' / '.join(a['phase'] for a in arms)
            if any(c is not None and c!=0 for c in live['pair_return_codes']):phase='実行・照合処理に失敗: '+phase
            if live.get('segment_outcome')=='handoff_pending':phase='引き継ぎ待ち（保存検証済み・実験未完走）'
            if live.get('segment_outcome')=='failure':phase='実行・監査・保存検証の失敗: '+phase
            if live['stale']:phase='監視情報が5分以上古い: '+phase
            if live['segment_ended']:phase='区間終了: '+phase
        if row.get('job_conclusion') in ('failure','timed_out','cancelled') or row.get('conclusion') in ('failure','timed_out','cancelled'):
            phase='GitHub実行失敗（保存・アップロードの確認が必要）: '+phase
        advance=row['latest_collected_advance']
        change=f"{advance['from_tick']}→{advance['to_tick']} (+{advance['added']})" if advance else '今回の再開後は未回収'
        lines.append(f"| {row['key']} | {row['last_verified_saved']['common_boundary']}/{row['requested_turns']} | {change} | {tick} | {last} | {phase} |")
    lines+=['','`computing_no_new_checkpoint`: CPU使用は増加、保存手数はまだ同じ。',
            '`suspected_stall_no_cpu_or_checkpoint_15m`: CPUと保存手数が15分増えず、要調査。自動で実験を捨てない。',
            '`failed` / `observation_error`: 実行失敗 / 観測失敗。完走・品質改善とは区別する。','']
    lines+=['`pause_awaiting_validation`: 時間枠による停止。保存検証が終わるまで正常な引き継ぎとは認定しない。',
            '`handoff_pending`: 保存検証済みの引き継ぎ待ち。GitHubの区間処理が成功でも、実験は未完走。','']
    lines+=['監視情報は約60秒ごとに公開・取得。凍結済み実験の保存間隔は4手ごとのままなので、計算中でも保存手数はしばらく変わらない。','']
    (ROOT/'results/registered_execution_progress.md').write_text('\n'.join(lines),encoding='utf8')


def main():
    state = read(ROOT/'results/registered_world_completion/state.json')
    pending = [t for t in state['tasks'] if t['status'] != 'complete']
    checks_by_head={}
    for head in {t['head'] for t in pending}:
        checks_by_head[head]=[];page=1
        while True:
            checks=api(f'commits/{head}/check-runs?per_page=100&filter=all&page={page}')['check_runs']
            checks_by_head[head].extend(checks)
            if len(checks)<100:break
            page+=1
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda t:check(t,checks_by_head), pending))
    status = read(ROOT/'results/registered_evidence_status.json')
    output = dict(checked_at=datetime.now(timezone.utc).isoformat(),
                  ledger_updated=state['updated'], supervisor_error=state.get('error'),
                  counts={k:{f:v[f] for f in ('complete','completed_conditions','registered_conditions')} for k,v in status.items()},
                  pending=rows,
                  caveat='The latest running segment has no externally readable checkpoints until artifact upload. Do not infer live progress from in_progress or from an older run checkpoint.')
    target = ROOT/'results/registered_execution_progress.json'
    temp = target.with_suffix('.new')
    temp.write_text(json.dumps(output,indent=2)+'\n',encoding='utf8')
    temp.replace(target)
    display(output)
    print(json.dumps({'checked_at':output['checked_at'],'counts':output['counts'],
                      'pending':[{'key':r['key'],'saved':r['last_verified_saved']['common_boundary'],
                                  'current_advance':r['current_run_resolved_advance'],
                                  'status':r['remote_status']} for r in rows]}))
    return all(v['complete'] for v in output['counts'].values())


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    while True:
        try:
            complete=main()
            save_error=None
        except Exception as error:
            save_error=dict(at=datetime.now(timezone.utc).isoformat(),error=repr(error))
            complete=False
            if not args.watch:raise
            print(json.dumps(save_error),flush=True)
            report=ROOT/'results/registered_execution_progress.md'
            if report.exists():
                old=report.read_text(encoding='utf8')
                start=old.find('# 長期検証の実行状態')
                report.write_text('監視の更新に失敗: '+save_error['at']+'。以下は最後に取得できた状態。\n\n'+old[start:],encoding='utf8')
        (ROOT/'results/registered_execution_progress_error.json').write_text(json.dumps(save_error)+'\n',encoding='utf8')
        if not args.watch or complete:break
        time.sleep(60)
