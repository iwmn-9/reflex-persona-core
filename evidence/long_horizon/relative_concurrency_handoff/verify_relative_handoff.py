"""Audit the control-only handoff, including the rejected first priority helper."""
import copy, json, shutil
from pathlib import Path
from datetime import datetime, timezone
from handoff_relative_coordinator import local, source, out, new_finish, old_finish, new_post, old_post, manifest, processes, verify_files
from snapshot_registered_queues import match_workers
from finish_relative_study import read, sha, write

receipt=read(out/'receipt.json')
assert receipt['state']=='post_started'
shutil.copy2(out/'receipt.json',out/'before_helper_recovery.json')
old=read(old_finish/'manifest.json');new=read(new_finish/'manifest.json')
expected=copy.deepcopy(old);expected['parent_pid']=receipt['coordinator']
assert new==expected
prior_post=read(old_post/'manifest.json');current_post=read(new_post/'manifest.json')
assert current_post['steps']==prior_post['steps'] and current_post['arithmetic']==prior_post['arithmetic']
assert all(current_post['files'].get(k)==v for k,v in prior_post['files'].items())
assert current_post['completed_regression_sha256']==sha(current_post['regression_status'])
verify_files(new['files']);verify_files(current_post['files'])
registration=read(source/'evidence/relative_gain/worlds/registration.json')
verify_files({str(source/name):checksum for name,checksum in registration['sources'].items()})
rows=processes();byid={row['ProcessId']:row for row in rows}
workers=match_workers(rows,manifest)
assert set(workers)==set(range(8)) and all(len(x)==1 for x in workers.values())
for index,previous in receipt['retained_workers'].items():assert workers[int(index)]==previous
for pid in (receipt['coordinator'],receipt['finisher'],receipt['post'],17660):assert pid in byid
assert 26372 not in byid
assert 'prioritize_relative_surviving.ps1' in byid[17660]['CommandLine']
assert (out/'priority_fixed.stderr.log').stat().st_size==0
assert b'Coordinator identity changed' in (out/'priority.stderr.log').read_bytes()
receipt.update(state='complete',at=datetime.now(timezone.utc).isoformat(),active_workers=workers,
    priority_helper=17660,initial_priority_helper=26372,
    repaired_issue='CIM DateTime vs parsed DateTimeOffset compared as different types. UTC instants match; corrected check, same workers retained.',
    finisher_manifest_sha256=sha(new_finish/'manifest.json'),post_manifest_sha256=sha(new_post/'manifest.json'),
    source_files_verified=len(registration['sources']),regression_reused_without_rerun=True,
    subsequent_stages_unchanged=True)
write(out/'receipt.json',receipt)
repo=local.parents[2]/'reflex-persona-core';target=repo/'evidence/long_horizon/relative_concurrency_handoff'
target.mkdir(exist_ok=False)
for path in out.iterdir():
    if path.is_file() and path.name not in ('coordinator.stdout.log','coordinator.stderr.log','priority_fixed.stdout.log','priority_fixed.stderr.log'):
        shutil.copy2(path,target/path.name)
for name in ('handoff_relative_coordinator.py','finish_relative_post_resumed.py','test_relative_post_resume.py','prioritize_relative_surviving.ps1','verify_relative_handoff.py'):
    shutil.copy2(local/name,target/name)
for name,folder in [('old_finisher',old_finish),('new_finisher',new_finish),('old_post',old_post),('new_post',new_post)]:
    shutil.copy2(folder/'manifest.json',target/(name+'_manifest.json'))
write(target/'verification.json',dict(operation_complete=True,scientific_completion=False,adoption=False,
    source_files_verified=len(registration['sources']),existing_four_workers_retained=True,total_workers=8,
    unchanged_stages=True,all_original_post_files_bound=True,completed_regression_reused=True,
    first_helper_failed_safely_then_repaired=True,
    limits='Only scheduling/control handoff. No quality result, new random seeds, persona changes, or eliminated N conditions.',
    artifacts={p.name:sha(p) for p in target.iterdir() if p.is_file()}))
path=repo/'results/long_horizon_resume_status.json';state=read(path);state['updated']=receipt['at']
n=state['active_worlds']['relative'];n.update(coordinator_pid=receipt['coordinator'],workers=8)
n['finisher'].update(pid=receipt['finisher'],root=str(new_finish),manifest_sha256=receipt['finisher_manifest_sha256'])
n['priority']=dict(helper_pid=17660,script=str(local/'prioritize_relative_surviving.ps1'),root=str(out),
    scope='All eight registered workers Normal; no affinity or AboveNormal change. Finite helper ends when coordinator exits.')
state['relative_integration']['post_audit'].update(pid=receipt['post'],root=str(new_post),
    dependencies=[receipt['finisher']],manifest_sha256=receipt['post_manifest_sha256'],
    completed_regression_status_sha256=current_post['completed_regression_sha256'],
    note='Original regression9856 completed; hash-bound passed receipt replaces waiting on its already exited PID. Same package/history steps and180 prior bound files preserved.')
state['operational']['compute_focus']=dict(state='superseded',previous_helper_pid=26196,
    helper_pid=17660,all_surviving_workers_normal=True,evidence='evidence/long_horizon/relative_concurrency_handoff/verification.json')
state['operational']['cpu_fairness']['current_state']='Historical helpers21496/1836/26196 are intentionally retired; do not restart.'
state['next_actions'][0]='Only N relative study remains active: coordinator13760,finisher10132,post21948,priority17660,eight workers including four inherited from27860. Never restart Q/G or old controls/helpers.'
write(path,state)
with (repo/'evidence/long_horizon/REPORT.md').open('a',encoding='utf8') as f:
    f.write('\n\n空いたCPUへ修正版の登録済み実行を回すため、計算中の4プロセスを生成時刻ごと維持し、管理だけを引き継いで8並列にした。新しい条件や乱数は増やしていない。全612ソース、完走後の全履歴監査・全悪化診断・集計、配布物での全20履歴再生を照合した。終了済み回帰は同一の成功記録をハッシュで固定し、再実行しない。優先度補正の初回は日時型比較の不備で変更前に停止したため、UTC時刻の一致を確認する形へ直し、8計算を通常優先度にした。[引継ぎ・不備・復旧・同一性の記録](relative_concurrency_handoff/verification.json)。得たのは修正版へ同時に使える計算枠で、長期成績や8倍の速度を主張するものではない。\n')
print(json.dumps(dict(verified=True,workers=8,retained=4,coordinator=receipt['coordinator'],finisher=receipt['finisher'],post=receipt['post'])))
