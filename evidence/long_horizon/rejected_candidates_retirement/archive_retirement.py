"""Preserve authorized rejected runs, without completing or adopting them."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil

local=Path(__file__).resolve().parent
repo=local.parents[2]/'reflex-persona-core'
source=local/'rejected_candidates_retirement_20261101'
target=repo/'evidence/long_horizon/rejected_candidates_retirement'
def read(p): return json.loads(p.read_bytes())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
operation=read(source/'operation_complete.json')
assert operation['operation_complete'] and operation['remaining_target_processes']==0
assert not operation['scientific_completion'] and not operation['files_deleted']
plan=read(source/'operation_plan.json')
assert plan['authorization']=='古い2候補を打ち切り、修正版へ集中'
assert operation['plan_sha256']==sha(source/'operation_plan.json')
target.mkdir(exist_ok=False)
rows={}
for key,name in [('purpose','purpose_prefixes'),('origin_gain','origin_gain_prefixes')]:
    proof=read(source/name/'verification.json')
    assert proof['audit_complete'] and proof['source_unchanged']
    assert not proof['study_complete'] and not proof['adoption']
    assert sha(source/name/'preserved_histories.zip')==proof['history_archive_sha256']
    shutil.copytree(source/name,target/name)
    rows[key]={k:proof[k] for k in ('source_ref','registered_executions','original_completed','retained_actual_steps','history_archive_sha256')}
    rows[key]['partial_executions']=sum(r['state']=='retired_saved_partial' for r in proof['rows'])
    rows[key]['not_started']=sum(r['state']=='not_started_no_history' for r in proof['rows'])
for name in ('operation_plan.json','operation_complete.json','operations.jsonl'):
    shutil.copy2(source/name,target/name)
for name in ('retire_rejected_studies.py','audit_retired_study.py','test_retirement_guard.py','snapshot_registered_queues.py','archive_retirement.py'):
    shutil.copy2(local/name,target/name)
proof=dict(operation_and_prefix_audits_complete=True,scientific_studies_complete=False,adoption=False,
    at=datetime.now(timezone.utc).isoformat(),authorization=plan['authorization'],studies=rows,
    stopped_processes=len(operation['stopped']),remaining_target_processes=0,
    original_files_preserved=True,relative_scientific_workers_untouched=True,
    gained='Compute is available to the corrected candidate and required independent confirmation.',
    forgone='Remaining whole-cohort and loss-diagnostic observations for two rejected candidates.',
    unchanged='Stable release/Rust, relative source, all relative conditions, persona, world RNG and acceptance gates.',
    artifacts={str(p.relative_to(target)).replace('\\','/'):sha(p) for p in target.rglob('*') if p.is_file()})
write(target/'verification.json',proof)
path=repo/'results/long_horizon_resume_status.json';state=read(path)
state['version']=5;state['updated']=proof['at']
state['retired_worlds']={}
for key in rows:
    previous=state['active_worlds'].pop(key)
    previous.update(status='rejected_retired_incomplete',restart=False,complete=False,adoption=False,
        summary=rows[key],evidence='evidence/long_horizon/rejected_candidates_retirement/verification.json')
    state['retired_worlds'][key]=previous
state['pending_user_decision']=dict(status='resolved',answer=plan['authorization'],executed=True,
    evidence='evidence/long_horizon/rejected_candidates_retirement/verification.json')
state['constraints']=[c.replace('dropped conditions,','dropped conditions in the surviving candidate,') for c in state['constraints']]
state['constraints'].append('Human authorized retirement of rejected Q/G on November1; do not resume those incomplete cohorts. Their evidence is preserved. All N gates and reserved40 remain mandatory.')
state['next_actions']=[
    'Only N relative study remains active. Never restart retired Q/G or their finishers; retained evidence is explicitly incomplete.',
    'Complete N ten fresh histories, whole20/2560 audit, every original loss/personality diagnostic, and existing post-audit package replay. Do not duplicate the ongoing chain.',
    'Full regression is complete; do not rerun without a source change or concrete new defect.',
    'Review all N gates. Only then prepare reserved40 through the strict registered bridge and qualified casewise diagnostic runner.',
    'If N fails, diagnose full losses before a common correction. No game-specific tuning, partial adoption or new features.']
state['operational']['compute_focus']['retired_candidates']='Q/G terminated with human authorization; priority helper only affects surviving N.'
write(path,state)
report=repo/'evidence/long_horizon/REPORT.md'
with report.open('a',encoding='utf8') as f:
    f.write('\n\n2026-11-01、利用者が「古い2候補を打ち切り、修正版へ集中」と明示したため、上記の先行2候補を全条件完走する指定は撤回された。目的評価候補と旧増分評価候補の管理・集計・計算19プロセスを登録と生成時刻で照合して停止した。修正版の4計算と後処理は継続し、登録・人格・乱数・採用基準は変えていない。\n\n目的評価候補は20実行中13完走・7途中、保存済み2,198実手。旧増分評価候補は20実行中1完走・8途中・11未着手、保存済み588実手。全保存履歴を各々の固定ソースで再生し、本人・学習・世界・乱数を監査した。元ファイルと完走／途中の区別を維持し、両候補とも「不採用・残り未完了」として残す。[許可・停止・全保存履歴の監査](rejected_candidates_retirement/verification.json)。\n\n得たのは修正版と必須の独立確認へ計算を集中できること。捨てたのは不採用2候補の残る全分野・悪化局面の追加診断であり、修正版の条件や検証基準ではない。修正版は全10組と全悪化・人格・配布物の確認を終え、既定基準を通過した場合だけ独立40条件へ進める。安定版とRustは変更していない。\n')
print(json.dumps(dict(archived=True,studies=rows),ensure_ascii=False))
