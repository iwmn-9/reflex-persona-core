"""Finish only registered worlds; preserve failures and resume verified inputs.

This is a bounded execution supervisor, not an optimizer or a test selector.
It cannot change playing source, conditions, learning history, or acceptance.
Unknown failures stop for investigation. An audit-only retry still uses the
existing <=8 ULP forecast check and exact owner/physics/RNG verification.
"""
import hashlib,json,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from tools.summarize_value_contract import read_run
from tools.value_contract_config import jobs
from tools.horizon_segment import verify_handoff,handoff_facts,RECEIPT,PAUSED

ROOT=R/'results/registered_world_completion';ROOT.mkdir(exist_ok=True)
LEDGER=ROOT/'state.json'
COLLECT=R/'tools/collect_value_contract.py'

def save(state):
    state['updated']=datetime.now(timezone.utc).isoformat()
    tmp=LEDGER.with_suffix('.new');tmp.write_text(json.dumps(state,indent=2)+'\n',encoding='utf8');tmp.replace(LEDGER)

def api(path):
    # GET only. Retry transient transport/server failures, never dispatches or
    # simulation/audit failures. An exhausted retry still stops for inspection.
    command=['gh','api','repos/iwmn-9/reflex-persona-core/'+path]
    for attempt in range(5):
        try:
            response=subprocess.run(command,cwd=R,encoding='utf8',capture_output=True,timeout=60)
        except subprocess.TimeoutExpired:
            if attempt==4:raise
            reason='GET timeout'
        else:
            if response.returncode==0:return json.loads(response.stdout)
            reason=response.stderr.strip()
            transient_http=any(f'HTTP {code}' in reason for code in (429,500,502,503,504))
            transient_connection='error connecting to api.github.com' in reason.lower()
            if attempt==4 or not (transient_http or transient_connection):
                raise subprocess.CalledProcessError(response.returncode,command,response.stdout,response.stderr)
        print(json.dumps(dict(read_retry=path,attempt=attempt+1,reason=reason)),flush=True)
        time.sleep(min(15*(attempt+1),60))

def invoke(script,args,log,allow_failure=False):
    with log.open('w',encoding='utf8') as f:
        result=subprocess.run([sys.executable,str(script),*map(str,args)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
    if not allow_failure:result.check_returncode()
    return result.returncode

def initialize():
    tasks=[]
    for run,root,plan,exclude in (
        (39404741708,'results/long_horizon_sampling_check_confirmation_continuation','evidence/long_horizon/sampling_check_confirmation_plan.json',{'combat-0-56102-shift-return'}),
        (39404634553,'results/long_horizon_sampling_check_diagnostic_continuation','evidence/long_horizon/sampling_check_diagnostic_plan.json',set()),
        (39412321707,'results/long_horizon_tail_replanning_diagnostic','evidence/long_horizon/tail_replanning_diagnostic_plan.json',set()),
    ):
        meta=json.loads((R/root/'workflow.json').read_text())
        for key in meta['collected_case_keys']:
            if key not in exclude:
                tasks.append(dict(key=key,run=run,root=root,plan=plan,head=meta['head_sha'],external_collector=True,
                                  history=[],status='active'))
    return dict(version=1,tasks=tasks,complete=False,
                excluded_resolved={'combat-0-56102-shift-return':'results/long_horizon_sampling_check_confirmation_audit'},
                rules='Only observed 345-minute timeout with increasing resolved boundary, or one audit-only exact-replay recovery. All conditions remain required. No candidate acceptance or default change.')

state=json.loads(LEDGER.read_text()) if LEDGER.exists() else initialize()
save(state)
try:
    while any(t['status']!='complete' for t in state['tasks']):
        active=[t for t in state['tasks'] if t['status']!='complete']
        run_jobs={run:api(f'actions/runs/{run}/jobs?per_page=100')['jobs'] for run in {t['run'] for t in active}}
        changed=False
        for task in active:
            key=task['key'];root=R/task['root']
            found=[j for j in run_jobs[task['run']] if j['name'].startswith((f'study ({key},',f'study ({key}-baseline,'))]
            if not found:continue  # matrix may still be queued
            assert len(found)==1
            job=found[0]
            if job['status']!='completed':continue
            dirs=[root/f'value-contract-{key}-{side}' for side in ('baseline','candidate')]
            if not all((d/'contract_pair.json').exists() for d in dirs):
                if task['external_collector']:continue
                invoke(COLLECT,['--run',task['run'],'--sha',task['head'],'--root',task['root'],
                    '--plan',task['plan'],'--keys',key,'--once'],ROOT/f"collect-{task['run']}.log",allow_failure=True)
                assert all((d/'contract_pair.json').exists() for d in dirs),'completed job has missing evidence'
            plan_path=R/task['plan'];plan=json.loads(plan_path.read_text())
            spec=next(s for s in plan['cases'] if s['key']==key)
            proofs=[json.loads((d/'contract_pair.json').read_text()) for d in dirs]
            assert proofs[0]==proofs[1]
            if proofs[0]['complete']:
                assert job['conclusion']=='success'
                checks=[read_run(d,spec['job'],bool(i),plan,hashlib.sha256(plan_path.read_bytes()).hexdigest())
                        for i,d in enumerate(dirs)]
                assert checks[0][1]==checks[1][1]
                task.update(status='complete',evidence=[c[2] for c in checks],
                            scores=[c[0]['score'] for c in checks])
                print(json.dumps(dict(complete=key,plan=task['plan'],scores=task['scores'])),flush=True)
                changed=True;save(state);continue
            handoff=None
            if any((d/RECEIPT).exists() for d in dirs):
                assert job['conclusion']=='success','handoff upload or workflow failed'
                pair_jobs=jobs(spec,plan)
                handoff=verify_handoff(dirs,[pair_jobs[s] for s in ('baseline','candidate')],run=task['run'],
                    plan_sha=hashlib.sha256(plan_path.read_bytes()).hexdigest(),sources=plan['sources'])
            else:assert job['conclusion']!='success','successful job has incomplete proof without verified handoff'
            log=ROOT/f"{job['id']}.log"
            if not handoff and not log.exists():
                log.write_bytes(subprocess.check_output(['gh','api','--allow-escape-sequences',
                    f"repos/iwmn-9/reflex-persona-core/actions/jobs/{job['id']}/logs"],cwd=R))
            text=b'' if handoff else log.read_bytes()
            audit=b'AssertionError: stored actual action/feedback replay drift' in text
            timeout=b'has timed out after 345 minutes' in text
            failed_pause=False
            if audit and PAUSED in (proofs[0].get('return_codes') or []):
                pair_jobs=jobs(spec,plan)
                handoff_facts(dirs,[pair_jobs[s] for s in ('baseline','candidate')],failed_peer=True)
                failed_pause=True  # Workflow remains failed; only its paused peer is verified.
            assert handoff or audit or timeout, f'Unknown failure in {key}; inspect {log}'
            audit_only=audit and not (timeout or failed_pause)
            guarded=bool(handoff) or timeout or failed_pause or proofs[0].get('retained_history_required',False)
            assert len(task['history'])<16,'unexpectedly many continuations need investigation'
            if audit_only:assert not any(h['audit_only'] for h in task['history']),'audit retry failed; investigate rather than relax'
            prefix=f"results/rwc/p-{task['run']}-{hashlib.sha256(key.encode()).hexdigest()[:8]}-"+('h' if handoff else 'm' if audit and (timeout or failed_pause) else 'a' if audit else 'c')
            prep=R/prefix
            if not (prep/'dispatch.json').exists():
                if not (prep/'preparation.json').exists():
                    args=['--run',task['run'],'--root',task['root'],'--plan',task['plan'],'--out',prefix,'--keys',key,
                          '--reason','Verified planned handoff; experiment remains incomplete.' if handoff else 'Observed exact replay mismatch after completion.' if audit else 'Observed 345-minute execution limit; retain original source, owner, world and RNG.']
                    if handoff:args+=['--handoff']
                    if failed_pause:args+=['--failed-pause']
                    if audit:args+=['--recorded-predictions']
                    if guarded:args+=['--guard-history']
                    invoke(R/'tools/prepare_active_value_continuation.py',args,ROOT/f"prepare-{task['run']}-{key}.log")
                prepared=json.loads((prep/'preparation.json').read_text())
                assert len(prepared['pairs'])==1
                boundary=prepared['pairs'][0]['common_boundary']
                previous=task['history'][-1]['boundary'] if task['history'] else 0
                assert boundary>previous or audit_only,'no resolved-world progress; investigate'
                target=prefix+'-result'
                invoke(R/'tools/dispatch_registered_continuation.py',
                       ['--prepare',prefix,'--label',f"finish-{task['run']}-{key}",'--root',target],
                       ROOT/f"dispatch-{task['run']}-{key}.log")
            dispatch=json.loads((prep/'dispatch.json').read_text())
            assert dispatch.get('outcome')=='submitted' and dispatch.get('run'),'inspect unknown dispatch before retry'
            prepared=json.loads((prep/'preparation.json').read_text())
            task['history'].append(dict(run=task['run'],root=task['root'],boundary=prepared['pairs'][0]['common_boundary'],
                                        audit_only=audit_only,guard_history=guarded,mixed_failure=audit and (timeout or failed_pause),planned_handoff=bool(handoff),job=job['id']))
            task.update(run=dispatch['run'],root=dispatch['root'],head=dispatch['workflow_sha'],external_collector=False)
            print(json.dumps(dict(resumed=key,run=task['run'],audit_only=audit_only,
                mixed_failure=audit and timeout,boundary=task['history'][-1]['boundary'])),flush=True)
            changed=True;save(state)
        if not changed:save(state)
        if any(t['status']!='complete' for t in state['tasks']):time.sleep(45)
    state['complete']=True;save(state);print('All supervised registered pairs completed and audited.',flush=True)
except Exception as error:
    state['error']=repr(error);save(state);raise
