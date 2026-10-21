"""Retain every planned value-contract artifact, including failed executions."""
import argparse,hashlib,json,subprocess,sys,time,uuid
from pathlib import Path
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R))
from tools.resume_horizon_pair import verify_segmented_result
from tools.value_contract_config import configurations,jobs
from tools.horizon_segment import verify_handoff
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--sha',required=True)
p.add_argument('--root',default='results/long_horizon_cumulative_diagnostic')
p.add_argument('--keys',nargs='+');p.add_argument('--plan',default='evidence/long_horizon/cumulative_diagnostic_plan.json');p.add_argument('--once',action='store_true');a=p.parse_args()
plan_path=R/a.plan;plan=json.loads(plan_path.read_text());root=R/a.root;root.mkdir(parents=True,exist_ok=True)
selected=plan['cases']
if a.keys is not None:
    assert len(set(a.keys))==len(a.keys) and set(a.keys)<={s['key'] for s in selected}
    selected=[s for s in selected if s['key'] in a.keys]
expected={f'value-contract-{s["key"]}-{arm}':(s,arm) for s in selected for arm in ('baseline','candidate')}

def api(path):
    delays=(5,10,20,40,45)
    for attempt in range(len(delays)+1):
        try:
            raw=subprocess.check_output(['gh','api','repos/iwmn-9/reflex-persona-core/'+path],cwd=R,stderr=subprocess.PIPE)
            return json.loads(raw)
        except subprocess.CalledProcessError as error:
            message=(error.stderr or b'').decode('utf8',errors='replace')
            transient=any(word in message.lower() for word in ('connectex','timed out','timeout','connection reset','temporary failure','http 502','http 503','http 504','tls handshake'))
            if not transient or attempt==len(delays):
                print(message,file=sys.stderr,flush=True)
                raise
            print(json.dumps(dict(collection_network_retry=attempt+1,wait_seconds=delays[attempt],api=path)),flush=True)
            time.sleep(delays[attempt])

def verify(target,spec,arm):
    proof=json.loads((target/'contract_pair.json').read_text())
    assert proof['version'] in ('same-host-value-contract-v1','segmented-same-host-value-contract-v1') and proof['pair_id']
    assert proof['job']==spec['job'] and proof['sources']==plan['sources']
    assert proof.get('jobs',dict(baseline=spec['job'],candidate=spec['job']))==jobs(spec,plan)
    assert proof['plan_sha256']==hashlib.sha256(plan_path.read_bytes()).hexdigest()
    assert proof['configurations']==configurations(plan)
    if not (target/'source_snapshot.json').exists():return dict(complete=False,source_started=False)
    source=json.loads((target/'source_snapshot.json').read_text())
    for name,sha in source.items():assert hashlib.sha256((target/'_source'/name).read_bytes()).hexdigest()==sha,name
    assert {k:v for k,v in source.items() if k.endswith('.py')}==plan['sources']
    if not (target/'registration.json').exists():return dict(complete=False,registered=False)
    reg=json.loads((target/'registration.json').read_text())
    assert reg['jobs']==[jobs(spec,plan)[arm]] and reg['stage']==plan['stage']
    assert reg.get('configuration',{})==proof['configurations'][arm]
    if not (target/'audit.json').exists():return dict(complete=False,checkpoint=(target/'checkpoint.json').exists())
    audit=json.loads((target/'audit.json').read_text());trajectory=target/'trajectories.jsonl'
    assert audit['complete'] and audit['source_freeze_verified'] and audit['completed_runs']==1
    assert audit['trajectory_sha256']==hashlib.sha256(trajectory.read_bytes()).hexdigest()
    row=json.loads(trajectory.read_text())
    for key in ('cumulative_achievement','tail_goal_refinement','sampling_check','tail_replanning'):
        assert row.get(key,False)==proof['configurations'][arm].get(key,False)
    assert row.get('tail_replanning_horizon',2)==proof['configurations'][arm].get('tail_replanning_horizon',2)
    from tools.audit_recorded_forecasts import verify_receipt
    verify_receipt(target,audit,row)
    assert 0<len(row['trace'])==row['executed']<=row['turns']==spec['job'][4]
    for key in ('world_rng_replay','restored_action_feedback','physical_replay','owner_parity','fixed_axes'):
        assert audit['checks'][key]==row['executed'],key
    if proof['complete'] and 'host_segments' in proof:verify_segmented_result(target,row,proof,int(arm=='candidate'))
    return dict(complete=True,pair_complete=proof['complete'],score=row['score'],executed=row['executed'])

while True:
    state=api('actions/runs/'+a.run);assert state['head_sha']==a.sha
    artifacts=[];page=1
    while True:
        batch=api('actions/runs/'+a.run+f'/artifacts?per_page=100&page={page}')['artifacts'];artifacts.extend(batch)
        if len(batch)<100:break
        page+=1
    for item in artifacts:
        name=item['name'];assert name in expected,name
        target=root/name;assert target.resolve().parent==root.resolve()
        if target.exists():continue
        staging=root/('.d-'+uuid.uuid4().hex[:10])
        result=subprocess.run(['gh','run','download',a.run,'--name',name,'--dir',str(staging)],cwd=R)
        if result.returncode:continue
        check=verify(staging,*expected[name]);staging.rename(target)
        print(json.dumps(dict(downloaded=name,**check)),flush=True)
    meta={k:state[k] for k in ('id','head_sha','status','conclusion','html_url','created_at','updated_at')}
    meta['collected_case_keys']=[s['key'] for s in selected]
    meta['artifacts']=[{k:x[k] for k in ('id','name','size_in_bytes','expired','created_at')} for x in artifacts]
    (root/'workflow.json').write_text(json.dumps(meta,indent=2)+'\n')
    if state['status']=='completed':
        checks={name:verify(root/name,*spec) if (root/name).exists() else dict(missing=True) for name,spec in expected.items()}
        (root/'collection.json').write_text(json.dumps(checks,indent=2)+'\n')
        if state['conclusion']=='success' and all(c.get('complete') and c.get('pair_complete') for c in checks.values()):
            print('All paired value-contract worlds collected and verified.',flush=True);break
        assert state['conclusion']=='success','incomplete evidence preserved'
        handoffs=[]
        for spec in selected:
            dirs=[root/f'value-contract-{spec["key"]}-{s}' for s in ('baseline','candidate')]
            if json.loads((dirs[0]/'contract_pair.json').read_text())['complete']:continue
            pair_jobs=jobs(spec,plan)
            handoffs.append(verify_handoff(dirs,[pair_jobs[s] for s in ('baseline','candidate')],run=a.run,
                plan_sha=hashlib.sha256(plan_path.read_bytes()).hexdigest(),sources=plan['sources']))
        assert handoffs,'successful incomplete execution lacks a handoff receipt'
        (root/'handoffs.json').write_text(json.dumps(handoffs,indent=2)+'\n')
        print('Verified handoff artifacts collected; experiment remains INCOMPLETE.',flush=True);break
    if a.once:break
    time.sleep(45)
