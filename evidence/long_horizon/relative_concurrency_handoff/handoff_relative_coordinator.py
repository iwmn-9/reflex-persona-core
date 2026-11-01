"""Increase only finite queue concurrency; preserve every running scientific worker."""
import json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
from retire_rejected_studies import stop, SHELL, kernel
from snapshot_registered_queues import argv, canonical, match_workers
from finish_relative_post_resumed import validate_regression
from finish_relative_study import read, sha, write

local=Path(__file__).resolve().parent
source=local.parents[2]/'reflex-persona-relative-worlds'
queue=source/'results/relative_development_20261101'
out=local/'relative_concurrency_handoff_20261101'
old_finish=local/'relative_world_finish';new_finish=local/'relative_world_finish_eight'
old_post=local/'relative_post_archive';new_post=local/'relative_post_archive_eight'
manifest=queue/'manifest.json'

def processes():
    command="@(Get-CimInstance Win32_Process | Where-Object { $_.Name -in @('python.exe','pwsh.exe') } | Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine) | ConvertTo-Json -Compress -Depth 3"
    return json.loads(subprocess.check_output([SHELL,'-NoProfile','-Command',command],encoding='utf-8-sig'))

def verify_files(files):
    for name,checksum in files.items():
        if sha(name)!=checksum:raise ValueError('Bound file changed: '+name)

def launch(args,cwd,folder,label,environment):
    with (folder/(label+'.stdout.log')).open('xb') as stdout,(folder/(label+'.stderr.log')).open('xb') as stderr:
        return subprocess.Popen(args,cwd=cwd,env=environment,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)

def main():
    for folder in (out,new_finish,new_post):
        if folder.exists():raise FileExistsError('Inspect existing handoff; do not repeat: '+str(folder))
    assert sha(manifest)=='0044afae7b0dfb8c711560ec4908115b27ca95b49d92ef3c0c116191d8fd3e81'
    fm=read(old_finish/'manifest.json');pm=read(old_post/'manifest.json')
    verify_files(fm['files']);verify_files(pm['files'])
    registration=read(source/'evidence/relative_gain/worlds/registration.json')
    verify_files({str(source/name):h for name,h in registration['sources'].items()})
    validate_regression(read(pm['regression_status']))
    for stage in fm['stages']+pm['steps']:
        if Path(stage['proof']).exists():raise ValueError('A stage has advanced; do not hand off blindly')
    initial=processes();byid={row['ProcessId']:row for row in initial}
    expected={25740:'finish_relative_post.py',27664:'finish_relative_study.py',27860:str(manifest),
        1836:'prioritize_relative_worlds_resumed.ps1',26196:'focus_relative_worlds.ps1'}
    for pid,part in expected.items():
        row=byid.get(pid)
        if not row or part.replace('/','\\').lower() not in row['CommandLine'].replace('/','\\').lower():raise ValueError('Control identity changed')
    if any(r['ParentProcessId'] in (25740,27664) for r in initial):raise ValueError('Postprocessing already has a child')
    workers=match_workers(initial,manifest)
    if set(workers)!=set(range(4)) or any(len(x)!=1 for x in workers.values()):raise ValueError('Review changed initial workers')
    retained={r[0]['pid']:r[0]['created'] for r in workers.values()}
    for folder in (out,new_finish,new_post):folder.mkdir()
    receipt=dict(state='prepared',at=datetime.now(timezone.utc).isoformat(),old_coordinator=27860,
        old_finisher=27664,old_post=25740,retained_workers=workers,scientific_manifest_sha256=sha(manifest),
        previous_workers=4,requested_workers=8,scientific_workers_terminated=False,
        source_seed_persona_criteria_changed=False,scientific_completion=False,adoption=False)
    write(out/'receipt.json',receipt)
    for pid in (1836,26196,25740,27664,27860):
        result=stop(byid[pid]);result['result']='orchestration_replaced_worker_history_retained'
        with (out/'operations.jsonl').open('a',encoding='utf8') as log:log.write(json.dumps(result)+'\n')
    fs=read(old_finish/'status.json');ps=read(old_post/'status.json');qs=read(queue/'state.json')
    for name,value in [('finisher_before.json',fs),('post_before.json',ps),('queue_before.json',qs)]:write(out/name,value)
    if fs['state']!='waiting_for_original_coordinator' or fs['steps'] or fs.get('child_pid'):raise ValueError('Unexpected prior finisher state')
    if ps['state']!='waiting' or ps.get('child_pid'):raise ValueError('Unexpected prior post state')
    if qs['pid']!=27860 or qs['complete'] or qs['failures']:raise ValueError('Unexpected prior coordinator state')
    if {int(v) for v in qs['active'].values()}!=set(retained):raise ValueError('Prior worker accounting differs')
    environment=dict(os.environ,**fm['arithmetic'])
    coordinator=launch([sys.executable,'-u',str(queue/'driver.py'),'--manifest',str(manifest),'--workers','8'],source,out,'coordinator',environment)
    receipt.update(state='coordinator_started',coordinator=coordinator.pid);write(out/'receipt.json',receipt)
    deadline=time.monotonic()+45
    while True:
        if coordinator.poll() is not None:raise RuntimeError('Replacement coordinator failed; inspect retained workers')
        rows=processes();new=match_workers(rows,manifest)
        if len(new)==8:break
        if time.monotonic()>deadline:raise TimeoutError('Eight workers not established')
        time.sleep(1)
    flat={r[0]['pid']:r[0]['created'] for r in new.values() if len(r)==1}
    if len(flat)!=8 or any(flat.get(pid)!=created for pid,created in retained.items()):raise ValueError('Original workers not retained exactly once')
    fm['parent_pid']=coordinator.pid
    write(new_finish/'manifest.json',fm)
    finisher=launch([sys.executable,'-u',str(local/'finish_relative_study.py'),'--root',str(new_finish)],source,new_finish,'execution',environment)
    receipt.update(state='finisher_started',finisher=finisher.pid);write(out/'receipt.json',receipt)
    pm.pop('dependency_pids');pm.update(dependency_pid=finisher.pid,finisher_status=str(new_finish/'status.json'),
        finisher_manifest_sha256=sha(new_finish/'manifest.json'),completed_regression_sha256=sha(pm['regression_status']))
    pm['files'][str(local/'finish_relative_post_resumed.py')]=sha(local/'finish_relative_post_resumed.py')
    pm['files'][str(new_finish/'manifest.json')]=sha(new_finish/'manifest.json')
    write(new_post/'manifest.json',pm)
    post=launch([sys.executable,'-u',str(local/'finish_relative_post_resumed.py'),'--root',str(new_post)],source,new_post,'execution',environment)
    parent=next(r for r in rows if r['ProcessId']==coordinator.pid)
    write(out/'priority_plan.json',dict(coordinator=coordinator.pid,coordinator_created=parent['CreationDate'],manifest=str(manifest)))
    helper=launch([SHELL,'-NoProfile','-File',str(local/'prioritize_relative_surviving.ps1'),'-Root',str(out)],source,out,'priority',environment)
    receipt.update(state='post_started',post=post.pid,priority_helper=helper.pid);write(out/'receipt.json',receipt)
    time.sleep(2)
    for process in (coordinator,finisher,post,helper):
        if process.poll() is not None:raise RuntimeError('Replacement control exited; inspect logs')
    verify_files(fm['files']);verify_files(pm['files'])
    verify_files({str(source/name):h for name,h in registration['sources'].items()})
    receipt.update(state='complete',at=datetime.now(timezone.utc).isoformat(),active_workers=new,
        finisher_manifest_sha256=sha(new_finish/'manifest.json'),post_manifest_sha256=sha(new_post/'manifest.json'),
        regression_reused_without_rerun=True,source_files_verified=len(registration['sources']),
        subsequent_stages_unchanged=True)
    write(out/'receipt.json',receipt)
    print(json.dumps({k:receipt[k] for k in ('state','coordinator','finisher','post','priority_helper','scientific_workers_terminated')}),flush=True)

if __name__=='__main__':main()
