"""Human-authorized retirement of two identified, already rejected studies.

Preserves every file and stops only the two registered finite orchestration
chains. A subsequent frozen-source factual replay audits each saved prefix.
"""
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from snapshot_registered_queues import argv, canonical, processes

BASE=Path(__file__).resolve().parent
WORKSPACE=BASE.parents[2]
SHELL=r'C:\Program Files\WindowsApps\OpenAI.CodexPrimaryRuntime.v26-1020-978-0_26.1020.978.0_x64__3k8sg7r9htsxt\dependencies\native\powershell\pwsh.exe'
SPECS=[dict(label='purpose',manifest=WORKSPACE/'reflex-persona-purpose-tail/results/purpose_development_20261031/manifest.json',
    sha256='4211226e040e2142a3315f4eac7d08dd9b0d92dc50b96eb11d46b5ea66810380',
    finisher=BASE/'purpose_finish_env',reason='Complete auction family fails both unchanged non-regression gates.'),
    dict(label='origin_gain',manifest=WORKSPACE/'reflex-persona-preview-runtime/results/gain_development_20261031/manifest.json',
    sha256='49c252d35358f55de32c0bb269323e757a1a2133e43002051a717c3a1334a496',
    finisher=BASE/'gain_world_finish_eight_workers',reason='Known background-income termination defect; cannot adopt unchanged.')]
N=WORKSPACE/'reflex-persona-relative-worlds/results/relative_development_20261101/manifest.json'

def parameter(args,name):
    return args[args.index(name)+1] if name in args and args.index(name)+1<len(args) else None

def classified(rows):
    selected=[];protected=[]
    for row in rows:
        args=argv(row.get('CommandLine') or '')
        manifest=parameter(args,'--manifest');root=parameter(args,'--root')
        if ((manifest and canonical(manifest)==canonical(N)) or
            (root and canonical(root)==canonical(BASE/'relative_world_finish')) or
            any(Path(a).name=='finish_relative_post.py' for a in args)):
            protected.append(row);continue
        for spec in SPECS:
            kind=None
            if manifest and canonical(manifest)==canonical(spec['manifest']):
                if '--job' in args:kind='worker'
                elif '--workers' in args:kind='coordinator'
                else:raise ValueError('Unknown registered process role')
            elif root and canonical(root)==canonical(spec['finisher']):
                if not any(Path(a).name=='finish_purpose_study_v2.py' for a in args):
                    raise ValueError('Unexpected finisher implementation')
                kind='finisher'
            if kind:selected.append(dict(**row,study=spec['label'],role=kind))
    return selected,protected

class FT(ctypes.Structure):
    _fields_=[('low',wintypes.DWORD),('high',wintypes.DWORD)]

kernel=ctypes.WinDLL('kernel32',use_last_error=True)
kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
kernel.OpenProcess.restype=wintypes.HANDLE
kernel.GetProcessTimes.argtypes=[wintypes.HANDLE,*([ctypes.POINTER(FT)]*4)]
kernel.GetProcessTimes.restype=wintypes.BOOL
kernel.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT]
kernel.TerminateProcess.restype=wintypes.BOOL
kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
kernel.WaitForSingleObject.restype=wintypes.DWORD
kernel.CloseHandle.argtypes=[wintypes.HANDLE]

def stop(row):
    handle=kernel.OpenProcess(0x1000|0x100000|0x0001,False,row['ProcessId'])
    if not handle:
        if ctypes.get_last_error()==87:return dict(**row,result='already_exited')
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        creation,exit_time,cpu_kernel,cpu_user=FT(),FT(),FT(),FT()
        if not kernel.GetProcessTimes(handle,ctypes.byref(creation),ctypes.byref(exit_time),ctypes.byref(cpu_kernel),ctypes.byref(cpu_user)):
            raise ctypes.WinError(ctypes.get_last_error())
        ticks=(creation.high<<32)|creation.low
        created=datetime.fromisoformat(row['CreationDate'].replace('Z','+00:00')).astimezone(timezone.utc)
        delta=created-datetime(1601,1,1,tzinfo=timezone.utc)
        expected=((delta.days*86400+delta.seconds)*1000000+delta.microseconds)*10
        if not 0<=ticks-expected<10:raise ValueError('PID creation time changed; refusing stop')
        if kernel.WaitForSingleObject(handle,0)==0:return dict(**row,result='already_exited')
        if not kernel.TerminateProcess(handle,130):raise ctypes.WinError(ctypes.get_last_error())
        if kernel.WaitForSingleObject(handle,10000)!=0:raise RuntimeError('Target did not exit; do not inspect checkpoint')
        return dict(**row,result='intentionally_retired',termination_code=130)
    finally:kernel.CloseHandle(handle)

def main():
    out=BASE/'rejected_candidates_retirement_20261101'
    out.mkdir(exist_ok=False)
    for spec in SPECS:
        assert hashlib.sha256(spec['manifest'].read_bytes()).hexdigest()==spec['sha256']
    assert hashlib.sha256(N.read_bytes()).hexdigest()=='0044afae7b0dfb8c711560ec4908115b27ca95b49d92ef3c0c116191d8fd3e81'
    initial,protected=classified(processes(SHELL))
    if not protected:raise ValueError('Expected relative study chain unavailable; inspect first')
    plan=dict(at=datetime.now(timezone.utc).isoformat(),authorization='古い2候補を打ち切り、修正版へ集中',
        authorization_scope='Explicit human reply in this chat on2026-11-01; preserve failure and partial histories; candidate confirmation remains required.',
        studies=[{k:str(v) if isinstance(v,Path) else v for k,v in s.items()} for s in SPECS],
        initial_targets=initial,protected_relative_processes=protected,files_deleted=False,
        sources_seeds_persona_criteria_changed=False,scientific_completion=False)
    (out/'operation_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    stopped=[]
    for role in ('finisher','coordinator','worker'):
        current,now_protected=classified(processes(SHELL))
        for row in current:
            if row['role']!=role:continue
            result=stop(row);stopped.append(result)
            with (out/'operations.jsonl').open('a',encoding='utf-8') as log:
                log.write(json.dumps(result,ensure_ascii=False)+'\n');log.flush()
            print(json.dumps(dict(pid=row['ProcessId'],study=row['study'],role=role,result=result['result'])),flush=True)
    first,protected_after=classified(processes(SHELL))
    second,_=classified(processes(SHELL))
    if first or second:raise ValueError('Retired chain still active; do not open checkpoints')
    proof=dict(operation_complete=True,at=datetime.now(timezone.utc).isoformat(),
        plan_sha256=hashlib.sha256((out/'operation_plan.json').read_bytes()).hexdigest(),
        stopped=stopped,remaining_target_processes=0,protected_relative_processes_after=protected_after,
        sources_seeds_persona_criteria_changed=False,files_deleted=False,
        scientific_completion=False,adoption=False,prefix_audit='required_next')
    (out/'operation_complete.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(operation_complete=True,stopped=len(stopped),relative_processes=len(protected_after))),flush=True)

if __name__=='__main__':main()
