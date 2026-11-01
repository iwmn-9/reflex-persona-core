"""Recover the original arithmetic path using exact historical replay only.

No source patch, tolerance, seed replacement or score-based mode selection.
Run outside the frozen worker checkout. A mode must reproduce EVERY original
owner hash, world event and RNG before that exact worker can resume.
"""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def write(path,value):
    Path(path).write_text(json.dumps(value,sort_keys=True,indent=2)+'\n',encoding='utf8')


def child(args):
    sys.path.insert(0,str(Path(args.source).resolve()))
    import numpy as np
    from tools.adverse_segment import audit
    from tools.run_adverse_recovery import verify_registration
    from numpy._core._multiarray_umath import __cpu_dispatch__,__cpu_features__
    info=io.StringIO()
    with redirect_stdout(info):np.show_runtime()
    out=dict(dispatch=list(__cpu_dispatch__),features=__cpu_features__,runtime=info.getvalue(),
        numpy_disabled=os.environ.get('NPY_DISABLE_CPU_FEATURES',''),
        blas_core=os.environ.get('OPENBLAS_CORETYPE',''))
    cpu=Path('/proc/cpuinfo')
    if cpu.exists():out['cpu_model']=next((s for s in cpu.read_text().splitlines() if s.startswith('model name')),None)
    plan_path=Path(args.source)/'evidence/long_horizon/adverse_recovery/plan.json'
    plan=json.loads(plan_path.read_bytes());spec=next(s for s in plan['specs'] if s['id']==args.case)
    registration=verify_registration(plan_path)
    try:
        out['audit']=audit(args.directory,spec,args.arm,registration)
        out['exact']=True
    except AssertionError:
        out.update(exact=False,error=traceback.format_exc())
    write(args.output,out)
    return 0 if out['exact'] else 2


def recover(args):
    source=Path(args.source).resolve();directory=Path(args.directory).resolve()
    checkpoint=directory/'checkpoint.json'
    initial=checkpoint.read_bytes()
    if hashlib.sha256(initial).hexdigest()!=args.expected_sha:raise ValueError('original checkpoint identity changed')
    trace=json.loads(initial)['payload']['trace']
    saved=directory/'runtime_recovery';saved.mkdir(exist_ok=True)
    base=os.environ.copy();base.pop('NPY_DISABLE_CPU_FEATURES',None);base.pop('OPENBLAS_CORETYPE',None)
    command=[sys.executable,str(Path(__file__).resolve()),'--child','--source',str(source),
        '--directory',str(directory),'--case',args.case,'--arm',args.arm]
    attempts=[]
    def attempt(disabled,blas):
        env=dict(base)
        if disabled:env['NPY_DISABLE_CPU_FEATURES']=','.join(disabled)
        if blas:env['OPENBLAS_CORETYPE']=blas
        output=saved/f'probe-{len(attempts)}.json'
        p=subprocess.run(command+['--output',str(output)],env=env,capture_output=True,text=True)
        row=json.loads(output.read_bytes()) if output.exists() else dict(exact=False,error=p.stderr[-4000:])
        row['returncode']=p.returncode;attempts.append(row)
        write(saved/'attempts.json',attempts)
        if p.returncode not in (0,2):raise RuntimeError('arithmetic probe failed before exact audit: '+str(row))
        assert (p.returncode==0)==row['exact']
        return env if row['exact'] else None
    selected=attempt([],None)
    metadata=attempts[0]
    available=[f for f in metadata['dispatch'] if metadata['features'].get(f,False)]
    profiles=[[],[f for f in available if f.startswith('AVX512')],
        [f for f in available if f.startswith('AVX512') or f in ('AVX2','FMA3')],available]
    tried={((),None)}
    for disabled in profiles:
        for blas in (None,'Haswell'):
            if selected is not None:break
            key=(tuple(disabled),blas)
            if key in tried:continue
            tried.add(key);selected=attempt(disabled,blas)
        if selected is not None:break
    manifest=dict(original_checkpoint_sha256=args.expected_sha,original_steps=len(trace),
        selected_mode=None if selected is None else dict(NPY_DISABLE_CPU_FEATURES=selected.get('NPY_DISABLE_CPU_FEATURES',''),
            OPENBLAS_CORETYPE=selected.get('OPENBLAS_CORETYPE','')),
        selection='First mode that reproduces every original owner hash and physical/random event. No outcome score used.',
        state='exact_replay_recovered' if selected is not None else 'no_exact_runtime_found')
    write(saved/'recovery.json',manifest)
    if selected is None:raise RuntimeError('No arithmetic mode reproduces the original history exactly; do not resume')
    assert checkpoint.read_bytes()==initial,'probing must not change the original checkpoint'
    if args.resume:
        run=[sys.executable,'-u',str(source/'tools/adverse_segment.py'),'--plan',
            str(source/'evidence/long_horizon/adverse_recovery/plan.json'),
            '--case',args.case,'--arm',args.arm,'--output',str(directory),'--minutes','80']
        p=subprocess.run(run,cwd=source,env=selected)
        manifest['worker_returncode']=p.returncode
        manifest['state']='worker_finished' if p.returncode==0 else 'worker_failed'
        write(saved/'recovery.json',manifest)
        if p.returncode:raise RuntimeError('Original worker failed after exact runtime recovery')
        after=json.loads(checkpoint.read_bytes())['payload']
        assert after['trace'][:len(trace)]==trace,'original experience was replaced'
        manifest.update(retained_prefix=True,final_steps=len(after['trace']),
            original_worker_status=json.loads((directory/'status.json').read_bytes()))
        write(saved/'recovery.json',manifest)
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--directory',required=True)
    p.add_argument('--case',required=True);p.add_argument('--arm',required=True)
    p.add_argument('--expected-sha');p.add_argument('--output');p.add_argument('--child',action='store_true')
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.child:sys.exit(child(a))
    recover(a)
