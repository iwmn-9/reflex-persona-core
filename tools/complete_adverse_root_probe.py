"""Recover missing registered roots against their archived computation bytes."""
import argparse
from contextlib import redirect_stdout
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


def prepare(directory):
    directory=Path(directory).resolve();reg=json.loads((directory/'registration.json').read_bytes())
    assert hashlib.sha256((directory/'inputs.json.gz').read_bytes()).hexdigest()==reg['inputs_sha256']
    source=directory/'replay_source'
    with zipfile.ZipFile(directory/'source.zip') as archive:
        for name,checksum in reg['sources'].items():
            target=(source/name).resolve();assert source in target.parents
            data=archive.read(name);assert hashlib.sha256(data).hexdigest()==checksum
            if target.exists():assert target.read_bytes()==data
            else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    return directory,source,reg,json.loads(gzip.decompress((directory/'inputs.json.gz').read_bytes()))


def child(args):
    directory,source,reg,inputs=prepare(args.directory)
    sys.path.insert(0,str(source))
    from tools.probe_adverse_world_continuation import probe
    from tools.run_adverse_recovery import game_for
    from reflex.core import digest
    from reflex.decision_loop import DecisionLoop
    from reflex.model_observer import resolve_feedback
    import numpy as np
    data=inputs[args.case_index];name,tick=reg['cases'][args.case_index]
    g=game_for(data['spec']);w=g.start();actor=g.actor(g.context(w),learning='segmented')
    for t in data['result']['trace'][:tick]:
        if digest(actor.record())!=t['before_owner'] or w!=t['before']:return 2
        c=g.context(w,actor.state)
        r=DecisionLoop.decide_batch([(actor,g.request(c))],False,_counterfactual_roots=(t['action'],))[0]
        after,flow=g.transition(w,t['action'],t['response'])
        resolve_feedback(actor,r['ticket'],g.feedback(w,after,t['action'],c,flow))
        if digest(actor.record())!=t['owner']:return 2
        w=after
    assert digest(actor.record())==data['result']['trace'][tick]['before_owner']
    if args.check:return 0
    runtime=io.StringIO()
    with redirect_stdout(runtime):np.show_runtime()
    previous=[json.loads(s) for s in (directory/'rows.jsonl').read_bytes().splitlines()]
    seen={(r['id'],r['tick'],r['world_model'],r['sampling']) for r in previous}
    output=dict(exact_prefix=True,case=name,tick=tick,factual_owner=digest(actor.record()),
        runtime=runtime.getvalue(),source_zip_sha256=hashlib.sha256((directory/'source.zip').read_bytes()).hexdigest(),
        arithmetic={k:os.environ.get(k,'') for k in ('NPY_DISABLE_CPU_FEATURES','OPENBLAS_CORETYPE')},rows=[])
    destination=directory/f'recovered-root-{args.case_index}.json'
    if destination.exists():raise FileExistsError('do not rerun recorded roots')
    for sampling in reg['sampling']:
        for model in reg['world_models']:
            if (name,tick,model,sampling) in seen:continue
            row=probe(data,tick,sampling,model);output['rows'].append(row)
            destination.write_text(json.dumps(output,sort_keys=True)+'\n',encoding='utf8')
            print(json.dumps({k:row[k] for k in ('id','tick','sampling','world_model','action','seconds')}),flush=True)
    output['complete']=True;destination.write_text(json.dumps(output,sort_keys=True)+'\n',encoding='utf8')
    return 0


def recover(args):
    command=[sys.executable,str(Path(__file__).resolve()),'--directory',args.directory,'--case-index',str(args.case_index),'--child']
    base=os.environ.copy();base.pop('NPY_DISABLE_CPU_FEATURES',None);base.pop('OPENBLAS_CORETYPE',None)
    avx='AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL'
    attempts=[];selected=None
    for disabled in ('',avx,avx+',AVX2,FMA3'):
        for blas in ('','Haswell'):
            env=dict(base)
            if disabled:env['NPY_DISABLE_CPU_FEATURES']=disabled
            if blas:env['OPENBLAS_CORETYPE']=blas
            p=subprocess.run(command+['--check'],env=env,capture_output=True,text=True)
            attempts.append(dict(disabled=disabled,blas=blas,returncode=p.returncode,error=p.stderr[-3000:]))
            if p.returncode not in (0,2):raise RuntimeError(str(attempts[-1]))
            if p.returncode==0:selected=env;break
        if selected is not None:break
    Path(args.directory,f'root-{args.case_index}-arithmetic-attempts.json').write_text(json.dumps(attempts,sort_keys=True)+'\n',encoding='utf8')
    if selected is None:raise RuntimeError('no exact prefix arithmetic; no forecast is accepted')
    subprocess.run(command,env=selected,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--case-index',type=int,choices=range(5),required=True)
    p.add_argument('--child',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.child:sys.exit(child(a))
    recover(a)
