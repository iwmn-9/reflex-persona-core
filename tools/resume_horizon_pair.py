"""Continue an interrupted pair at a common, resolved decision boundary.

Both policies retain the same sequence of execution hosts. The faster arm's
extra work is preserved in the input artifact, but replayed on the new host.
No policy, world duration, source bytes, experience, or random seed is changed.
"""
import argparse,copy,hashlib,json,os,platform,shutil,subprocess,sys,uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf8')


def sha(data):return hashlib.sha256(data).hexdigest()


def owner_digest(value):
    return sha(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf8'))


def inspect(path,expected_manifest,expected_checkpoint,job):
    path=Path(path).resolve();manifest=(path/'source_snapshot.json').read_bytes()
    assert sha(manifest)==expected_manifest,'wrong frozen source manifest'
    files=json.loads(manifest);source=path/'_source'
    for name,fingerprint in files.items():
        target=(source/name).resolve()
        assert target.is_relative_to(source) and target.is_file(),'invalid frozen source path'
        assert sha(target.read_bytes())==fingerprint,('source changed',name)
    raw=(path/'checkpoint.json').read_bytes()
    assert sha(raw)==expected_checkpoint,'wrong original checkpoint'
    envelope=json.loads(raw);saved=envelope['payload'];trace=saved['trace']
    assert sha(encoded(saved))==envelope['sha256'],'corrupt checkpoint'
    assert saved['contract']['job']==job and saved['contract']['arm']=='receding_recovery'
    assert all(files[k]==v for k,v in saved['contract']['sources'].items()),'execution source mismatch'
    registration=json.loads((path/'registration.json').read_text())
    assert registration['jobs']==[job] and registration['arms']==['receding_recovery']
    assert registration.get('configuration',{})==saved['contract'].get('configuration',{})
    assert trace and len(trace)==saved['world']['tick']==saved['owner']['last_tick']+1
    assert trace[-1]['after']==saved['world'] and trace[-1]['owner']==owner_digest(saved['owner'])
    assert set(saved['checks'])=={'fixed_axes','owner_parity','restore'},'unknown execution checks'
    assert saved['checks']==dict(fixed_axes=len(trace),owner_parity=len(trace),restore=(len(trace)-1)//23+1)
    rng=np.random.default_rng(job[2])
    for index,t in enumerate(trace):
        assert 'forecast_error' not in t and 'realized_at_forecast_end' not in t,'resolved checkpoint required before final summaries'
        assert t['tick']==index and t['before']['tick']==index and t['after']['tick']==index+1
        assert owner_digest(t['checkpoint'])==t['before_owner']
        if index:
            assert trace[index-1]['after']==t['before'] and trace[index-1]['owner']==t['before_owner']
        total=0.;u=rng.random();symbol=list(t['truth'])[-1]
        for key,probability in t['truth'].items():
            total+=probability
            if u<total:symbol=key;break
        assert symbol==t['response'],'original random stream drift'
    assert rng.bit_generator.state==saved['rng'],'original random state drift'
    return dict(path=path,files=files,saved=saved,registration=registration,
                pair=json.loads((path/'pair_execution.json').read_text()),
                manifest_sha256=expected_manifest,checkpoint_sha256=expected_checkpoint)


def prepare(root,inputs,jobs,manifests,checkpoints):
    """Validate both inputs before creating a new, independently resumable pair."""
    root=Path(root).resolve()
    if root.exists():raise FileExistsError('fresh continuation directory required')
    assert all(len(x)==2 for x in (inputs,jobs,manifests,checkpoints))
    assert all(len(j)==8 and j[5]==1 and j[7]==0 and type(j[6]) is int and 1<=j[6]<=127 for j in jobs)
    assert jobs[0][:6]==jobs[1][:6] and jobs[0][7:]==jobs[1][7:] and jobs[0][6]!=jobs[1][6]
    data=[inspect(*args) for args in zip(inputs,manifests,checkpoints,jobs)]
    assert all(not root.is_relative_to(x['path']) for x in data),'continuation output must be outside original inputs'
    left,right=data;prior=left['pair']
    assert prior==right['pair'] and prior['jobs']==jobs,'different original pair'
    assert left['files']==right['files'],'paired source drift'
    assert left['saved']['contract']['runtime']==right['saved']['contract']['runtime'],'paired runtime drift'
    runtime=dict(python=sys.version,numpy=np.__version__,system=platform.system(),machine=platform.machine())
    assert runtime==left['saved']['contract']['runtime'],'continuation runtime changed'
    assert left['registration']['stage']==right['registration']['stage']==prior['stage']
    assert left['registration'].get('configuration',{})==right['registration'].get('configuration',{})
    boundary=min(len(x['saved']['trace']) for x in data)
    assert 0<boundary<jobs[0][4],'pair already complete'
    segments=copy.deepcopy(prior.get('host_segments',[dict(execution_id=prior['pair_id'],start=0,end=None,
        runtime=left['saved']['contract']['runtime'],cpu_count=prior.get('cpu_count'))]))
    assert segments[0]['start']==0 and all(a['end']==b['start'] for a,b in zip(segments,segments[1:]))
    assert segments[-1]['start']<=boundary and segments[-1]['end'] is None,'original segment already finalized'
    segments[-1]['end']=boundary
    execution_id=uuid.uuid4().hex;segments.append(dict(execution_id=execution_id,start=boundary,end=None,
        runtime=runtime,cpu_count=os.cpu_count()))
    header=copy.deepcopy(prior)
    header.update(version='segmented-same-host-horizon-pair-v1',complete=False,same_source=True,
        execution_id=execution_id,host_segments=segments,return_codes=None,
        continuation=dict(boundary=boundary,inputs=[dict(manifest_sha256=x['manifest_sha256'],
            checkpoint_sha256=x['checkpoint_sha256'],original_turns=len(x['saved']['trace'])) for x in data]),
        semantics='Both arms share a host in each recorded interval; the faster arm is rewound to their common resolved boundary.')
    root.mkdir(parents=True)
    for item,job in zip(data,jobs):
        saved=copy.deepcopy(item['saved']);trace=saved['trace'];original_turns=len(trace)
        if boundary<original_turns:
            saved['world']=trace[boundary]['before'];saved['owner']=trace[boundary]['checkpoint']
            saved['trace']=trace[:boundary]
            rng=np.random.default_rng(job[2])
            for _ in range(boundary):rng.random()
            saved['rng']=rng.bit_generator.state
            saved['checks']=dict(fixed_axes=boundary,owner_parity=boundary,restore=(boundary-1)//23+1)
        assert saved['owner']['last_tick']==boundary-1
        assert owner_digest(saved['owner'])==saved['trace'][-1]['owner']
        child=root/f'h{job[6]+1}';child.mkdir()
        shutil.copytree(item['path']/'_source',child/'_source')
        for name in ('source_snapshot.json','registration.json'):
            shutil.copyfile(item['path']/name,child/name)
        if (item['path']/'environment.json').exists():
            shutil.copyfile(item['path']/'environment.json',child/'environment.json')
        (child/'checkpoint.json').write_bytes(encoded(dict(payload=saved,sha256=sha(encoded(saved)))))
        (child/'boundary_provenance.json').write_text(json.dumps(dict(
            source_manifest_sha256=item['manifest_sha256'],original_checkpoint_sha256=item['checkpoint_sha256'],
            original_turns=original_turns,resume_at=boundary,retained_prefix_sha256=sha(encoded(saved['trace'])),
            rewritten_checkpoint_sha256=sha((child/'checkpoint.json').read_bytes()),
            discarded_work_retained_in_input=True),indent=2)+'\n')
        (child/'pair_execution.json').write_text(json.dumps(header,indent=2)+'\n')
    (root/'pair_execution.json').write_text(json.dumps(header,indent=2)+'\n')
    return header


def run_pair(root,inputs,jobs,manifests,checkpoints):
    header=prepare(root,inputs,jobs,manifests,checkpoints);root=Path(root).resolve()
    script=Path(__file__).with_name('resume_horizon_study.py')
    def execute(item):
        job,manifest=item;child=root/f'h{job[6]+1}'
        return subprocess.run([sys.executable,str(script),'--root',str(child),
            '--job',json.dumps(job),'--manifest',manifest],check=False).returncode
    with ThreadPoolExecutor(max_workers=2) as pool:
        codes=list(pool.map(execute,zip(jobs,manifests)))
    header.update(return_codes=codes,complete=codes==[0,0])
    if header['complete']:
        header['host_segments'][-1]['end']=jobs[0][4]
        for job in jobs:
            child=root/f'h{job[6]+1}'
            assert json.loads((child/'resume_provenance.json').read_text())['original_prefix_unchanged']
            assert json.loads((child/'audit.json').read_text())['complete']
    (root/'pair_execution.json').write_text(json.dumps(header,indent=2)+'\n')
    for job in jobs:(root/f'h{job[6]+1}'/'pair_execution.json').write_text(json.dumps(header,indent=2)+'\n')
    if not header['complete']:raise RuntimeError('incomplete pair continuation retained')
    return header


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    for name in ('inputs','jobs','manifests','checkpoints'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(run_pair(a.root,*[json.loads(getattr(a,k)) for k in ('inputs','jobs','manifests','checkpoints')])))
