"""Continue an interrupted pair at a common, resolved decision boundary.

Both policies retain the same sequence of execution hosts. The faster arm's
extra work is preserved in the input artifact, but replayed on the new host.
No policy, world duration, source bytes, experience, or random seed is changed.
"""
import argparse,copy,hashlib,json,os,platform,shutil,subprocess,sys,uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf8')


def sha(data):return hashlib.sha256(data).hexdigest()


def owner_digest(value):
    return sha(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf8'))


def inspect(path,expected_manifest,expected_checkpoint,job,*,pair_file='pair_execution.json'):
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
                pair=json.loads((path/pair_file).read_text()),
                manifest_sha256=expected_manifest,checkpoint_sha256=expected_checkpoint)


def prepare(root,inputs,jobs,manifests,checkpoints,*,value_contract=False):
    """Validate both inputs before creating a new, independently resumable pair."""
    root=Path(root).resolve()
    if root.exists():raise FileExistsError('fresh continuation directory required')
    assert all(len(x)==2 for x in (inputs,jobs,manifests,checkpoints))
    assert all(len(j)==8 and j[5]==1 and j[7]==0 and type(j[6]) is int and 1<=j[6]<=127 for j in jobs)
    assert type(value_contract) is bool
    pair_file='contract_pair.json' if value_contract else 'pair_execution.json'
    names=('baseline','candidate') if value_contract else tuple(f'h{j[6]+1}' for j in jobs)
    if value_contract:assert jobs[0][:6]==jobs[1][:6] and jobs[0][7:]==jobs[1][7:]
    else:assert jobs[0][:6]==jobs[1][:6] and jobs[0][7:]==jobs[1][7:] and jobs[0][6]!=jobs[1][6]
    data=[inspect(*args,pair_file=pair_file) for args in zip(inputs,manifests,checkpoints,jobs)]
    assert all(not root.is_relative_to(x['path']) for x in data),'continuation output must be outside original inputs'
    left,right=data;prior=left['pair']
    assert prior==right['pair'],'different original pair'
    assert not prior.get('complete',False),'pair already complete'
    versions=('same-host-value-contract-v1','segmented-same-host-value-contract-v1') if value_contract else ('same-host-horizon-pair-v1','segmented-same-host-horizon-pair-v1')
    assert prior['version'] in versions,'unknown pair execution contract'
    if value_contract:
        assert prior['job']==jobs[1]
        assert prior.get('jobs',dict(baseline=prior['job'],candidate=prior['job']))==dict(zip(names,jobs)),'different original pair'
    else:assert prior['jobs']==jobs,'different original pair'
    assert left['files']==right['files'],'paired source drift'
    assert left['saved']['contract']['runtime']==right['saved']['contract']['runtime'],'paired runtime drift'
    runtime=dict(python=sys.version,numpy=np.__version__,system=platform.system(),machine=platform.machine())
    assert runtime==left['saved']['contract']['runtime'],'continuation runtime changed'
    assert left['registration']['stage']==right['registration']['stage']
    if value_contract:
        from tools.value_contract_config import configurations
        assert prior['configurations']==configurations(prior)
        assert prior['sources']=={k:v for k,v in left['files'].items() if k.endswith('.py')}
        assert len(prior['plan_sha256'])==64
        for name,item in zip(names,data):assert item['registration'].get('configuration',{})==prior['configurations'][name]
    else:
        assert left['registration']['stage']==prior['stage']
        assert left['registration'].get('configuration',{})==right['registration'].get('configuration',{})
    boundary=min(len(x['saved']['trace']) for x in data)
    assert 0<boundary<=jobs[0][4],'invalid common execution boundary'
    segments=copy.deepcopy(prior.get('host_segments',[dict(execution_id=prior['pair_id'],start=0,end=None,
        runtime=left['saved']['contract']['runtime'],cpu_count=prior.get('cpu_count'))]))
    assert segments[0]['start']==0 and all(a['end']==b['start'] for a,b in zip(segments,segments[1:]))
    assert segments[-1]['start']<=boundary and segments[-1]['end'] is None,'original segment already finalized'
    segments[-1]['end']=boundary
    execution_id=uuid.uuid4().hex;segments.append(dict(execution_id=execution_id,start=boundary,end=None,
        runtime=runtime,cpu_count=os.cpu_count()))
    header=copy.deepcopy(prior)
    header.update(version='segmented-same-host-value-contract-v1' if value_contract else 'segmented-same-host-horizon-pair-v1',complete=False,same_source=True,
        execution_id=execution_id,host_segments=segments,return_codes=None,
        continuation=dict(boundary=boundary,at_scheduled_end=boundary==jobs[0][4],inputs=[dict(manifest_sha256=x['manifest_sha256'],
            checkpoint_sha256=x['checkpoint_sha256'],original_turns=len(x['saved']['trace'])) for x in data]),
        semantics='Both arms share a host in each recorded interval; the faster arm is rewound to their common resolved boundary.')
    root.mkdir(parents=True)
    for name,item,job in zip(names,data,jobs):
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
        child=root/name;child.mkdir()
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
        (child/pair_file).write_text(json.dumps(header,indent=2)+'\n')
    (root/pair_file).write_text(json.dumps(header,indent=2)+'\n')
    return header


def run_pair(root,inputs,jobs,manifests,checkpoints,*,value_contract=False,recorded_predictions=False):
    assert type(recorded_predictions) is bool
    header=prepare(root,inputs,jobs,manifests,checkpoints,value_contract=value_contract);root=Path(root).resolve()
    names=('baseline','candidate') if value_contract else tuple(f'h{j[6]+1}' for j in jobs)
    pair_file='contract_pair.json' if value_contract else 'pair_execution.json'
    script=Path(__file__).with_name('resume_horizon_study.py')
    def execute(item):
        name,job,manifest=item;child=root/name
        args=[sys.executable,str(script),'--root',str(child),
            '--job',json.dumps(job),'--manifest',manifest]
        if recorded_predictions:args+=['--recorded-predictions']
        return subprocess.run(args,check=False).returncode
    with ThreadPoolExecutor(max_workers=2) as pool:
        codes=list(pool.map(execute,zip(names,jobs,manifests)))
    header.update(return_codes=codes,complete=codes==[0,0])
    if recorded_predictions:header['recorded_prediction_replay_permitted']=True
    if header['complete']:
        header['host_segments'][-1]['end']=jobs[0][4]
        added=[]
        for name in names:
            child=root/name
            assert json.loads((child/'resume_provenance.json').read_text())['original_prefix_unchanged']
            assert json.loads((child/'audit.json').read_text())['complete']
            added.append(json.loads((child/'trajectories.jsonl').read_text())['executed']-header['continuation']['boundary'])
        assert all(n>=0 for n in added)
        header['continuation'].update(added_decisions=added,no_new_decisions=not any(added))
    (root/pair_file).write_text(json.dumps(header,indent=2)+'\n')
    for name in names:(root/name/pair_file).write_text(json.dumps(header,indent=2)+'\n')
    if not header['complete']:raise RuntimeError('incomplete pair continuation retained')
    return header


def verify_segmented_result(path,row,header,side):
    """Read-only proof shared by collectors and comparison summaries."""
    path=Path(path);segments=header['host_segments']
    assert header['complete'] and side in (0,1)
    assert segments[0]['start']==0 and segments[-1]['end']==row['turns']
    assert all(a['end']==b['start'] for a,b in zip(segments,segments[1:]))
    assert all(0<=s['start']<=s['end']<=row['turns'] and s['execution_id'] for s in segments)
    boundary=json.loads((path/'boundary_provenance.json').read_text())
    resumed=json.loads((path/'resume_provenance.json').read_text())
    assert resumed['complete'] and resumed['original_prefix_unchanged']
    assert resumed['resume_at']==boundary['resume_at']==header['continuation']['boundary']
    assert resumed['source_manifest_sha256']==sha((path/'source_snapshot.json').read_bytes())
    assert resumed['original_checkpoint_sha256']==boundary['rewritten_checkpoint_sha256']
    assert boundary['original_checkpoint_sha256']==header['continuation']['inputs'][side]['checkpoint_sha256']
    prefix=[{k:v for k,v in t.items() if k not in ('forecast_error','realized_at_forecast_end')}
            for t in row['trace'][:boundary['resume_at']]]
    assert len(prefix)==boundary['resume_at'] and sha(encoded(prefix))==boundary['retained_prefix_sha256']
    if 'added_decisions' in header['continuation']:
        assert header['continuation']['added_decisions'][side]==row['executed']-boundary['resume_at']
    return segments


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    for name in ('inputs','jobs','manifests','checkpoints'):p.add_argument('--'+name,required=True)
    p.add_argument('--value-contract',action='store_true');p.add_argument('--recorded-predictions',action='store_true')
    a=p.parse_args();print(json.dumps(run_pair(a.root,*[json.loads(getattr(a,k)) for k in ('inputs','jobs','manifests','checkpoints')],value_contract=a.value_contract,recorded_predictions=a.recorded_predictions)))
