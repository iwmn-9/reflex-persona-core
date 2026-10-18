"""Place up to four independent frozen studies on a host without changing trials.

Restored studies continue their original source and checkpoint. Fresh studies
use the explicitly checked-out source. This is an execution batch, not a new
comparison policy or permission to discard partially executed worlds.
"""
import argparse,json,os,platform,shutil,subprocess,sys,uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def run_batch(root,studies,fresh_source,stage):
    root=Path(root).resolve();fresh_source=Path(fresh_source).resolve()
    if not isinstance(studies,list) or not 1<=len(studies)<=4:
        raise ValueError('one to four independent registered studies required')
    if len({s['key'] for s in studies})!=len(studies):raise ValueError('distinct studies required')
    if not (fresh_source/'tools/run_horizon_study.py').is_file():raise ValueError('explicit execution source required')
    manifest=root/'batch_execution.json'
    if manifest.exists():raise FileExistsError('fresh batch required; retain prior execution records')
    children=[root/f'study{i}' for i in range(len(studies))]
    for child,spec in zip(children,studies):
        if 'resume_manifest' in spec:
            audit=child/'audit.json'
            if not (child/'checkpoint.json').is_file() or (audit.exists() and json.loads(audit.read_text())['complete']):
                raise ValueError('restored incomplete checkpoint required')
        elif child.exists():raise FileExistsError('fresh study must not overwrite prior work')
    root.mkdir(parents=True,exist_ok=True)
    header=dict(version='independent-study-batch-v1',batch_id=uuid.uuid4().hex,studies=studies,
        stage=stage,python=sys.version,system=platform.system(),machine=platform.machine(),
        cpu_count=os.cpu_count(),complete=False)
    manifest.write_text(json.dumps(header,indent=2)+'\n')
    for child in children:
        child.mkdir(exist_ok=True);shutil.copyfile(manifest,child/manifest.name)
    def execute(item):
        child,spec=item
        if 'resume_manifest' in spec:
            registration=json.loads((child/'registration.json').read_text())
            assert registration['stage']==stage and registration['jobs']==[spec['job']]
            args=[sys.executable,str(Path(__file__).with_name('resume_horizon_study.py')),
                '--root',str(child),'--manifest',spec['resume_manifest'],'--job',json.dumps(spec['job'])]
            if spec.get('recorded_predictions') is True:args+=['--recorded-predictions']
        else:
            args=[sys.executable,str(fresh_source/'tools/run_horizon_study.py'),
                '--root',str(child),'--job',json.dumps(spec['job']),'--stage',stage]
        return subprocess.run(args,check=False).returncode
    with ThreadPoolExecutor(max_workers=len(studies)) as pool:
        codes=list(pool.map(execute,zip(children,studies)))
    header.update(return_codes=codes,complete=all(code==0 for code in codes))
    if header['complete']:
        snapshots=[json.loads((child/'source_snapshot.json').read_text()) for child in children]
        assert all(s==snapshots[0] for s in snapshots),'registered batch source drift'
        header['same_source']=True
        for child,spec in zip(children,studies):
            assert json.loads((child/'registration.json').read_text())['jobs']==[spec['job']]
            assert json.loads((child/'audit.json').read_text())['complete']
    manifest.write_text(json.dumps(header,indent=2)+'\n')
    for child in children:shutil.copyfile(manifest,child/manifest.name)
    if not header['complete']:raise RuntimeError('incomplete batch retained with original checkpoints')
    return header


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--studies',required=True)
    p.add_argument('--fresh-source',required=True);p.add_argument('--stage',required=True);a=p.parse_args()
    print(json.dumps(run_batch(a.root,json.loads(a.studies),a.fresh_source,a.stage)),flush=True)
