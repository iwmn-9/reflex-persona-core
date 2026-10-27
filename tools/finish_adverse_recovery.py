"""Collect one registered study and finitely resume only its original worlds.

No scheduler, fresh seeds, new experiment, automatic adoption, or failed-worker
retry. Completed artifacts are collected while other matrix jobs still run.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.owned_checkpoint import encoded,sha
from tools.review_adverse_recovery import summarize


def execute(gh, *args):
    command=[gh,*args]
    # An unknown dispatch outcome may already have started a workflow. Retrying
    # a mutation would duplicate twenty experiments; reconcile via run listing.
    attempts = 1 if args[:2] == ('workflow', 'run') else 3
    for attempt in range(attempts):
        p=subprocess.run(command,capture_output=True,text=True,encoding='utf8',errors='replace')
        if p.returncode==0:return p.stdout
        if attempt+1<attempts:time.sleep(10)
    raise RuntimeError('GitHub operation failed: '+p.stderr[-1800:])


def finish(root,run,ref,gh,max_segments):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    plan_path=ROOT/'evidence/long_horizon/adverse_recovery/plan.json'
    plan=json.loads(plan_path.read_bytes());registration=json.loads((plan_path.parent/'registration.json').read_bytes())
    assert sha(plan_path.read_bytes())==registration['plan_sha256']
    state_path=root/'state.json'
    state=json.loads(state_path.read_bytes()) if state_path.exists() else dict(
        initial_run=run,current_run=run,source_ref=ref,segments=[run],collected={},complete=False,
        study_plan_sha256=registration['plan_sha256'],phase='collecting',pid=__import__('os').getpid())
    assert state['initial_run']==run and state['source_ref']==ref and state['study_plan_sha256']==registration['plan_sha256']
    state['pid']=__import__('os').getpid()
    state['collector_sha256']=sha(Path(__file__).read_bytes())
    def save(**values):
        state.update(values,updated=datetime.now(timezone.utc).isoformat())
        temp=state_path.with_suffix('.new');temp.write_bytes(encoded(state));temp.replace(state_path)
    save()
    repo='iwmn-9/reflex-persona-core';prefix=f'repos/{repo}'
    wanted={f"adverse-{s['id']}-{arm}" for s in plan['specs'] for arm in plan['arms']}
    deadline=time.monotonic()+max_segments*7*3600
    try:
        while time.monotonic()<deadline:
            current=str(state['current_run'])
            info=json.loads(execute(gh,'api',f'{prefix}/actions/runs/{current}'))
            assert info['head_sha']==ref,'do not continue another source revision'
            listing=json.loads(execute(gh,'api',f'{prefix}/actions/runs/{current}/artifacts?per_page=100'))
            for artifact in listing['artifacts']:
                if artifact['name'] not in wanted or artifact['expired']:continue
                name=artifact['name'];identity=str(artifact['id'])
                if state['collected'].get(name)==identity:continue
                target=root/'downloads'/identity
                if not target.exists():
                    staging=root/'downloads'/f'{identity}.partial'
                    # A failed download remains distinct from a completed copy.
                    attempt=0
                    while staging.exists():
                        attempt+=1;staging=root/'downloads'/f'{identity}.partial-{attempt}'
                    execute(gh,'run','download',current,'--repo',repo,'--name',name,'--dir',str(staging))
                    staging.rename(target)
                shutil.copytree(target,root/'latest'/name,dirs_exist_ok=True)
                state['collected'][name]=identity;save()
            report=summarize(root/'latest',plan)
            (root/'summary.json').write_bytes(encoded(report))
            progress={p['id']+'/'+p['arm']:p['actual_steps'] if p['state']!='missing' else None for p in report['progress']}
            jobs=json.loads(execute(gh,'api',f'{prefix}/actions/runs/{current}/jobs?per_page=100'))['jobs']
            save(pairs_complete=report['pairs_complete'],progress=progress,
                 progress_semantics='Last retrieved resolved steps; null means not yet retrieved, not zero execution.',
                 workers=[dict(name=j['name'],status=j['status'],conclusion=j['conclusion']) for j in jobs],
                 workflow_status=info['status'],workflow_conclusion=info['conclusion'])
            if info['status']!='completed':
                time.sleep(60);continue
            if info['conclusion']!='success':
                raise RuntimeError('Research workflow failed; original evidence retained for diagnosis: '+info['html_url'])
            expected=list(wanted)
            assert all(name in state['collected'] for name in expected),'missing worker artifacts'
            if report['complete']:
                save(complete=True,phase='complete_needs_review',development_gate_passed=report['development_gate_passed'])
                print(json.dumps(dict(complete=True,summary=str(root/'summary.json'),development_gate_passed=report['development_gate_passed'])),flush=True)
                return
            assert all(p['state'] in ('complete','handoff_pending') for p in report['progress']),'failed worker is not a handoff'
            if len(state['segments'])>=max_segments:
                raise RuntimeError('Finite study budget reached with incomplete original worlds; inspect progress before continuing')
            # One dispatch per completed handoff, never a fresh run of the same conditions.
            prior=state.get('dispatch_pending')
            if not prior:
                save(dispatch_pending=current)
                try:
                    execute(gh,'workflow','run','adverse-recovery.yml','--repo',repo,'--ref',ref,'-f',f'previous_run={current}')
                except RuntimeError as error:
                    save(dispatch_error=repr(error))
            candidates=[]
            for _ in range(12):
                runs=json.loads(execute(gh,'api',f'{prefix}/actions/workflows/adverse-recovery.yml/runs?per_page=40'))['workflow_runs']
                candidates=[r for r in runs if r['head_sha']==ref and r['display_title']==f'Adverse recovery {current}' and r['id'] not in state['segments']]
                if candidates:break
                time.sleep(10)
            if len(candidates)!=1:raise RuntimeError('Ambiguous/missing dispatch; inspect before issuing another')
            next_run=candidates[0]['id'];state['segments'].append(next_run)
            save(current_run=next_run,dispatch_pending=None)
        raise RuntimeError('Finite collector deadline reached; no new work automatically scheduled')
    except BaseException as error:
        save(phase='needs_attention',error=repr(error))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--run',type=int,required=True);p.add_argument('--ref',required=True)
    p.add_argument('--gh',default='gh');p.add_argument('--max-segments',type=int,default=4)
    a=p.parse_args()
    if not 1<=a.max_segments<=4:raise ValueError('one to four finite execution segments required')
    finish(a.root,a.run,a.ref,a.gh,a.max_segments)
