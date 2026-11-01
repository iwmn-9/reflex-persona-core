"""Finite post-audit continuation with a hash-bound completed regression receipt."""
import argparse, ctypes, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from finish_relative_post import wait_existing
from prepare_reserved_purpose_confirmation import read, encoded, sha

def validate_regression(value):
    if (value.get('complete') is not True or value.get('passed') is not True
            or value.get('source_commit')!='493a685ab2837c999d13d832dc25a51068845fb5'
            or value.get('failures') or value.get('errors') or value.get('unexpected_successes')):
        raise ValueError('fixed whole-source regression must be complete and passed')

def validate_finisher(value, checksum):
    if (value.get('complete') is not True or value.get('state')!='complete_requires_scientific_review'
            or value.get('manifest_sha256')!=checksum):
        raise ValueError('whole world and all original diagnostics must finish')

def main(root):
    root=Path(root).resolve();manifest=read(root/'manifest.json');status=root/'status.json'
    if status.exists():raise FileExistsError('inspect existing post-audit; never duplicate')
    state=dict(pid=os.getpid(),state='waiting',complete=False,adoption=False,
        manifest_sha256=sha((root/'manifest.json').read_bytes()),started=datetime.now(timezone.utc).isoformat())
    def save():
        state['updated']=datetime.now(timezone.utc).isoformat()
        temporary=status.with_suffix('.new');temporary.write_bytes(encoded(state));temporary.replace(status)
    def sources():
        for name,checksum in manifest['files'].items():
            if sha(Path(name).read_bytes())!=checksum:raise ValueError('bound execution file changed: '+name)
        if sha(Path(manifest['regression_status']).read_bytes())!=manifest['completed_regression_sha256']:
            raise ValueError('completed regression receipt changed')
        validate_regression(read(Path(manifest['regression_status'])))
    handle=None
    try:
        sources();kernel,handle=wait_existing(manifest['dependency_pid']);save()
        if kernel.WaitForSingleObject(handle,0xFFFFFFFF)!=0:raise RuntimeError('dependency wait failed')
        sources();validate_finisher(read(Path(manifest['finisher_status'])),manifest['finisher_manifest_sha256'])
        for step in manifest['steps']:
            sources();state['state']=step['kind'];save()
            if Path(step['proof']).exists():raise FileExistsError('inspect previous output rather than rerun')
            with (root/(step['kind']+'.stdout.log')).open('wb') as stdout, (root/(step['kind']+'.stderr.log')).open('wb') as stderr:
                child=subprocess.Popen([sys.executable,*step['args']],cwd=Path(__file__).resolve().parent,
                    env=dict(os.environ,**manifest['arithmetic']),stdout=stdout,stderr=stderr,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                state['child_pid']=child.pid;save();code=child.wait()
            state.pop('child_pid',None)
            if code:raise subprocess.CalledProcessError(code,step['args'])
            proof=read(Path(step['proof']))
            if proof.get('complete') is not True:raise ValueError('complete proof required')
            if step['kind']=='package_history' and (proof['executions'],proof['actual_steps'])!=(20,2560):
                raise ValueError('all original development histories must replay')
        sources();state.update(state='complete_requires_scientific_review',complete=True);save()
    except BaseException as error:
        state.update(state='failed_requires_inspection',error=repr(error));save();raise
    finally:
        if handle:kernel.CloseHandle(handle)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);main(parser.parse_args().root)
