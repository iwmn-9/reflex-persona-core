"""Prepare all registered loss probes from a COMPLETE audited study.

No simulations, adoption decision, or resumption occurs here. The finite driver
and collector remain separate; the original frozen worker is never modified.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import uuid
import zipfile


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())


def inside(parent,name):
    path=(parent/name).resolve()
    if not path.is_relative_to(parent.resolve()):raise ValueError('path leaves its declared root')
    return path


def prepare(study,worker,source_plan,resolver,driver,probe,output):
    study,worker,resolver,driver,probe,output=map(lambda p:Path(p).resolve(),
        (study,worker,resolver,driver,probe,output))
    if output.exists():raise FileExistsError('retain the existing prepared execution')
    proof=read(study/'verification.json')
    if proof.get('complete') is not True:raise ValueError('complete audited study required')
    for name,expected in proof['artifacts'].items():
        if sha(inside(study,name).read_bytes())!=expected:raise ValueError('audited artifact changed: '+name)
    registered=inside(worker,source_plan)
    registration=read(registered.parent/'registration.json')
    if sha(registered.read_bytes())!=registration['plan_sha256']:
        raise ValueError('original registered plan changed')
    if read(registered)!=read(study/'plan.json'):raise ValueError('source and completed study plans differ')
    for name,expected in registration['sources'].items():
        if sha(inside(worker,name).read_bytes())!=expected:raise ValueError('frozen scientific source changed: '+name)
    bound=dict(plan_sha256=registration['plan_sha256'],source_sha256=sha(encoded(registration['sources'])))
    if bound!=proof['registration']:raise ValueError('study used another frozen registration')
    head=subprocess.check_output(['git','-C',str(worker),'rev-parse','HEAD'],text=True).strip()
    if head!=proof['source_ref']:raise ValueError('worker HEAD differs from audited scientific source')
    specification=importlib.util.spec_from_file_location('registered_loss_resolver',resolver)
    module=importlib.util.module_from_spec(specification);specification.loader.exec_module(module)
    plan=module.prepare(study,None)
    if (not plan['complete_archive'] or plan['adoption'] or plan['registration']!=bound
            or plan['prior_diagnosis'] is not None or any(j['reuse'] for j in plan['jobs'])):
        raise ValueError('complete original registered diagnostic plan required')
    if len(plan['jobs'])!=plan['pending_evaluations'] or len(plan['jobs'])!=plan['required_evaluations']:
        raise ValueError('diagnostic count mismatch')
    inputs={}
    with zipfile.ZipFile(study/'checkpoints.zip') as packed:
        for job in plan['jobs']:
            name=job['input_archive_member'];raw=packed.read(name)
            if sha(raw)!=job['input_sha256']:raise ValueError('factual input checksum mismatch')
            saved=json.loads(raw)
            if (not saved['complete'] or saved['trace'][job['tick']]['before_owner']!=job['factual_owner']
                    or saved['configuration']['registration']!=bound):
                raise ValueError('incomplete or substituted factual owner')
            inputs[name]=raw
    plan_raw=encoded(plan)
    manifest=dict(worker=str(worker),source_plan=str(registered.relative_to(worker)).replace('\\','/'),
        source_ref=head,plan_sha256=sha(plan_raw),driver_sha256=sha(driver.read_bytes()),
        probe_sha256=sha(probe.read_bytes()),study_verification_sha256=sha((study/'verification.json').read_bytes()),
        resolver_sha256=sha(resolver.read_bytes()),preparer_sha256=sha(Path(__file__).read_bytes()))
    # Keep the temporary name short enough for native Windows path limits.
    staging=output.parent/('.prep-'+uuid.uuid4().hex[:12])
    staging.mkdir(parents=True)
    (staging/'plan.json').write_bytes(plan_raw)
    (staging/'manifest.json').write_bytes(encoded(manifest))
    (staging/'driver.py').write_bytes(driver.read_bytes())
    (staging/'probe_registered_owner.py').write_bytes(probe.read_bytes())
    (staging/'resolver.py').write_bytes(resolver.read_bytes())
    for name,raw in inputs.items():
        target=inside(staging/'inputs',name);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        if sha(target.read_bytes())!=sha(raw):raise ValueError('prepared input write mismatch')
    receipt=dict(prepared_complete=True,executed=False,adoption=False,root_evaluations=len(plan['jobs']),
        roots=len({(j['id'],j['tick']) for j in plan['jobs']}),loss_ids=[r['id'] for r in plan['losses']],
        manifest_sha256=sha(encoded(manifest)),study_verification_sha256=manifest['study_verification_sha256'])
    (staging/'prepared.json').write_bytes(encoded(receipt))
    if staging.resolve().parent!=output.parent or output.resolve().parent!=output.parent:
        raise ValueError('prepared directory must stay within the requested parent')
    staging.rename(output)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('study','worker','source-plan','resolver','driver','probe','output'):
        parser.add_argument('--'+name,required=True)
    print(json.dumps(prepare(**vars(parser.parse_args()))))
