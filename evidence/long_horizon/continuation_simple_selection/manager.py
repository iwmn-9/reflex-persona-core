"""Select a preregistered controller, preferring fixed H32 when adequate.

Finished development and preparation screens select a candidate for independent
confirmation, never for adoption. Comparison is same-source, same conditions;
extra cutoff voting must earn its necessity by rescuing a failed adequacy gate.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def choose_candidate(choices):
    """The order is an explicit simplicity preference, not a score search."""
    for name in ('fixed32', 'consensus'):
        evidence = choices[name]
        if not evidence['complete']:
            raise ValueError('Both complete comparisons required before selection')
    for name in ('fixed32', 'consensus'):
        evidence = choices[name]
        if evidence['preparation_no_loss'] and evidence['no_family_loss']:
            return name
    return None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    args=parser.parse_args()
    root=Path(args.root).resolve();base=Path.cwd()
    registration=json.loads((root/'registration.json').read_bytes())
    assert sha(Path(__file__).read_bytes())==registration['manager_sha256']
    for name,value in registration['files'].items():
        assert sha((root/name).read_bytes())==value
    lock=root/'execution.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,encoded(dict(pid=os.getpid(),created=datetime.now(timezone.utc).isoformat())))
    os.close(fd)
    state=dict(complete=False,adoption=False,pid=os.getpid(),phase='waiting-complete-comparison')
    def save(**values):
        state.update(values,updated=datetime.now(timezone.utc).isoformat())
        tmp=root/'status.new';tmp.write_bytes(encoded(state));tmp.replace(root/'status.json')
    save()
    try:
        for item in registration['required_studies']:
            location=base/item['root']
            while True:
                path=location/'status.json'
                current=json.loads(path.read_bytes()) if path.exists() else {}
                if current.get('error'):raise RuntimeError(str(location)+': '+current['error'])
                if current.get('complete'):
                    assert current['phase']==item['phase']
                    break
                time.sleep(45)
        sys.path.insert(0,str(root))
        from gate import verified_evidence
        choices={};proofs={}
        for name in ('fixed32','consensus'):
            candidate=registration['candidates'][name]
            screen=verified_evidence(base/candidate['screen'],registration['source'],candidate['screen_plan_sha256'])
            worlds=verified_evidence(base/candidate['worlds'],registration['source'],candidate['world_plan_sha256'])
            # Numeric reconstruction is additional to the factual owner audit.
            out=root/(name+'_quality.json')
            subprocess.run([sys.executable,str(root/'reviewer.py'),'--root',str(base/candidate['worlds_root']),
                '--output',str(out)],check=True,cwd=base)
            review=json.loads(out.read_bytes())
            choices[name]=dict(complete=True,preparation_no_loss=all(r['delta']>=-1e-12 for r in screen['rows']),
                no_family_loss=review['inherited_quality_requirements']['checks']['no_family_loss'],
                families=review['inherited_quality_requirements']['families'])
            proofs[name]=dict(screen_sha256=sha((base/candidate['screen']/'verification.json').read_bytes()),
                worlds_sha256=sha((base/candidate['worlds']/'verification.json').read_bytes()),
                numeric_review_sha256=sha(out.read_bytes()))
        selected=choose_candidate(choices)
        result=dict(complete=True,selected=selected,choices=choices,proofs=proofs,adoption=False,
            rule=registration['selection_rule'],registration_sha256=sha((root/'registration.json').read_bytes()))
        (root/'selection.json').write_bytes(encoded(result))
        dest=base/registration['evidence'];dest.mkdir(parents=True,exist_ok=True)
        for name in ('registration.json','selection.json','fixed32_quality.json','consensus_quality.json'):
            (dest/name).write_bytes((root/name).read_bytes())
        if selected is None:
            save(complete=True,phase='both-candidates-rejected',choices=choices);return
        # Each template, including all forty world conditions, was hashed before
        # independent play. Selection cannot invent a more favorable confirmation.
        template=registration['candidates'][selected]['confirmation_template']
        chosen=json.loads((root/template).read_bytes())
        target=base/registration['confirmation_root']
        assert chosen['selected_candidate']==selected
        path=target/'registration.json'
        if path.exists():assert json.loads(path.read_bytes())==chosen
        else:path.write_bytes(encoded(chosen))
        save(phase='independent-confirmation',selected=selected)
        subprocess.run([sys.executable,str(target/'manager.py'),'--root',str(target),
            '--workers',str(registration['workers'])],check=True,cwd=base)
        status=json.loads((target/'status.json').read_bytes())
        assert status['complete'] and status['phase']=='audited-needs-review' and not status.get('error')
        out=target/'quality_review.json'
        subprocess.run([sys.executable,str(root/'reviewer.py'),'--root',str(target),
            '--output',str(out)],check=True,cwd=base)
        review=json.loads(out.read_bytes())
        (base/chosen['evidence']/'quality_review.json').write_bytes(out.read_bytes())
        save(phase='waiting-original-audit',quality_passed=review['inherited_quality_requirements']['passed'])
        while True:
            ledger=json.loads((base/'results/registered_world_completion/state.json').read_bytes())
            if ledger.get('error'):raise RuntimeError('Original audit: '+ledger['error'])
            original=json.loads((base/'results/registered_evidence_status.json').read_bytes())
            if all(v['complete'] for v in original.values()):break
            time.sleep(45)
        save(complete=True,phase='all-evidence-ready-for-adoption-review',
            note='Numerical gate only. Review individual losses and persona behavior before adopting.')
    except Exception as error:
        save(error=repr(error));raise
    finally:
        lock.unlink()


if __name__=='__main__':
    main()
