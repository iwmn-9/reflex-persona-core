"""Finite numerical review of registered studies; no dispatch or adoption."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args()
    root=Path(args.root).resolve();base=Path.cwd();cfg=json.loads((root/'configuration.json').read_bytes())
    assert hashlib.sha256((root/'reviewer.py').read_bytes()).hexdigest()==cfg['reviewer_sha256']
    status=dict(complete=False,pid=os.getpid(),adoption=False,phase='waiting-development')
    def save(**extra):
        status.update(extra,updated=datetime.now(timezone.utc).isoformat())
        temp=root/'status.new';temp.write_text(json.dumps(status,indent=2));temp.replace(root/'status.json')
    def read(path):return json.loads(path.read_bytes())
    save()
    try:
        for stage in ('development','confirmation'):
            location=base/cfg[stage]
            while True:
                state=read(location/'status.json')
                if state.get('error'):raise RuntimeError(stage+': '+state['error'])
                if state['phase'] in ('screen-regression-needs-analysis','not-started-gate-rejected'):
                    save(complete=True,phase='candidate-rejected-needs-analysis',reason=state);return
                if state['complete']:break
                time.sleep(45)
            comparison=location/'worlds' if stage=='development' else location
            output=comparison/'quality_review.json'
            subprocess.run([sys.executable,str(root/'reviewer.py'),'--root',str(comparison),
                '--output',str(output)],check=True,cwd=base)
            review=read(output)
            (base/cfg[stage+'_evidence']/'quality_review.json').write_bytes(output.read_bytes())
            save(**{stage+'_review':str(output)},phase='waiting-confirmation' if stage=='development' else 'waiting-original-audit')
            if stage=='development' and not review['inherited_quality_requirements']['checks']['no_family_loss']:
                save(complete=True,phase='candidate-rejected-needs-analysis');return
        ablation=read(base/cfg['ablation']/'status.json')
        assert ablation['complete'] and ablation['phase']=='audited-needs-review' and not ablation.get('error')
        while True:
            original=read(base/cfg['original'])
            ledger=read(base/'results/registered_world_completion/state.json')
            if ledger.get('error'):raise RuntimeError('Original supervisor: '+ledger['error'])
            if all(x['complete'] for x in original.values()):break
            time.sleep(45)
        save(complete=True,phase='all-evidence-ready-for-adoption-review',
            quality_passed=review['inherited_quality_requirements']['passed'],
            note='Review all individual losses, personality behavior and H32 ablation. No automatic adoption or human-level claim.')
    except Exception as error:
        save(error=repr(error));raise

if __name__=='__main__':main()
