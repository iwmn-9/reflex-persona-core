"""One finite frozen experiment pipeline. No scheduler or automatic new study."""
import argparse
import json
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.freeze_experiment import freeze
from tools.owned_checkpoint import encoded,sha
from tools.run_continuation_portfolio import prepare_references


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();root=Path(args.root).resolve();root.mkdir(parents=True,exist_ok=True)
    base=Path(__file__).resolve().parents[1];status=dict(complete=False,phase='preparing',pid=__import__('os').getpid())
    def save(**values):
        status.update(values,updated=datetime.now(timezone.utc).isoformat())
        tmp=root/'status.new';tmp.write_bytes(encoded(status));tmp.replace(root/'status.json')
    save()
    try:
        screen=root/'screen';worlds=root/'worlds'
        if not (worlds/'reference_specs.json').exists():
            prepare_references(worlds,base/'results/long_horizon_sampling_check_confirmation/summary.json')
        for target in (screen,worlds):
            if not (target/'_source').exists():freeze(target)
        manifests=[json.loads((path/'source_snapshot.json').read_bytes()) for path in (screen,worlds)]
        assert manifests[0]==manifests[1],'screen and world actor sources must match'
        save(source=sha(encoded(manifests[0])),screen=str(screen),worlds=str(worlds))
        for name,path,world_mode in (('screen',screen,False),('worlds',worlds,True)):
            save(phase=name,error=None)
            command=[sys.executable,'-u',str(path/'_source/tools/run_continuation_portfolio.py'),
                '--root',str(path),'--frozen','--workers',str(args.workers)]
            if world_mode:command.append('--worlds')
            subprocess.run(command,check=True,cwd=base)
            data=json.loads((path/'summary.json').read_bytes())
            assert data['complete'] and data['completed_pairs']==data['registered_pairs']
            destination=base/'evidence/long_horizon'/('continuation_portfolio_v2' if not world_mode else 'continuation_portfolio_worlds')
            save(phase=name+'-audit')
            subprocess.run([sys.executable,str(path/'_source/tools/audit_continuation_portfolio.py'),
                '--root',str(path),'--publish',str(destination)],check=True,cwd=base)
            if not world_mode:
                if any(r['delta'] < -1e-12 for r in data['rows']):
                    save(phase='screen-regression-needs-analysis');return
        data=json.loads((worlds/'summary.json').read_bytes())
        groups=[]
        for game in sorted({r['spec']['game'] for r in data['rows']}):
            rows=[r for r in data['rows'] if r['spec']['game']==game]
            groups.append(dict(game=game,conditions=len(rows),mean_delta=sum(r['delta'] for r in rows)/len(rows),
                conditional_delta=sum(r['conditional_delta'] for r in rows)/len(rows),
                worse=sum(r['delta'] < -1e-12 for r in rows)))
        save(complete=True,phase='audited-needs-adoption-review',families=groups,
            adoption=False,note='Known-case diagnostic, not fresh independent acceptance. No default changed.')
    except Exception as exc:
        save(phase=status['phase']+'-failed',error=repr(exc));raise


if __name__=='__main__':main()
