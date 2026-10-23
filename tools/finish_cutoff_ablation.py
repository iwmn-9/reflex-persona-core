"""One finite H32-only ablation of a registered cutoff-consensus development run.

Reuse its audited same-runtime H16 controls, without rerunning or relabeling
them as independent evidence. Playing source, conditions and model seeds are
identical; only the minimum cutoff becomes 32. No automatic adoption.
"""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed

def encoded(x):return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def sha(x):return hashlib.sha256(x).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--workers',type=int,default=6)
    a=p.parse_args();root=Path(a.root).resolve();base=Path.cwd();registration=json.loads((root/'registration.json').read_bytes())
    assert sha(Path(__file__).read_bytes())==registration['manager_sha256']
    source=base/registration['development'];state=dict(complete=False,adoption=False,pid=os.getpid())
    def save(**values):
        state.update(values,updated=datetime.now(timezone.utc).isoformat())
        tmp=root/'status.new';tmp.write_bytes(encoded(state));tmp.replace(root/'status.json')
    save(phase='waiting-development')
    try:
        while True:
            status=json.loads((source/'status.json').read_bytes())
            if status.get('error'):raise RuntimeError(status['error'])
            if status['phase']=='screen-regression-needs-analysis':
                save(complete=True,phase='not-started-preparation-rejected');return
            if status['complete']:break
            time.sleep(45)
        source=source/'worlds'
        manifest=json.loads((root/'source_snapshot.json').read_bytes())
        assert sha(encoded(manifest))==registration['source']
        assert manifest==json.loads((source/'source_snapshot.json').read_bytes())
        for name,value in manifest.items():assert sha((root/'_source'/name).read_bytes())==value
        proof=json.loads((source/'verification.json').read_bytes())
        assert proof['complete'] and proof['actual_owner_replayed'] and proof['source_verified']
        development=json.loads((source/'plan.json').read_bytes())
        assert sha((source/'plan.json').read_bytes())==registration['development_plan_sha256']
        specs=[];(root/'references').mkdir(exist_ok=True)
        for original in development['specs']:
            data=(source/original['id']/'baseline/result.json').read_bytes();baseline=json.loads(data)
            assert baseline['complete'] and baseline['configuration']['spec']==original
            archive='references/'+original['id']+'.json';(root/archive).write_bytes(data)
            spec=dict(original,controller=dict(original['controller'],minimum_horizon=32),reference=dict(
                archive=archive,sha256=sha(data),original_source=development['source'],
                original_path=str((source/original['id']/'baseline/result.json').relative_to(base)),
                audited_development_plan_sha256=registration['development_plan_sha256']))
            specs.append(spec)
        plan=dict(version=1,source=registration['source'],specs=specs,
            purpose=__doc__,criteria='All 8 known conditions, every loss retained; not an adoption cohort.')
        target=root/'plan.json'
        if target.exists():assert json.loads(target.read_bytes())==plan
        else:target.write_bytes(encoded(plan))
        sys.path.insert(0,str(root/'_source'))
        from tools.run_continuation_portfolio import pair
        save(phase='longitudinal-ablation');rows=[]
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for future in as_completed([pool.submit(pair,root,spec) for spec in specs]):
                rows.append(future.result())
                (root/'summary.json').write_bytes(encoded(dict(complete=len(rows)==len(specs),
                    completed_pairs=len(rows),registered_pairs=len(specs),rows=sorted(rows,key=lambda r:r['id']))))
                save(completed_pairs=len(rows),registered_pairs=len(specs))
        save(phase='auditing')
        subprocess.run([sys.executable,str(root/'_source/tools/audit_continuation_portfolio.py'),
            '--root',str(root),'--publish',str(base/registration['evidence'])],check=True,cwd=base)
        comparison=[]
        primary={r['id']:r for r in json.loads((source/'summary.json').read_bytes())['rows']}
        for row in rows:
            baseline=json.loads((root/row['spec']['reference']['archive']).read_bytes())
            long=json.loads((root/row['id']/'candidate/result.json').read_bytes())
            assert baseline['runtime']==long['runtime']
            comparison.append(dict(id=row['id'],spec=row['spec'],h16=baseline['score'],h32=long['score'],
                consensus=primary[row['id']]['scores'][1],consensus_minus_h32=primary[row['id']]['scores'][1]-long['score'],
                conditional_consensus_minus_h32=primary[row['id']]['conditional_delta']-row['conditional_delta']))
        output=dict(complete=True,rows=comparison,adoption=False,limit=__doc__)
        (root/'comparison.json').write_bytes(encoded(output))
        (base/registration['evidence']/'comparison.json').write_bytes(encoded(output))
        save(complete=True,phase='audited-needs-review')
    except Exception as error:
        save(error=repr(error));raise

if __name__=='__main__':main()
