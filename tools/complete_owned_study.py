"""Complete an interrupted registration with its original frozen runtime.

Finished rows are never rerun or selected by score. The original artifact stays
intact. Only missing whole executions run again; this is not mid-game resume.
The merged study is audited on the matching platform and records both sources.
"""
import argparse,hashlib,json,platform,shutil,subprocess,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.laboratory import PROFILES

FIELDS=('game','profile','seed','regime','turns','interval','tail_steps','offset','arm')
def key(row):return tuple(row[k] for k in FIELDS)
def missing(registration,rows):
    expected={tuple([g,PROFILES[p]['id'],s,r,t,i,h,o,arm]):(job,arm)
        for job in registration['jobs'] for g,p,s,r,t,i,h,o in [job] for arm in registration['arms']}
    assert len(expected)==len(registration['jobs'])*len(registration['arms']),'duplicate registration'
    observed=[key(row) for row in rows]
    assert len(set(observed))==len(observed) and set(observed)<=set(expected),'duplicate or unregistered execution'
    return [expected[k] for k in sorted(set(expected)-set(observed))]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('original');parser.add_argument('output')
    parser.add_argument('--original-run',required=True);args=parser.parse_args()
    original=Path(args.original).resolve();output=Path(args.output).resolve();repo=Path(__file__).resolve().parents[1]
    assert original.is_dir() and not output.exists()
    manifest=json.loads((original/'source_snapshot.json').read_text());frozen=original/'_source'
    assert all(sha(frozen/name)==value for name,value in manifest.items())
    runtime_names=[n for n in manifest if n.startswith('reflex/')]+[
        'tools/run_owned_value.py','tools/continuing_games.py','tools/audit_owned_value.py']
    assert all(sha(repo/n)==manifest[n] for n in runtime_names),'runtime changed since interrupted registration'
    environment=json.loads((original/'environment.json').read_text())
    current=dict(python=sys.version,numpy=np.__version__,system=platform.system(),machine=platform.machine(),release=platform.release())
    assert all(current[k]==environment[k] for k in ('python','numpy','system','machine')),'matching execution platform required'
    registration=json.loads((original/'registration.json').read_text())
    rows=[json.loads(line) for line in (original/'trajectories.jsonl').read_text().splitlines()]
    remaining=missing(registration,rows);assert remaining,'no missing executions'
    original_count=len(rows);output.mkdir(parents=True)
    shutil.copytree(frozen,output/'_source')
    for name in ('source_snapshot.json','registration.json','environment.json'):shutil.copyfile(original/name,output/name)
    for index,(job,arm) in enumerate(remaining):
        game,pid,seed,regime,turns,interval,tail,offset=job
        part=output/'completion_work'/str(index)
        command=[sys.executable,str(frozen/'tools/run_owned_value.py'),'--frozen','--root',str(part),
            '--games',game,'--profiles',str(pid),'--seeds',str(seed),'--regimes',regime,'--arms',arm,
            '--turns',str(turns),'--interval',str(interval),'--tail-steps',str(tail),'--offset',str(offset),
            '--workers','1','--stage',registration['stage']]
        subprocess.run(command,cwd=frozen,check=True)
        added=[json.loads(line) for line in (part/'trajectories.jsonl').read_text().splitlines()]
        assert len(added)==1 and key(added[0])==(game,PROFILES[pid]['id'],seed,regime,turns,interval,tail,offset,arm)
        rows+=added
    assert not missing(registration,rows)
    (output/'trajectories.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
    (output/'summary.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='trace'} for r in rows],indent=2)+'\n')
    provenance=dict(original_run=args.original_run,original_complete=False,original_rows=original_count,
        completed_rows=len(rows)-original_count,registration_unchanged=True,
        original_trajectory_sha256=sha(original/'trajectories.jsonl'),
        original_registration_sha256=sha(original/'registration.json'),original_source_manifest_sha256=sha(original/'source_snapshot.json'),
        original_environment=environment,completion_environment=current,completed_executions=[key(r) for r in rows[original_count:]])
    (output/'completion_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    subprocess.run([sys.executable,str(frozen/'tools/audit_owned_value.py'),str(output)],cwd=frozen,check=True)
    print(json.dumps(dict(original_rows=original_count,new_rows=len(rows)-original_count,complete=True)),flush=True)

if __name__=='__main__':main()
