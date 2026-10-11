"""Assemble completed phase accounting without pooling unlike game outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

STUDIES={
    'discovery-clock-corrected':'a1d70f8',
    'identity-confirmation':'e60a3e6',
    'public-transfer-corrected':'f313945',
    'public-confirmation':'48e7da4',
    'searched-rival-contract-corrected':'58b1f1b',
    'public-rival-uncertainty':'afcf12f',
    'actual-oracle':'e702ab3',
}


def assemble(root):
    root=Path(root);repo=Path(__file__).resolve().parents[1];records={};totals={}
    for study,commit in STUDIES.items():
        folder=root/study
        result=json.loads((folder/'evaluation.json').read_text(encoding='utf-8'))
        registered=json.loads((folder/'preregister.json').read_text(encoding='utf-8'));assert result['plan']==registered
        sources=json.loads((folder/'source_snapshot.json').read_text(encoding='utf-8'))
        tree=subprocess.check_output(['git','ls-tree','-r',commit,'reflex','tools'],cwd=repo).decode().splitlines()
        blobs={line.split('\t',1)[1]:line.split()[2] for line in tree}
        for name,sha in sources.items():
            data=(folder/'_source'/name).read_bytes();assert hashlib.sha256(data).hexdigest()==sha,name
            git_hash=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            assert blobs[name]==git_hash,(study,name,commit)
        if study=='actual-oracle':
            records[study]=dict(source_commit=commit,cases=result['reference_replays'],alternative_branches=result['alternative_branches'],
                summary=result['summary'],primary_games=0,
                scope='reuses registered source interventions; exact original-root replay; hindsight actual world not expected action value')
            continue
        if 'matches' in result:
            proof=json.loads((folder/'analysis.json').read_text(encoding='utf-8'))
            counts=proof['audit']['counts'];games=counts['games'];contrasts=proof['contrasts'];duplicates=0
        else:
            counts=json.loads((folder/'audit.json').read_text(encoding='utf-8'))['counts'];games=counts['primary_games']
            contrasts=result.get('contrasts',[]);duplicates=result.get('duplicate_baseline_replays',0)
        for key,value in counts.items():totals[key]=totals.get(key,0)+value
        records[study]=dict(source_commit=commit,primary_games=games,duplicate_baseline_replays=duplicates,
            independent_units=len(registered['seeds']),profiles=registered.get('profiles',['growth','steady','care','ego']),
            summary=proof['summary'] if 'matches' in result else result.get('summary',result.get('rates')),
            contrasts=[{k:v for k,v in c.items() if k!='series_differences'} for c in contrasts],
            audit_counts=counts,evaluation_sha256=hashlib.sha256((folder/'evaluation.json').read_bytes()).hexdigest(),
            source_snapshot_sha256=hashlib.sha256((folder/'source_snapshot.json').read_bytes()).hexdigest())
    output=dict(version='actual-persona-policy-rollout-milestone-v1',studies=records,
        primary_games=sum(r['primary_games'] for r in records.values()),
        duplicate_baseline_replays=sum(r.get('duplicate_baseline_replays',0) for r in records.values()),audit_totals=totals,
        components=['paired terminal policy evaluator','public and owned-state future-controller adapters','searched anonymous rival priors','public rival-strength mixture'],
        kept=['fixed traits and values','existing needs/Policy/progress rules','same actual baseline CPU budgets and public information','default original controllers'],
        changed='optional match-driver controller hook and new modules; no prior numerical evaluator rewritten',
        withheld=['broad default switch','online learning for new searched rival semantics','continuous upgraded-policy claims','human-level claims','affect model expansion'],
        limits=['single root intervention then incumbent actual future','source cohorts differ in samples and own-seed treatment; do not pool effects',
                'few independent seed clusters despite many games','sampling uncertainty excludes model mismatch','no new planner efficacy verified in combat/auction/resource genres'])
    (root/'implementation.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(primary_games=output['primary_games'],duplicate_baseline_replays=output['duplicate_baseline_replays'],audit_totals=totals),indent=2))
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();assemble(a.root)
