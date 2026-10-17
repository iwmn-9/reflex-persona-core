"""Check the entire registered matrix before interpreting any final scores."""
import hashlib,json,sys
from pathlib import Path
W=Path(__file__).resolve().parents[2];R=W/'github-preview/reflex-persona-core';sys.path.insert(0,str(R))
from tools.compare_owned_studies import compare
from reflex.laboratory import PROFILES
root=R/'results/owner_recovery_confirmation';plan=json.loads((R/'results/owner_recovery_confirmation_PLAN.json').read_text())
workflow=json.loads((root/'workflow.json').read_text());assert workflow['conclusion']=='success'
candidate=plan['candidate'];baselines=plan['baselines']
def check(studies,expected,label):
    actual=[]
    for study in studies:
        manifest=json.loads((study/'source_snapshot.json').read_text())
        assert {k:v for k,v in manifest.items() if k.startswith('reflex/')}==plan['source_core']
        environment=json.loads((study/'environment.json').read_text())
        assert environment['python'].split()[0]==plan['runtime']['python'] and environment['numpy']==plan['runtime']['numpy']
        for row in map(json.loads,(study/'trajectories.jsonl').read_text().splitlines()):
            actual.append(tuple(row[k] for k in ('game','seed','profile','regime','turns','arm','interval')))
    assert len(actual)==len(set(actual)) and set(actual)==expected,('registered matrix mismatch',label,len(actual),len(expected))
    result=compare(studies,[(b+'@7',candidate+'@1') for b in baselines])
    result.update(complete=True,registered_runs=len(expected),workflow=workflow,
        plan_sha256=hashlib.sha256((R/'results/owner_recovery_confirmation_PLAN.json').read_bytes()).hexdigest(),
        negative_group_means=[s for s in result['summaries'] if s['mean_delta']< -1e-12])
    (root/(label+'_evaluation.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(stage=label,runs=result['runs'],conditions=result['conditions'],
        summaries=[s for s in result['summaries'] if s['group_by']=='all'],
        negative_groups=len(result['negative_group_means'])),indent=2))
expected={(w['game'],w['seed'],PROFILES[i]['id'],regime,plan['turns'],arm,1 if arm==candidate else 7)
    for w in plan['worlds'] for i in plan['profiles'] for regime in plan['regimes'] for arm in [candidate,*baselines]}
studies=[root/f'owner-recovery-confirm-{w["game"]}-{w["seed"]}-{group}'/part
    for w in plan['worlds'] for group in plan['profile_groups'] for part in ('candidate','baseline')]
check(studies,expected,'primary')
stress=plan['stress']
expected={(g,stress['seed'],PROFILES[i]['id'],stress['regime'],stress['turns'],arm,1 if arm==candidate else 7)
    for g in stress['games'] for i in stress['profiles'] for arm in [candidate,*baselines]}
studies=[root/f'owner-recovery-stress-{game}'/part for game in stress['games'] for part in ('candidate','baseline')]
check(studies,expected,'stress')
