"""A complete stress matrix can be reviewed while the primary jobs continue.

This never marks the full confirmation complete or changes its acceptance gate.
The final all-cadence summarizer still requires both complete workflow runs.
"""
import hashlib,json,sys
from pathlib import Path
W=Path(__file__).resolve().parents[2];R=W/'github-preview/reflex-persona-core';sys.path.insert(0,str(R))
from tools.compare_owned_studies import compare
from reflex.laboratory import PROFILES
plan=json.loads((R/'results/owner_recovery_confirmation_PLAN.json').read_text());stress=plan['stress']
supp=json.loads((R/'results/owner_periodic_eight_PLAN.json').read_text())
assert supp['original_plan_sha256']==hashlib.sha256((R/'evidence/owner_continuity/confirmation_plan.json').read_bytes()).hexdigest()
studies=[]
for game in stress['games']:
    studies += [R/'results/owner_recovery_confirmation'/f'owner-recovery-stress-{game}'/p for p in ('candidate','baseline')]
    studies.append(R/'results/owner_periodic_eight'/f'owner-periodic-eight-stress-{game}'/'baseline')
candidate=plan['candidate'];variants=[(candidate,1)]+[(b,i) for i in (7,8) for b in plan['baselines']]
expected={(g,PROFILES[p]['id'],stress['seed'],stress['regime'],stress['turns'],arm,i,plan['tail_steps'],0)
          for g in stress['games'] for p in stress['profiles'] for arm,i in variants}
actual=[]
for study in studies:
    source=json.loads((study/'source_snapshot.json').read_text())
    assert {k:v for k,v in source.items() if k.startswith('reflex/')}==plan['source_core']
    env=json.loads((study/'environment.json').read_text())
    assert env['python'].split()[0]==plan['runtime']['python'] and env['numpy']==plan['runtime']['numpy']
    for row in map(json.loads,(study/'trajectories.jsonl').read_text().splitlines()):
        actual.append(tuple(row[k] for k in ('game','profile','seed','regime','turns','arm','interval','tail_steps','offset')))
assert len(actual)==len(set(actual)) and set(actual)==expected and len(expected)==40
result=compare(studies,[(f'{b}@{i}',candidate+'@1') for i in (7,8) for b in plan['baselines']])
result.update(complete_stress_matrix=True,full_confirmation_complete=False,registered_runs=len(expected),
              note='All registered stress executions, using the same candidate against four comparator configurations. Primary confirmation continues separately.')
target=R/'results/owner_periodic_eight/stress_complete_evaluation.json'
target.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(runs=result['runs'],conditions=result['conditions'],summaries=[s for s in result['summaries'] if s['group_by']=='all']),indent=2))
