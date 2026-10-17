"""Check the original confirmation and its preregistered default-cadence control."""
import hashlib,json,sys
from pathlib import Path
W=Path(__file__).resolve().parents[2];R=W/'github-preview/reflex-persona-core';sys.path.insert(0,str(R))
from tools.compare_owned_studies import compare
from reflex.laboratory import PROFILES
original=R/'results/owner_recovery_confirmation';extra=R/'results/owner_periodic_eight'
plan=json.loads((R/'results/owner_recovery_confirmation_PLAN.json').read_text())
supp=json.loads((R/'results/owner_periodic_eight_PLAN.json').read_text())
assert supp['original_plan_sha256']==hashlib.sha256((R/'evidence/owner_continuity/confirmation_plan.json').read_bytes()).hexdigest()
assert supp['source_core']==plan['source_core'] and supp['runtime']==plan['runtime']
for root in (original,extra):assert json.loads((root/'workflow.json').read_text())['conclusion']=='success'
candidate=plan['candidate'];baselines=[f'{arm}@{interval}' for interval in (7,8) for arm in plan['baselines']]

def check(stage,studies,worlds,profiles,regimes,turns):
    expected={(game,PROFILES[pid]['id'],seed,regime,turns,arm,interval)
        for game,seed in worlds for pid in profiles for regime in regimes
        for arm,interval in [(candidate,1)]+[(b,i) for i in (7,8) for b in plan['baselines']]}
    actual=[]
    for study in studies:
        source=json.loads((study/'source_snapshot.json').read_text())
        assert {k:v for k,v in source.items() if k.startswith('reflex/')}==plan['source_core']
        env=json.loads((study/'environment.json').read_text())
        assert env['python'].split()[0]==plan['runtime']['python'] and env['numpy']==plan['runtime']['numpy']
        for row in map(json.loads,(study/'trajectories.jsonl').read_text().splitlines()):
            actual.append(tuple(row[k] for k in ('game','profile','seed','regime','turns','arm','interval')))
    assert len(actual)==len(set(actual)) and set(actual)==expected,('full cadence matrix incomplete',stage,len(actual),len(expected))
    result=compare(studies,[(b,candidate+'@1') for b in baselines])
    result.update(complete=True,registered_runs=len(expected),negative_group_means=[s for s in result['summaries'] if s['mean_delta']< -1e-12],
        note='The same candidate executions are reused across four comparator configurations; these are not additional independent world trials.')
    (extra/(stage+'_all_cadences.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(stage=stage,runs=result['runs'],conditions=result['conditions'],negative_groups=len(result['negative_group_means']),
        summaries=[s for s in result['summaries'] if s['group_by']=='all']),indent=2))

studies=[original/f'owner-recovery-confirm-{w["game"]}-{w["seed"]}-{group}'/part
    for w in plan['worlds'] for group in plan['profile_groups'] for part in ('candidate','baseline')]
studies += [extra/f'owner-periodic-eight-{w["game"]}-{w["seed"]}-{group}'/'baseline'
    for w in plan['worlds'] for group in plan['profile_groups']]
check('primary',studies,[(w['game'],w['seed']) for w in plan['worlds']],plan['profiles'],plan['regimes'],plan['turns'])
stress=plan['stress']
studies=[original/f'owner-recovery-stress-{game}'/part for game in stress['games'] for part in ('candidate','baseline')]
studies += [extra/f'owner-periodic-eight-stress-{game}'/'baseline' for game in stress['games']]
check('stress',studies,[(g,stress['seed']) for g in stress['games']],stress['profiles'],[stress['regime']],stress['turns'])
