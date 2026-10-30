"""Register the original ten development conditions after complete prerequisites.

The forty reserved conditions cannot be dispatched by this registration.
Existing source, old trials, adverse cases and numerical gates are retained.
"""
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.owned_checkpoint import encoded, sha


def register():
    destination = ROOT/'evidence/long_horizon/adverse_purpose_tail'
    if any((destination/n).exists() for n in ('plan.json','registration.json','source.zip')):
        raise FileExistsError('never replace a registered study')
    preparation = destination/'preparation_complete/verification.json'
    validation = destination/'validation.json'
    connection = destination/'confirmation_connection.json'
    prerequisites = {p.name: json.loads(p.read_bytes()) for p in (preparation, validation, connection)}
    p, v, c = [prerequisites[path.name] for path in (preparation, validation, connection)]
    assert p['complete'] and p['preparation_no_loss']
    assert (p['pairs'], p['executions'], p['actual_steps'], p['restores']) == (12,24,3072,144)
    assert p['exact_actual_learning_replay'] and p['fixed_persona'] and not p['adoption']
    for name, checksum in p['artifacts'].items():
        assert sha((preparation.parent/name).read_bytes()) == checksum
    assert v['complete'] and v['state'] == 'passed' and v['source_unchanged']
    assert (v['tests'], v['failures'], v['errors'], v['skipped']) == (1327,0,0,19)
    assert sha((destination/'validation_execution.zip').read_bytes()) == v['archive_sha256']
    assert c['complete'] and c['passed'] and not c['adoption']
    assert all(sha((ROOT/name).read_bytes()) == checksum for name,checksum in c['sources'].items())
    assert sha((destination/c['log']).read_bytes()) == c['log_sha256']
    # The new comparative runner adds the existing H16 control; the candidate's
    # scientific implementation must remain the one that passed the full suite.
    for name in ('reflex/_continuation_portfolio.py','reflex/_horizon_consensus.py',
                 'tools/template_transfer_prediction.py','tools/continuing_games.py'):
        checked = subprocess.check_output(['git','show',f"{v['source']}:{name}"], cwd=ROOT)
        assert (ROOT/name).read_bytes() == checked
    previous = ROOT/'evidence/long_horizon/adverse_transfer/plan.json'
    original = json.loads(previous.read_bytes())
    plan = dict(original, version='adverse-purpose-tail-1',
        candidate='H32 template-transfer owner with existing public-purpose H1 fallback inside hypothetical continuation',
        arms=['stable','transfer_purpose'], reference_arm='stable', candidate_arm='transfer_purpose',
        purpose='Same ten known development conditions, fresh paired controls and actual learning. A known causal root and regression checks do not qualify adoption.',
        predecessor_plan_sha256=sha(previous.read_bytes()),
        model_seeds=dict(stable=[701,702,703],transfer_purpose=[701,702,703]),
        execution=dict(original['execution'], reason='Finite PC-local runs; no Actions dispatch or change to acceptance.'),
        change='Relative to transfer, connect existing H1 public-purpose evaluation to hypothetical fallback when existing H2 recovery is absent. Retain persona, factual learning, public world law, forced retry proposals, H32, model seeds and cumulative preference. The original segmented stable H32 remains a fresh control.',
        prerequisites={str(path.relative_to(ROOT)):sha(path.read_bytes()) for path in (preparation,validation,connection)},
        limitations=['H1 fallback is not an exact model of a future H32 replanning owner.',
            'The known bond-versus-bet root changed, but its modeled value is not actual-world improvement.',
            'The failed transfer and consistent studies remain retained; this candidate is not a combination with the consistent reader.',
            'No game/action-name branch, personality coefficient, physical rule, actual RNG, selection gate or independent condition is changed.'])
    plan.pop('predecessor_complete_audit_sha256',None)
    assert len(plan['specs']) == 10
    assert plan['specs'] == original['specs'] and plan['criteria'] == original['criteria']
    assert plan['independent_confirmation'] == original['independent_confirmation']
    (destination/'plan.json').write_bytes(encoded(plan))
    files = [p for folder in ('reflex','tools','tests') for p in (ROOT/folder).glob('*.py')]
    sources = {p.relative_to(ROOT).as_posix():sha(p.read_bytes()) for p in sorted(files)}
    proof = dict(version='frozen-adverse-purpose-tail-1',plan_sha256=sha((destination/'plan.json').read_bytes()),
        stable_commit=original['stable_commit'],sources=sources)
    (destination/'registration.json').write_bytes(encoded(proof))
    with zipfile.ZipFile(destination/'source.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for name,checksum in sources.items():
            data=(ROOT/name).read_bytes();assert sha(data)==checksum;archive.writestr(name,data)
        for name in ('plan.json','registration.json'):
            archive.writestr(f'evidence/long_horizon/adverse_purpose_tail/{name}',(destination/name).read_bytes())
    print(encoded(dict(plan_sha256=proof['plan_sha256'],source_files=len(sources),conditions=10,
        executions=20,independent_confirmation='not started',adoption=False)).decode())


if __name__ == '__main__':
    register()
