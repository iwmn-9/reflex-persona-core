"""Freeze a common unknown-hypothesis transfer change, without changing gates."""
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.owned_checkpoint import encoded,sha


def register():
    previous=ROOT/'evidence/long_horizon/adverse_sample_budget/plan.json'
    original=json.loads(previous.read_bytes())
    dest=ROOT/'evidence/long_horizon/adverse_transfer'
    dest.mkdir(exist_ok=True)
    if (dest/'registration.json').exists():raise FileExistsError('do not replace a registered study')
    plan=dict(original,version='adverse-template-transfer-1',
        candidate='H32 with supplied-prior-preserving template transfer inside unknown response laws',
        arms=['stable','transfer'],reference_arm='stable',candidate_arm='transfer',
        purpose='Same ten known development conditions, fresh controls and actual learning. Separate model coverage from numerical sampling; not independent confirmation.',
        predecessor_plan_sha256=sha(previous.read_bytes()),
        model_seeds=dict(stable=[701,702,703],transfer=[701,702,703]),
        execution=dict(original['execution'],reason='Finite continuation of the same original worlds; the richer model changes inference cost, never seeds or acceptance.'),
        change='Only the segmented categorical learner changes. Keep half prior on supplied labeled hypotheses; unknown half averages transferred shapes and fresh Dirichlet laws. Same posterior model in factual learning and hypothetical continuations. Keep all persona, H32, retry and H2 settings.',
        predictor=dict(reference='SegmentedEventLearner',candidate='TemplateTransferLearner',
            maximum_completed_alphabet=4,larger_alphabets='unchanged baseline',
            supplied_prior=.5,shape_transfer_prior=.25,fresh_unknown_prior=.25),
        limitations=['This is a structural prior over categorical response probabilities, not discovery of new actions or hidden state.',
            'The fresh 128-stream prediction check retains two adverse novel-shape groups. A better prediction score does not prove better game decisions.',
            'No personality coefficient, game-name branch, real response draw, gate or independent confirmation condition is changed.',
            'The existing sample8 study continues under its original frozen source in a separate checkout; neither study substitutes for the other.'])
    assert plan['specs']==original['specs'] and plan['criteria']==original['criteria']
    assert plan['independent_confirmation']==original['independent_confirmation']
    plan.pop('predecessor_complete_audit_sha256',None)
    (dest/'plan.json').write_bytes(encoded(plan))
    files=list((ROOT/'reflex').glob('*.py'))+list((ROOT/'tools').glob('*.py'))
    files.append(ROOT/'.github/workflows/adverse-transfer.yml')
    sources={p.relative_to(ROOT).as_posix():sha(p.read_bytes()) for p in sorted(files)}
    proof=dict(version='frozen-adverse-template-transfer-1',plan_sha256=sha((dest/'plan.json').read_bytes()),
        stable_commit=original['stable_commit'],sources=sources)
    (dest/'registration.json').write_bytes(encoded(proof))
    with zipfile.ZipFile(dest/'source.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,checksum in sources.items():
            data=(ROOT/name).read_bytes();assert sha(data)==checksum;archive.writestr(name,data)
        for name in ('plan.json','registration.json'):
            archive.write(dest/name,'evidence/long_horizon/adverse_transfer/'+name)
    print(encoded(dict(plan_sha256=proof['plan_sha256'],source_files=len(sources),conditions=10,
        executions=20,independent_confirmation='not started')).decode())


if __name__=='__main__':register()
