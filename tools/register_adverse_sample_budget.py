"""Freeze the existing independent H32 controller with eight model samples."""
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.owned_checkpoint import encoded,sha


def register():
    previous=ROOT/'evidence/long_horizon/adverse_recovery/plan.json'
    original=json.loads(previous.read_bytes())
    prior=ROOT/'evidence/long_horizon/adverse_persistent/complete_study/verification.json'
    result=json.loads(prior.read_bytes())
    assert result['complete'] and result['pairs']==10 and not result['development_gate_passed']
    dest=ROOT/'evidence/long_horizon/adverse_sample_budget'
    dest.mkdir(exist_ok=True)
    if (dest/'registration.json').exists():raise FileExistsError('do not replace a frozen study')
    plan=dict(original,version='adverse-sample-budget-1',
        candidate='Existing posterior H32, independent samples expanded from three to eight',
        arms=['stable','sample8'],reference_arm='stable',candidate_arm='sample8',
        predecessor_plan_sha256=sha(previous.read_bytes()),
        predecessor_complete_audit_sha256=sha(prior.read_bytes()),
        purpose='Same ten known conditions with fresh controls; separate sparse forecast integration from model/personality changes. Not independent confirmation.',
        model_seeds=dict(stable=[701,702,703],sample8=list(range(701,709))),
        arithmetic=dict(NPY_DISABLE_CPU_FEATURES='AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL',
                        OPENBLAS_CORETYPE='Haswell'),
        change='Only increase the existing model-sample count. Keep posterior world model, H32, factual learner, persona, retry bank, H2 future recovery, fixed source and actual-world RNG.',
        execution=dict(maximum_finite_segments=8,slices_per_segment=4,minutes_per_slice=80,
            reason='More hypothetical paths cost more time. These are continuations of the same executions, never fresh attempts or different seeds.'),
        limitations=['More model samples cannot supply missing information or correct a wrong world law.',
            'The first three seeds are retained; the five additional consecutive seeds are fixed before this study, never selected for favorable actions.',
            'The previous gambling loss includes choices favored by observed beliefs. This study must not claim those losses are necessarily avoidable intelligence defects.',
            'Full development gates and individual review remain mandatory before the reserved forty; no promotion from a local root.'])
    assert plan['specs']==original['specs'] and plan['criteria']==original['criteria']
    assert plan['independent_confirmation']==original['independent_confirmation']
    (dest/'plan.json').write_bytes(encoded(plan))
    files=list((ROOT/'reflex').glob('*.py'))+list((ROOT/'tools').glob('*.py'))
    files.append(ROOT/'.github/workflows/adverse-sample-budget.yml')
    sources={p.relative_to(ROOT).as_posix():sha(p.read_bytes()) for p in sorted(files)}
    proof=dict(version='frozen-adverse-sample-budget-1',plan_sha256=sha((dest/'plan.json').read_bytes()),
        stable_commit=original['stable_commit'],sources=sources)
    (dest/'registration.json').write_bytes(encoded(proof))
    with zipfile.ZipFile(dest/'source.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,checksum in sources.items():
            data=(ROOT/name).read_bytes();assert sha(data)==checksum;archive.writestr(name,data)
        archive.write(dest/'plan.json','evidence/long_horizon/adverse_sample_budget/plan.json')
        archive.write(dest/'registration.json','evidence/long_horizon/adverse_sample_budget/registration.json')
    print(encoded(dict(plan_sha256=proof['plan_sha256'],source_files=len(sources),conditions=10,
        executions=20,independent_confirmation='not started')).decode())


if __name__=='__main__':register()
