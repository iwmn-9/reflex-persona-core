"""Freeze one existing-model alternative with the unchanged development gates."""
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
    dest=ROOT/'evidence/long_horizon/adverse_persistent'
    dest.mkdir(exist_ok=True)
    if (dest/'registration.json').exists():raise FileExistsError('do not replace a frozen study')
    plan=dict(original,version='adverse-persistent-1',
        candidate='Existing persistent world continuation, independent H32 with unchanged factual learner and personality',
        arms=['stable','persistent'],reference_arm='stable',candidate_arm='persistent',
        predecessor_plan_sha256=sha(previous.read_bytes()),
        purpose='Same ten known conditions with fresh paired controls; diagnose future regime assumptions, never independent confirmation.',
        arithmetic=dict(NPY_DISABLE_CPU_FEATURES='AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL',
                        OPENBLAS_CORETYPE='Haswell'),
        change='Use existing world_model=persistent for the outer forecast and its existing two-step future recovery. Keep actual segmented learning, personality, H32, three model seeds, retry bank, factual replanning and all gates.',
        limitations=['Assuming the current regime persists can itself be wrong; all changing-world cases remain.',
            'The world_model option also propagates into the existing future recovery planner. This is not an outer-kernel-only causal intervention.',
            'Known-root choices and known-condition development cannot justify adoption. The original independent forty conditions remain mandatory.'])
    # Preserve every condition, gate and reserved independent seed exactly.
    assert plan['specs']==original['specs'] and plan['criteria']==original['criteria']
    assert plan['independent_confirmation']==original['independent_confirmation']
    (dest/'plan.json').write_bytes(encoded(plan))
    files=list((ROOT/'reflex').glob('*.py'))+list((ROOT/'tools').glob('*.py'))
    files.append(ROOT/'.github/workflows/adverse-persistent.yml')
    sources={p.relative_to(ROOT).as_posix():sha(p.read_bytes()) for p in sorted(files)}
    proof=dict(version='frozen-adverse-persistent-1',plan_sha256=sha((dest/'plan.json').read_bytes()),
        stable_commit=original['stable_commit'],sources=sources)
    (dest/'registration.json').write_bytes(encoded(proof))
    with zipfile.ZipFile(dest/'source.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,checksum in sources.items():
            data=(ROOT/name).read_bytes();assert sha(data)==checksum;archive.writestr(name,data)
        archive.write(dest/'plan.json','evidence/long_horizon/adverse_persistent/plan.json')
        archive.write(dest/'registration.json','evidence/long_horizon/adverse_persistent/registration.json')
    print(encoded(dict(plan_sha256=proof['plan_sha256'],source_files=len(sources),conditions=len(plan['specs']),
        executions=len(plan['specs'])*len(plan['arms']),independent_confirmation='not started')).decode())


if __name__=='__main__':register()
