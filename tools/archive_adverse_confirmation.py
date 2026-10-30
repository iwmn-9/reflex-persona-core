"""Archive all reserved confirmation arms without development-only labels.

Reuses the existing complete source/history audit unchanged. The final receipt
contains both registered controls and never turns numeric passage into adoption.
The ongoing development collector does not load or call this module.
"""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.archive_adverse_study import archive as audit_study
from tools.analyze_adverse_results import analyze
from tools.owned_checkpoint import encoded, sha
from tools.review_adverse_confirmation import reserved_specs, review


def validate_plan(plan):
    if plan['specs'] != reserved_specs():
        raise ValueError('all original reserved conditions, seeds and turns required')
    candidate = plan['candidate_arm']
    if (candidate in ('original16', 'stable') or len(plan['arms']) != 3
            or set(plan['arms']) != {'original16', 'stable', candidate}):
        raise ValueError('exactly two original controls and one candidate required')


def confirmation_metadata(proof, comparison):
    """Require a complete audit even when the numeric quality gate fails."""
    if (not proof['complete'] or proof['pairs'] != 40
            or proof['executions'] != 120 or proof['actual_steps'] != 15360):
        raise ValueError('complete source/history audit for all 120 executions required')
    if not all(comparison['checks'][ref + '_complete'] for ref in ('original16', 'stable')):
        raise ValueError('both complete comparisons required')
    result = dict(proof)
    result.pop('development_gate_passed')
    result.update(stage='reserved_independent_confirmation', conditions=40,
        world_groups=10, profiles_per_world=4,
        independent_confirmation='complete_audited',
        numeric_gate_passed=comparison['numeric_gate_passed'],
        adoption=False, remaining_review=comparison['remaining_review'],
        source_history_auditor_sha256=proof['auditor_sha256'],
        auditor_sha256=sha(Path(__file__).read_bytes()),
        limitations=[proof['limitations'][0],
            'Forty conditions are ten game/world groups crossed with four personas, not forty independent world draws.',
            'Numeric passage alone does not adopt. Every loss against either control, personality effects and prior development evidence still require review.'])
    return result


def clean_comparisons(comparison):
    result = dict(comparison)
    result['comparisons'] = {}
    for reference, summary in comparison['comparisons'].items():
        clean = {k: v for k, v in summary.items()
                 if k not in ('development_gate_passed', 'gate_checks', 'independent_confirmation')}
        result['comparisons'][reference] = dict(clean, reference_arm=reference)
    return result


def archive_confirmation(root, plan_path, source_ref, output):
    root, plan_path, output = Path(root), Path(plan_path), Path(output)
    if output.exists():
        raise FileExistsError('do not replace an existing audit')
    plan = json.loads(plan_path.read_bytes())
    validate_plan(plan)
    comparison = review(root / 'latest', plan)
    if not all(comparison['checks'][ref + '_complete'] for ref in ('original16', 'stable')):
        raise ValueError('all reserved executions must finish before archival')
    stage = output.with_name(output.name + '.partial-' + uuid.uuid4().hex)
    # Keep a failed staging directory for diagnosis. It is never the published
    # complete output, and no raw worker result is moved, modified or deleted.
    with redirect_stdout(io.StringIO()):
        proof = audit_study(root, plan_path, source_ref, stage)
    if not all(sha((stage / name).read_bytes()) == checksum
               for name, checksum in proof['artifacts'].items()):
        raise ValueError('source/history archive changed before confirmation assembly')
    verification = confirmation_metadata(proof, comparison)
    outcome = dict(complete=True, adoption=False, comparisons={})
    audited_results = {f"adverse-{w['id']}-{w['arm']}": w['hashes']['result.json']
                       for w in proof['workers']}
    for reference in ('original16', 'stable'):
        account = analyze(root / 'latest', dict(plan, reference_arm=reference))
        if account['source_results'] != audited_results:
            raise ValueError('outcome review must use every exactly audited source result')
        account.pop('development_gate_passed')
        account['limits'][-1] = 'Complete confirmation and numeric gates are necessary but do not replace individual/personality review.'
        outcome['comparisons'][reference] = account
    (stage / 'summary.json').write_bytes(encoded(clean_comparisons(comparison)))
    (stage / 'outcome_review.json').write_bytes(encoded(outcome))
    verification['artifacts'] = {p.name: sha(p.read_bytes()) for p in stage.iterdir()
                                  if p.name != 'verification.json'}
    (stage / 'verification.json').write_bytes(encoded(verification))
    if output.exists():
        raise FileExistsError('audit destination was created concurrently')
    stage.rename(output)
    print(json.dumps({k: verification[k] for k in ('stage', 'complete', 'conditions',
        'executions', 'actual_steps', 'numeric_gate_passed', 'adoption')}), flush=True)
    return verification


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'plan', 'source-ref', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    archive_confirmation(args.root, args.plan, args.source_ref, args.output)
