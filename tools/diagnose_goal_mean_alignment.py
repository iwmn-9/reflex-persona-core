"""Offline causal check of an existing forecast's objective-mean estimator.

Only an isolated numeric selector is executed. No world trajectory, forecast,
factual learning or production configuration is changed.
For plans with a common objective time weight across all their branches, the
already retained projected goal mean also determines its subjective goal mean.
Mixed time weights need branch-level conditional estimates absent from this
packet and are explicitly unsupported, never approximated or omitted.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import platform
from pathlib import Path
import sys


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def diagnose(worker, source_plan, probe, source, output):
    worker, probe, source, output = map(Path, (worker, probe, source, output))
    if output.exists():
        raise FileExistsError('retain prior diagnosis')
    row = json.loads(source.read_bytes())
    assert checksum(probe) == row['diagnostic_source_sha256']
    os.environ.update(json.loads((worker / source_plan).read_bytes())['arithmetic'])
    os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    sys.path.insert(0, str(worker.resolve()))
    import numpy as np
    from reflex.core import Policy
    from reflex.deliberation import JointForecast, select
    from reflex.return_means import ReturnMeans
    from reflex.valuation import ReturnFrame
    from tools.run_adverse_recovery import verify_registration
    assert verify_registration(worker / source_plan) == row['registration']
    assert dict(python=sys.version, numpy=np.__version__, system=platform.system(),
                machine=platform.machine()) == row['runtime']
    module_spec = importlib.util.spec_from_file_location('original_diagnostic_probe', probe)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    assert module.personal_ranking(row) == row['personal_ranking']

    packet, audit = row['selection_inputs'], row['deliberation']
    metadata = audit['metadata']
    if not metadata.get('subjective_settlement_semantics', '').startswith('absolute terminal outcome once'):
        raise ValueError('this diagnostic requires the registered absolute-terminal objective convention')
    assert metadata['objective_basis']['origin'] is not None
    frame_record = metadata['return_frame']
    frame = ReturnFrame(tuple(frame_record['owners']), tuple(map(tuple, frame_record['effects'])),
                        tuple(frame_record['root_terms']), frame_record['unit'])
    means_record = metadata['cumulative_mean_correction']
    means = ReturnMeans(tuple(means_record['owners']), tuple(means_record['plans']),
        tuple(tuple(map(tuple, owner)) for owner in means_record['corrections']))
    assert len(frame.owners) == len(means.owners) == 1
    assert not any(packet['extra_selectors'].values())
    estimates = metadata['purpose_mean_estimator']['roots']
    deltas, unsupported = {}, {}
    for plan in means.plans:
        weights = {1. if endpoint['status'] != 'running' else
                   metadata['discount'] ** (endpoint['settlement_step'] - 1)
                   for endpoint in metadata['endpoint'][plan]}
        if len(weights) != 1:
            unsupported[plan] = sorted(weights)
            continue
        delta = row['purpose_by_plan'][plan] - estimates[plan]['sampled']
        # The original endpoint difference is transported at half scale. The
        # owned return frame restores 2 / objective_unit before linear scoring.
        deltas[plan] = delta * next(iter(weights)) * frame.effects[0][0] / 2.
    receipt = dict(adoption=False, id=row['id'], tick=row['tick'], seeds=row['seeds'],
        factual_owner=row['factual_owner'], input_sha256=checksum(source),
        registration=row['registration'], runtime=row['runtime'],
        diagnostic_sha256=checksum(__file__), original_selection_exact=True,
        supported=not unsupported, unsupported_mixed_time_weights=unsupported,
        scope='Same stored forecasts and persona; align only the linear objective mean. '
              'No new prediction, world continuation or evidence of better actual play.')
    if unsupported:
        output.write_text(json.dumps(receipt, sort_keys=True, allow_nan=False), encoding='utf-8')
        return receipt

    class Capture(Policy):
        def decide(self, batch, *args, **kwargs):
            result = super().decide(batch, *args, **kwargs)
            self.captured = batch, result
            return result

    base = packet['base_context']
    allowed = {a['id'] for a in base['actions'] if a['legal'] and not a['known_failure']}
    if packet['progress'] and packet['progress']['applied']:
        allowed -= set(packet['progress']['blocked'])

    def choose(returns):
        policy = Capture(**packet['policy'])
        meta = copy.deepcopy(metadata)
        meta['cumulative_mean_correction'] = returns.record()
        forecast = JointForecast(tuple(packet['forecast_contexts']),
            {k: tuple(v) for k, v in row['roots'].items()}, row['purpose_by_plan'],
            audit['horizon'], audit['max_regret'], meta, packet['target'],
            return_frame=frame, return_means=returns, outcome_limit=packet['outcome_limit'])
        decisions, result = select([base], forecast, policy, [allowed])
        return decisions[0], result, policy.captured

    before, old_audit, (batch, old_numeric) = choose(means)
    assert before == row['decision']
    changed = tuple((r[0] + deltas[plan], *r[1:])
                    for plan, r in zip(means.plans, means.corrections[0]))
    corrected = ReturnMeans(means.owners, means.plans, (changed,))
    after, new_audit, (new_batch, new_numeric) = choose(corrected)
    assert batch.ids == new_batch.ids
    assert np.array_equal(old_numeric.eligible, new_numeric.eligible)
    assert np.array_equal(old_numeric.primary, new_numeric.primary)
    assert np.array_equal(old_numeric.mode, new_numeric.mode)
    assert np.array_equal(old_numeric.features[:, :, 1:], new_numeric.features[:, :, 1:])
    for key in ('best_purpose', 'max_regret', 'progress_excluded', 'purpose_excluded'):
        assert old_audit[key] == new_audit[key]
    receipt.update(action_before=before['action_id'], action_after=after['action_id'],
        plan_before=old_audit['selected_plan'], plan_after=new_audit['selected_plan'],
        purpose_before=old_audit['selected_purpose'], purpose_after=new_audit['selected_purpose'],
        personality_risk_intrinsic_features_and_masks_unchanged=True,
        plans={plan: dict(root=row['roots'][plan][0], purpose=row['purpose_by_plan'][plan],
            eligible=bool(old_numeric.eligible[0, i]),
            old_objective_mean=float(old_numeric.features[0, i, 0]),
            new_objective_mean=float(new_numeric.features[0, i, 0]),
            added_objective_mean=deltas[plan],
            old_personal_score=float(old_numeric.scores[0, i]),
            new_personal_score=float(new_numeric.scores[0, i]))
            for i, plan in enumerate(batch.ids[0])})
    output.write_text(json.dumps(receipt, sort_keys=True, allow_nan=False), encoding='utf-8')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('worker', 'source-plan', 'probe', 'source', 'output'):
        parser.add_argument('--' + key, required=True)
    receipt = diagnose(**vars(parser.parse_args()))
    print(json.dumps({k: receipt[k] for k in ('id', 'tick', 'seeds', 'supported')}
        | {k: receipt[k] for k in ('action_before', 'action_after') if k in receipt}))
