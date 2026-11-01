"""Research control: stop real experience updates without stopping the person.

Feedback must resolve normally before this operation. Needs, intent, progress,
pressure, world and clocks keep advancing. Hypothetical learning inside a
forecast remains enabled in both arms. Social-memory ablations are unsupported.
"""
import copy
from reflex.decision_loop import DecisionLoop


class FrozenExperience:
    def __init__(self, actor):
        self.baseline = copy.deepcopy(actor.record())
        predictor = self.baseline.get('predictor')
        if not isinstance(predictor, dict) or predictor.get('version') != 'event-reader-v1':
            raise ValueError('registered event-reader control required')
        if self.baseline.get('social') is not None:
            raise ValueError('social memory needs its own declared ablation contract')

    def after_feedback(self, actor, context):
        current = actor.record()  # Reject unresolved tickets before changing anything.
        base = self.baseline
        for key in ('scope', 'personality', 'values', 'seed', 'policy_residual', 'read_control',
                    'principle_priority', 'policy_commitments', 'decision_metric', 'response_adoption'):
            if current.get(key) != base.get(key):
                raise ValueError('frozen experience owner or policy changed: ' + key)
        if current['last_tick'] <= base['last_tick']:
            raise ValueError('actual post-boundary feedback required')
        if current['memory']['capacity'] != base['memory']['capacity'] or current.get('social') is not None:
            raise ValueError('experience storage or social contract changed')
        expected = copy.deepcopy(current)
        expected['predictor'] = copy.deepcopy(base['predictor'])
        expected['memory']['entries'] = copy.deepcopy(base['memory']['entries'])
        expected['evidence'] = copy.deepcopy(base['evidence'])
        # OutcomeMemory's clock must match the person's current clock even when
        # its old samples are retained. EventReader retains its own evidence clock.
        restored = DecisionLoop.from_record(context, expected, predictor=actor.predictor)
        if restored.record() != expected:
            raise ValueError('lossless frozen experience restoration required')
        return restored
