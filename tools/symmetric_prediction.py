"""Research-only exchangeable completion of categorical response templates.

Provided shapes also act as hypotheses under response-label permutations.
This assumes labels carry no prior preference in the absence of observations.
It is NOT appropriate when the host intends a semantic prior over labels.
For >4 labels this bounded experiment declines completion, retaining baseline.
"""
import math
import numpy as np
from reflex.core import digest
from reflex.segmented_prediction import SegmentedEventLearner, segment_prediction, _normalize_logs
from reflex._response_templates import complete_templates


class SymmetricEventLearner(SegmentedEventLearner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.identity = digest(['exchangeable-template-completion-v1', self.identity])

    def predict(self, memory, context, legal, models):
        self._check(memory)
        current = self._inputs(context, legal, models)
        predictions, weights = [], []
        for mask in self.masks:
            groups = {}
            for row in memory.history:
                key = (row.legal, tuple(row.context[i] for i in mask))
                groups.setdefault(key, []).append(row)
            target = (legal, tuple(context[i] for i in mask))
            target_p = None
            log_weight = -len(mask) * math.log(2)
            for key, rows in groups.items():
                labels = {v: i for i, v in enumerate(key[0])}
                sequence = tuple(labels[r.outcome] for r in rows)
                libraries = tuple(complete_templates(r.models, len(key[0])) for r in rows)
                supplied = current if key == target else rows[-1].models
                p, evidence, _ = segment_prediction(sequence, libraries, complete_templates(supplied, len(key[0])), len(key[0]))
                log_weight += evidence
                if key == target:
                    target_p = p
            if target_p is None:
                target_p = segment_prediction((), (), complete_templates(current, len(legal)), len(legal))[0]
            predictions.append(target_p)
            weights.append(log_weight)
        p = _normalize_logs(weights) @ np.array(predictions)
        return dict(zip(legal, (float(v) for v in p / p.sum())))

    def persistent_continuation(self, *args, **kwargs):
        raise ValueError('template experiment requires its declared posterior continuation')
