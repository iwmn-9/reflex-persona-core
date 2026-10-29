"""Research opt-in: apply the owner's single complete model without an advice veto.

The changed key binds save/restore to these selection semantics. Prediction,
learning, actual feedback and imagined learning retain the same EventReader.
This is not a new model or a claim that its forecast is correct.
"""
import copy
from dataclasses import replace

from reflex.core import digest
from reflex.event_reader import EventReader


class CurrentModelReader(EventReader):
    @classmethod
    def from_reader(cls, reader, scope):
        if type(reader) is not EventReader or digest(scope)!=reader.owner:
            raise ValueError('matching ordinary public event reader required')
        new=cls(scope,reader.learner.names,reader.models_for,reader.outcome_for,
            legal_for=reader.legal_for,model_id=reader.learner.identity,
            known_conditionals=True,factors=reader.learner.factors,
            features_for=reader.features_for,key='current-model:'+reader.key)
        # Constructor checks the public callbacks. Replace its temporary
        # learner before use so the exact original model identity is retained.
        new.learner=copy.deepcopy(reader.learner)
        new.memory=copy.deepcopy(reader.memory)
        new.last_tick=reader.last_tick
        return new

    def __call__(self, context, cap):
        reading=super().__call__(context,cap)
        return None if reading is None else replace(reading,selection='current_model')
