"""Explicit Python integration of the unadopted gain-tail research candidate.

This composes existing algorithms; it does not certify long-world quality.
Stable FixedHorizonOwner and the Rust boundary keep their previous meaning.
"""
from .event_reader import EventReader
from .fixed_horizon import FixedHorizonOwner
from .template_transfer import TemplateTransferLearner


def transfer_event_reader(scope, names, models_for, outcome_for, *, legal_for,
                          model_id, known_conditionals, factors=(),
                          features_for=None, key='event-response'):
    """Create a fresh reader with the registered experimental prior.

    This is not a conversion of an experienced reader. Restore the original
    matching transfer checkpoint to resume experience; never discard history.
    The host's conditional effects and observation callbacks remain external.
    """
    reader=EventReader(scope,names,models_for,outcome_for,legal_for=legal_for,
        model_id=model_id,known_conditionals=known_conditionals,factors=factors,
        features_for=features_for,key=key,learning='segmented')
    previous=reader.learner
    reader.learner=TemplateTransferLearner(owner=reader.owner,model_id=model_id,
        factors=previous.factors,names=previous.names,window=previous.window)
    reader.memory=reader.learner.start()
    return reader


class GainHorizonOwner(FixedHorizonOwner):
    """Research composition with normal decide/resolve/checkpoint/restored APIs.

    Future fallback evaluates the public purpose and a retry proposal may end
    after its first modeled gain. Actual feedback always triggers fresh planning.
    Only H32 / seeds701..703 / recovery_cache64 belongs to the current study.
    Other budgets are configuration support, not equivalent quality evidence.
    """
    def __init__(self,actor,**options):
        if (type(getattr(actor,'predictor',None)) is not EventReader
                or type(actor.predictor.learner) is not TemplateTransferLearner):
            raise ValueError('explicit experimental transfer reader required; no automatic learner conversion')
        super().__init__(actor,**options)

    def _continuation_options(self):
        return dict(tail_purpose_reflex=True,retry_until_gain=True)

    def _checkpoint_extension(self):
        return dict(kind='experimental-gain-continuation-v1',recovery_cache=self.recovery_cache,
                    **self._continuation_options())


class RelativeGainHorizonOwner(GainHorizonOwner):
    """Unadopted correction using the paired ordinary continuation as reference.

    Common background income is not preparation's return. A retry stops when
    its modeled public purpose exceeds the same-response, same-time ordinary
    path. The same person's selector still chooses among complete proposals.
    Old gain-owner saves are deliberately incompatible with this composition.
    """
    def _continuation_options(self):
        return dict(tail_purpose_reflex=True,retry_until_gain=True,
                    retry_gain_reference='ordinary')

    def _checkpoint_extension(self):
        return dict(kind='experimental-relative-gain-continuation-v1',
                    recovery_cache=self.recovery_cache,**self._continuation_options())
