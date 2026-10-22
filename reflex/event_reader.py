"""Owned open-model prediction in the normal DecisionLoop lifecycle.

Known conditional effects belong to the game; response probabilities do not.
The existing EventLearner compares supplied hypotheses and empirical predictors
before each reveal, including distributions outside the hypothesis library.
No action utility, personality coefficient or hidden game state is learned here.
"""
from . import cloning as copy
from collections import OrderedDict

from .adaptive_dynamics import EventLearner
from .core import digest, identifier, MAX_OUTCOMES
from .decision_loop import Reading


class EventReader:
    """One public response law, shared by immediate decisions and a planner.

    `legal_for(context)` declares observable responses possible under the rules,
    NOT likely responses. `features_for` supplies <=4 categorical observations.
    All root actions must share this response law (e.g. simultaneous choices).
    Action-dependent opponents need a separate conditional model, not this API.
    `model_id` versions observation/rule/callback semantics for checkpoints.
    """
    def __init__(self, scope, names, models_for, outcome_for, *, legal_for,
                 model_id, known_conditionals, factors=(), features_for=None,
                 learned=True, key='event-response',renewal=False,learning='windows'):
        if known_conditionals is not True:
            raise ValueError('explicit known conditional effects required')
        if not all(callable(f) for f in (models_for, outcome_for, legal_for)):
            raise ValueError('pure public model, effect and legal-response callbacks required')
        if factors and not callable(features_for):
            raise ValueError('declared public factors need an observation callback')
        if features_for is not None and not callable(features_for):
            raise ValueError('public feature callback required')
        identifier(key)
        self.owner = digest(scope)
        if learning=='segmented':
            if learned is not True or renewal:raise ValueError('segmented prediction replaces window experts; it is not stacked on them')
            from .segmented_prediction import SegmentedEventLearner
            self.learner=SegmentedEventLearner(factors=factors,names=names,owner=self.owner,model_id=model_id)
        elif learning=='windows':
            self.learner = EventLearner(factors=factors, names=names, owner=self.owner,
                                        model_id=model_id, learned=learned,renewal=renewal)
        else:raise ValueError('known owned learning model required')
        self.memory = self.learner.start()
        self.models_for, self.outcome_for = models_for, outcome_for
        self.legal_for, self.features_for = legal_for, features_for
        self.key, self.last_tick = key, -1

    @property
    def observations(self):
        return self.memory.observations

    def _inputs(self, context):
        if digest(context['scope']) != self.owner:
            raise ValueError('predictor owner mismatch')
        legal = self.legal_for(copy.deepcopy(context))
        features = () if self.features_for is None else self.features_for(copy.deepcopy(context))
        models = self.models_for(copy.deepcopy(context))
        return features, legal, models

    def predict(self, context):
        """The current predictive law; not a certificate of model correctness."""
        return self.learner.predict(self.memory, *self._inputs(context))

    def response_comparison(self, context):
        # Expert loss is already scored prequentially inside EventLearner.
        # Do not gate the whole mixture against another possibly wrong model.
        return None

    def frozen_forecaster(self):
        """A detached public world law; imagined actor updates cannot change it.

        Cache only the declared factors, legal symbols and supplied hypotheses.
        The memory is fixed in this closure, so identical inputs are identical
        forecasts even when irrelevant public fields (e.g. the clock) change.
        """
        frozen=copy.deepcopy(self);cache=OrderedDict()
        def predict(context):
            inputs=frozen._inputs(context);key=digest(inputs)
            if key not in cache:
                cache[key]=frozen.learner.predict(frozen.memory,*inputs)
                if len(cache)>256:cache.popitem(last=False)
            cache.move_to_end(key)
            return copy.deepcopy(cache[key])
        return predict

    def predictive_continuation(self,*,steps=None,persistent=False):
        """Posterior world prediction from a fixed root and public future events.

        Only the segmented learner has the explicit generative model required
        here. Its factual root history is retained throughout a forecast, even
        when the imagined person's bounded memory forgets older observations.
        We integrate possible laws, not freeze their mean or replace physics
        with the person's preference/utility updates. This is conditional on
        the declared segment prior; it does not reveal the real world's law.
        """
        from .segmented_prediction import SegmentedEventLearner,SegmentMemory
        if not isinstance(self.learner,SegmentedEventLearner):
            raise ValueError('posterior world continuation requires a generative segmented model')
        if type(persistent) is not bool:raise ValueError('explicit future regime assumption required')
        root=copy.deepcopy(self);model=copy.deepcopy(root.learner)
        limit=min(root.learner.window,512-len(root.memory.history))
        steps=limit if steps is None else steps
        if type(steps) is not int or not 1<=steps<=limit:
            raise ValueError('forecast must fit the branch and retained-root world memories')
        model.window=512;model.identity=digest(['predictive-world-v1',root.record()])
        def predict(branch,context):
            if not isinstance(branch,EventReader) or branch.learner.identity!=root.learner.identity:
                raise ValueError('same observed public process required')
            branch.learner._check(branch.memory)
            added=branch.observations-root.observations
            if not 0<=added<=steps:
                raise ValueError('future evidence must fit the branch observation memory')
            prefix=branch.memory.history[:-added] if added else branch.memory.history
            if prefix and prefix!=root.memory.history[-len(prefix):]:
                raise ValueError('future observation branch does not extend this factual root')
            future=branch.memory.history[-added:] if added else ()
            if persistent:
                return root.learner.persistent_continuation(root.memory,future,*root._inputs(context))
            history=root.memory.history+future
            if len(history)>512:raise ValueError('bounded world history required')
            memory=SegmentMemory(model.identity,history,len(history))
            return model.predict(memory,*root._inputs(context))
        return predict

    def __call__(self, context, cap):
        probabilities = self.predict(context)
        required = len(context['actions']) * len(probabilities)
        if required > cap or len(probabilities) > MAX_OUTCOMES:
            return None
        result = copy.deepcopy(context)
        for action in result['actions']:
            action['outcomes'] = [dict(copy.deepcopy(self.outcome_for(
                copy.deepcopy(context), action['id'], response)), p=probability)
                for response, probability in probabilities.items()]
        result['facts']['predictor_model'] = 'public event mixture; pre-reveal prediction scores; no calibration guarantee'
        return Reading(result, required, self.key, responses=probabilities, known_conditionals=True)

    def updated_event(self, context, event, ticket):
        inputs = self._inputs(context)
        if (not isinstance(event, dict) or set(event) != {'revealed_action'}
                or type(context['tick']) is not int or context['tick'] <= self.last_tick
                or not isinstance(ticket, str) or not ticket):
            raise ValueError('one actual response at a new owned observation tick required')
        memory = self.learner.observe(self.memory, *inputs, event['revealed_action'])
        new = copy.deepcopy(self)
        new.memory, new.last_tick = memory, context['tick']
        return new, dict(scored=True, **new.learner.diagnostics(memory))

    def updated_effect(self, context, observed, ticket):
        if digest(context['scope']) != self.owner:
            raise ValueError('predictor owner mismatch')
        return copy.deepcopy(self), dict(scored=False, reason='response not observed')

    def updated(self, context, observed, event, ticket):
        return self.updated_event(context, event, ticket)

    def record(self):
        return dict(version='event-reader-v1', owner=self.owner, key=self.key,
                    last_tick=self.last_tick, memory=self.learner.record(self.memory))

    def restored(self, record):
        if (not isinstance(record, dict)
                or set(record) != {'version', 'owner', 'key', 'last_tick', 'memory'}
                or record['version'] != 'event-reader-v1' or record['owner'] != self.owner
                or record['key'] != self.key or type(record['last_tick']) is not int):
            raise ValueError('complete matching event reader checkpoint required')
        memory = self.learner.restore(record['memory'])
        tick = record['last_tick']
        if (memory.observations == 0 and tick != -1
                or memory.observations > 0 and tick < memory.observations - 1):
            raise ValueError('observation clock inconsistent with predictive history')
        new = copy.deepcopy(self)
        new.memory, new.last_tick = memory, tick
        return new
