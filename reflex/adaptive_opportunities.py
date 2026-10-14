"""One observed belief snapshot shared by long valuation and root execution.

Real observations update AdaptiveKernel outside this object. Rebuild after each
reveal. Imagined branches retain this snapshot: future learning is deliberately
omitted, not partly applied to the first step of an otherwise stationary solve.
The adapter must still provide sufficient public AND psychological state.
"""
import copy

from .adaptive_dynamics import AdaptiveKernel
from .core import digest


class AdaptiveOpportunity:
    def __init__(self, graph, kernel, state, owner, *, choose, chooser_id):
        if not isinstance(kernel, AdaptiveKernel):
            raise ValueError('explicit adaptive public kernel required')
        kernel._check(state)
        if not callable(choose):
            raise ValueError('explicit continuation chooser required')
        self.kernel = copy.copy(kernel)
        self.kernel.update = False
        self.kernel.identity = digest([kernel.state_identity, False])
        self.memory = copy.deepcopy(state.memory)
        self.memory_id = digest(kernel.learner.record(self.memory))
        self.owner_id = digest([owner.character, owner.policy.signature()])
        self.predictions = {}
        self.solution = graph.solve(self.law, law_id=self.memory_id,
                                    choose=choose, chooser_id=chooser_id, initial=state.public)
        self.identity = digest(['adaptive-opportunity-v1', self.kernel.identity,
                               self.memory_id, self.owner_id,
                               None if self.solution is None else self.solution.identity])

    def law(self, public, action, symbols):
        context = self.kernel.features(copy.deepcopy(public), action)
        models = self.kernel.models(copy.deepcopy(public), action)
        stamp = digest([context, symbols, models])
        if stamp not in self.predictions:
            self.predictions[stamp] = self.kernel.learner.predict(self.memory, context, symbols, models)
        return dict(self.predictions[stamp])

    def _check(self, state):
        self.kernel._check(state)
        if digest(self.kernel.learner.record(state.memory)) != self.memory_id:
            raise ValueError('prediction changed; rebuild terminal valuation before deciding')
        if self.solution is None:
            raise ValueError('incomplete graph valuation cannot supply a decision')
        self.solution.index(state.public)

    def __call__(self, state, action):
        self._check(state)
        return self.kernel(state, action)

    def evaluate(self, state, owner):
        self._check(state)
        if digest([owner.character, owner.policy.signature()]) != self.owner_id:
            raise ValueError('terminal valuation belongs to another personality or policy')
        return self.solution.evaluate(state.public)

    def action(self, state):
        self._check(state)
        return self.solution.action(state.public)
