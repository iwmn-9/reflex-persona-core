"""Public-rule graph adapter; stationary belief INSIDE each solve, live outside.

The four OpenGame diagnostics have constant need deficits and no social/effect
learner. Their future mode is resolved once by the existing Policy and remains
valid. Intent/age have no scoring effect (zero switch fee, constant familiarity).
This sufficiency is specific to these adapters, not assumed by the graph core.
"""
import copy
from reflex.adaptive_dynamics import AdaptiveState
from reflex.adaptive_opportunities import AdaptiveOpportunity
from reflex.core import Policy, character_key, digest, compile_batch
from reflex.opportunity_graph import OpportunityGraph, persona_choices
from reflex.integrated_trials import request_for
from reflex.examples import action, effect


def canonical(game,w):
    w=copy.deepcopy(w);w['signal']=None
    if game=='auction':
        w['episode']=0;w['position']['last']=[];w['position']['last_prize']=0
    else:w['episode']%=2
    return w


class OpenOpportunities:
    def __init__(self,g,episode=0,graph=None):
        self.g=g;self.placeholder=g.kernel.learner.start()
        if graph is not None:
            if graph.model_id!=g.model_id+':public-sufficient-v1':raise ValueError('same rules required')
            self.graph=graph;return
        self.graph=OpportunityGraph(actions=lambda w:g.keys(self.state(w)),events=g.events,
            terminal=lambda w:g.terminal(self.state(w)),assess=lambda w:g.goal(self.state(w)).record(),
            record=lambda w:canonical(g.game,w),model_id=g.model_id+':public-sufficient-v1')
        if not self.graph.compile(canonical(g.game,g.start(episode).public)):
            raise RuntimeError(self.graph.last)

    def state(self,w,memory=None):
        return AdaptiveState(w,self.placeholder if memory is None else memory,self.g.kernel.state_identity,w['tick'])

    def solve(self,s,owner,continuation='persona',*,preference_basis=None,bound=False):
        if continuation not in ('purpose','persona'):raise ValueError('declared continuation required')
        g=self.g;memory=s.memory;predictions={}
        before=digest([g.kernel.record(s),owner.record()])
        root=g.observe(s,owner.state)
        # Resolve the current mode exactly as the real owner; choices do not
        # affect mode selection in these fixed-needs, zero-switch-cost games.
        effective=owner._prepare(request_for(root,'open-model'))['effective']
        b=compile_batch([effective]);regime=owner.policy.decide(b,False).records(b)[0]['next_state']
        regime.update(intent_action=None,age=0)
        if owner.pressure is not None or owner.progress is not None or owner.social is not None or owner.predictor is not None or owner.memory.record()['entries']:
            raise ValueError('this diagnostic projection does not cover additional owner memory')
        def law(w,key,symbols):
            context=g.features(w,key);models=g.models(w,key);stamp=digest([context,symbols,models])
            if stamp not in predictions:predictions[stamp]=g.kernel.learner.predict(memory,context,symbols,models)
            return predictions[stamp]
        def choose(states,alternatives):
            cs=[]
            for w,alts in zip(states,alternatives):
                # Exact OpenGame context schema: fixed needs/values and root
                # fees. Outcomes are replaced by persona_choices; computing
                # the same EventLearner forecast again would add no evidence.
                c=copy.deepcopy(effective);c['tick']=w['tick'];c['state']=copy.deepcopy(regime)
                c['actions']=[action(a.key,effect()) for a in alts];cs.append(c)
            return persona_choices(cs,alternatives,owner.policy,preference_basis=preference_basis,target=g.target)
        if bound:
            if continuation!='persona':raise ValueError('bound execution requires the same persona continuation')
            result=AdaptiveOpportunity(self.graph,g.kernel,s,owner,choose=choose,
                chooser_id=digest([continuation,owner.policy.signature(),root['scope'],regime,
                                   None if preference_basis is None else preference_basis.identity]))
            if digest([g.kernel.record(s),owner.record()])!=before:raise AssertionError('solve mutated factual owner')
            return result
        result=self.graph.solve(law,law_id=digest(g.kernel.learner.record(memory)),
            choose=choose if continuation=='persona' else None,
            chooser_id=digest([continuation,owner.policy.signature(),root['scope'],regime]),initial=s.public)
        if digest([g.kernel.record(s),owner.record()])!=before:raise AssertionError('opportunity solve mutated factual owner')
        return OpenSolution(g,result,owner)


class OpenSolution:
    def __init__(self,g,solution,owner):
        self.g=g;self.solution=solution
        self.identity=solution.identity if solution is not None else 'incomplete'
        self.owner_reference=digest([owner.character,owner.policy.signature()])

    def evaluate(self,state,owner):
        if state.identity!=self.g.kernel.state_identity or digest([owner.character,owner.policy.signature()])!=self.owner_reference:
            raise ValueError('endpoint belongs to a different model or persona')
        if self.solution is None:return None
        return self.solution.evaluate(state.public)

    def action(self,state):return self.solution.action(state.public)
