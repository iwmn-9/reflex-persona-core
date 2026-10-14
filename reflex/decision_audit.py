"""Matched counterfactual policy evaluation; never feed reference truth to actor.

The decomposition is ordered and signed, not a unique psychological diagnosis.
All policies are evaluated under ONE declared reference law and objective.
"""
import math
import copy
from .core import digest


def policy_loss_audit(graph, initial, reference_law, *, reference_id, informed, repaired, legacy, informed_legacy=None):
    """Separate declared-persona, predictive-law and preference-proxy effects.

    informed: resolved preference meanings, reference law.
    repaired: same meanings/persona, actor's law.
    legacy: same persona/actor law, old proxies.
    Callers must supply these controls faithfully; names alone prove no causality.
    Full-trajectory evaluation includes future decisions, not just the root.
    Negative terms are retained: a flawed proxy can accidentally help purpose.
    """
    solutions = dict(informed=informed, repaired=repaired, legacy=legacy)
    if informed_legacy is not None:solutions['informed_legacy']=informed_legacy
    for solution in solutions.values():
        if solution.graph is not graph or solution.graph_identity != graph.identity:
            raise ValueError('same compiled public graph required for matched policies')
        solution.index(initial)
    # Materialize one reference snapshot. A sampled/stateful caller must not
    # accidentally supply different comparison worlds to successive policies.
    source_law=reference_law;probabilities={}
    def reference_law(state,key,symbols):
        stamp=digest([graph.record(copy.deepcopy(state)),key,symbols])
        if stamp not in probabilities:
            probabilities[stamp]=copy.deepcopy(source_law(copy.deepcopy(state),key,symbols))
        return copy.deepcopy(probabilities[stamp])
    optimal = graph.solve(reference_law, law_id=reference_id, initial=initial)
    if optimal is None:raise ValueError('complete reference evaluation required')
    evaluated = dict(optimal=optimal)
    for name, solution in solutions.items():
        fixed = graph.solve(reference_law, law_id=reference_id, initial=initial,
            choose=lambda states, alternatives, s=solution:tuple(s.action(w) for w in states),
            chooser_id=digest(['fixed-policy-audit',solution.identity]))
        if fixed is None:raise ValueError('complete fixed-policy evaluation required')
        evaluated[name] = fixed
    values = {k:math.fsum(p*v for p,v in s.evaluate(initial).outcomes) for k,s in evaluated.items()}
    parts = dict(persona=values['optimal']-values['informed'],
                 prediction=values['informed']-values['repaired'],
                 proxy=values['repaired']-values['legacy'])
    total = values['optimal']-values['legacy']
    if abs(math.fsum(parts.values())-total)>1e-10 or parts['persona'] < -1e-10:
        raise AssertionError('inconsistent matched reference evaluation')
    # Distinguish a current compromise from costs of future persona choices.
    # This exposes why a local corridor is not a whole-trajectory regret cap.
    node=graph.nodes[evaluated['informed'].index(initial)];root_values={}
    for key,events in node['branches'].items():
        p=reference_law(node['state'],key,tuple(s for s,_,_ in events))
        root_values[key]=math.fsum(p[s]*q*v for s,j,_ in events for q,v in evaluated['informed'].distributions[j])
    best=max(root_values.values(),default=values['informed'])
    persona_parts=dict(current=best-values['informed'],continuation=values['optimal']-best)
    interaction=None if informed_legacy is None else dict(
        proxy_with_reference_law=values['informed']-values['informed_legacy'],
        proxy_with_actor_law=parts['proxy'],
        difference=parts['proxy']-(values['informed']-values['informed_legacy']))
    return dict(values=values,loss=parts,total=total,reference=reference_id,reference_snapshot=digest(probabilities),graph=graph.identity,interaction=interaction,
                persona_parts=persona_parts,informed_root_values=root_values,
                policies={k:s.identity for k,s in solutions.items()},
                evaluation={k:s.identity for k,s in evaluated.items()},
                semantics='ordered signed effects under declared law; residual persona includes risk, principles, needs, style and local corridor')
