"""Research candidate: compare feasible continuations, without nested search.

The ordinary owned reflex remains available after every root. A second generic
policy retries that root whenever it is publicly legal and progress permits it;
otherwise it uses the learning owner's reflex. These are proposals, never game
labels or rewards for preparation. Actual feedback is followed by fresh review.

Inspired by parallel rollout, but finite sampling, changing beliefs and the
nonlinear personality selector do NOT inherit its MDP improvement theorem.
Kept outside reflex until independent long-play evidence supports integration.
"""
import copy
from dataclasses import replace

import numpy as np

from reflex.closed_loop import owned_rollout
from reflex.core import digest
from reflex.decision_loop import DecisionLoop
from reflex.examples import effect
from reflex.intertemporal import Branch
from reflex.model_observer import resolve_feedback
from reflex.purpose_plan import Goal, goal_forecast


def portfolio_planner(actor, request, world, *, observe, events, terminal,
                      assess, request_for, feedback, model_id, target,
                      horizon=16, seeds=(701, 702, 703), world_model='posterior',
                      cumulative_preference=None, include_retry=True, captured=None):
    if type(horizon) is not int or not 1 <= horizon <= 128:
        raise ValueError('explicit bounded continuation horizon required')
    if (not isinstance(seeds, tuple) or not 1 <= len(seeds) <= 8
            or any(type(s) is not int or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds) or type(include_retry) is not bool):
        raise ValueError('distinct model seeds and explicit continuation bank required')
    if terminal(world) or actor.pending is not None:
        raise ValueError('resolved owner in a live public world required')
    source = digest(actor.record())
    # Reuse the existing distinction between the modeled physical process and
    # the hypothetical person's changing beliefs. No true-world law is input.
    model = owned_rollout(actor, observe=observe, events=events, terminal=terminal,
        assess=assess, record=lambda w: w, request_for=request_for, feedback=feedback,
        model_id=model_id, target=target, tail_steps=horizon-1, seeds=seeds,
        world_model=world_model, cumulative_preference=cumulative_preference)

    def planner(contexts, decisions):
        if digest(actor.record()) != source or len(contexts) != 1:
            raise ValueError('forecast requires the unchanged factual owner')
        c = contexts[0]
        roots = sorted(a['id'] for a in c['actions'] if a['legal'] and not a['known_failure'])
        paths, audits, mapping, policies = {}, {}, {}, {}
        steps = 0
        for root_index, root in enumerate(roots):
            first = model.owner_transitions(copy.deepcopy(world), root, actor, copy.deepcopy(c))
            modes = ('reflex', 'retry') if include_retry else ('reflex',)
            for mode in modes:
                proposal = f'candidate-{root_index:03d}-{mode}'
                mapping[proposal] = root
                policies[proposal] = dict(kind=mode, preferred=root if mode == 'retry' else None)
                branches, rows = [], []
                for response in first:
                    for seed in seeds:
                        own, w = copy.deepcopy(actor), copy.deepcopy(world)
                        flows, actions, falls = [], [], []
                        current_request = copy.deepcopy(request)
                        for t in range(horizon):
                            if terminal(w):
                                break
                            if t:
                                current_request = request_for(observe(copy.deepcopy(w), copy.deepcopy(own.state)))
                            stage = own._prepare(current_request)
                            context = stage['effective']
                            legal = {a['id'] for a in context['actions'] if a['legal'] and not a['known_failure']}
                            key = root if t == 0 else None
                            if t and mode == 'retry' and root in legal and root in stage['progress_allowed']:
                                key = root
                            elif t and mode == 'retry':
                                falls.append(t)
                            model_owner = copy.deepcopy(own)
                            result = DecisionLoop.decide_batch([(own, current_request)], False,
                                _counterfactual_roots=(key,) if key is not None else None)[0]
                            key = result['decision']['action_id']
                            if t == 0:
                                after, flow = copy.deepcopy(response.state), copy.deepcopy(response.flow)
                            else:
                                outcomes = model.owner_transitions(copy.deepcopy(w), key, model_owner, copy.deepcopy(result['context']))
                                u = np.random.default_rng(np.random.SeedSequence([seed, context['tick']])).random()
                                mass = 0.
                                selected = outcomes[-1]
                                for outcome in outcomes:
                                    mass += outcome.probability
                                    if u < mass:
                                        selected = outcome
                                        break
                                after, flow = copy.deepcopy(selected.state), copy.deepcopy(selected.flow)
                            resolve_feedback(own, result['ticket'], feedback(w, after, key, result['context'], flow))
                            flows.append(flow); actions.append(key); w = after; steps += 1
                        goal = Goal(**assess(w))
                        probability = response.probability / len(seeds)
                        flows.extend(effect() for _ in range(horizon-len(flows)))
                        branches.append(Branch(probability, tuple(flows), (1.,)*horizon))
                        rows.append(dict(actions=actions, assessment=goal.record(), terminal=terminal(w),
                            fallback_steps=falls, final_owner=digest(own.record())))
                paths[proposal], audits[proposal] = tuple(branches), rows
        forecast = goal_forecast(c, paths, audits, horizon=horizon,
            unit='owner-decisions', target=target, policy=actor.policy,
            plan_roots=mapping, settlement_weight='absolute',
            objective_origin=Goal(**assess(world)), opportunity_regret=True,
            cumulative_preference=cumulative_preference, outcome_limit=256)
        metadata = copy.deepcopy(forecast.audit)
        # Opportunity means a choice between real first actions. Adding an
        # intentionally bad continuation under an already-covered root must
        # never lower the competence floor. The original one-plan-per-root
        # spread has this property; the raw expanded proposal spread does not.
        best_by_root = {}
        for name, roots_for_plan in forecast.roots.items():
            root = roots_for_plan[0]
            best_by_root[root] = max(best_by_root.get(root, -1.), forecast.purpose[name])
        spread = max(best_by_root.values()) - min(best_by_root.values())
        fraction = metadata['purpose_band']['fraction']
        metadata['purpose_band'].update(spread=spread, basis='best current-tier continuation per real root',
            best_by_root=best_by_root, raw_proposal_regret=forecast.max_regret,
            semantics='additional poor continuations cannot relax the real-action competence floor')
        metadata['continuation_portfolio'] = dict(policies=policies, simulated_steps=steps,
            source_owner=source, horizon=horizon, seeds=list(seeds), model_id=model_id,
            source_semantics='learning reflex or retry a public action with reflex fallback; only selected root is executed')
        forecast = replace(forecast, audit=metadata, max_regret=fraction*spread)
        if digest(actor.record()) != source:
            raise ValueError('counterfactual learning contaminated factual owner')
        if captured is not None:
            captured.update(forecast=forecast, modeled=audits, simulated_steps=steps)
        return forecast
    return planner
