"""Research candidate: compare feasible continuations, without nested search.

The ordinary owned policy remains available after every root. A second generic
policy retries that root whenever it is publicly legal and progress permits it;
otherwise it uses the learning owner's policy. Optional two-step recovery reuses
the existing owned_rollout controller. These are proposals, never game
labels or rewards for preparation. Actual feedback is followed by fresh review.

Inspired by parallel rollout, but finite sampling, changing beliefs and the
nonlinear personality selector do NOT inherit its MDP improvement theorem.
Packaged for explicit integration checks. Adoption still requires completed
independent long-play evidence; packaging alone is not acceptance.
"""
from . import cloning as copy
import math
from collections import OrderedDict
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
                      cumulative_preference=None, include_retry=True, captured=None,
                      purpose_estimator='sampled', tail_recovery=0, recovery_cache=0,
                      coherent_cumulative_mean=False, capture_prefixes=False,
                      sampling='independent', continuation_scope='root',
                      tail_purpose_reflex=False, retry_until_gain=False):
    if type(horizon) is not int or not 1 <= horizon <= 128:
        raise ValueError('explicit bounded continuation horizon required')
    if (not isinstance(seeds, tuple) or not 1 <= len(seeds) <= 8
            or any(type(s) is not int or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds) or type(include_retry) is not bool):
        raise ValueError('distinct model seeds and explicit continuation bank required')
    if terminal(world) or actor.pending is not None:
        raise ValueError('resolved owner in a live public world required')
    if purpose_estimator not in ('sampled', 'conditional'):
        raise ValueError('known purpose mean estimator required')
    if type(coherent_cumulative_mean) is not bool or coherent_cumulative_mean and (purpose_estimator!='conditional' or cumulative_preference is None):
        raise ValueError('coherent mean requires conditional cumulative value forecasts')
    if type(tail_recovery) is not int or tail_recovery not in (0,2) or (tail_recovery and horizon<2):
        raise ValueError('optional existing two-step owner recovery requires a future tail')
    if type(recovery_cache) is not int or not 0<=recovery_cache<=256:
        raise ValueError('bounded within-decision recovery cache required')
    if type(tail_purpose_reflex) is not bool:
        raise ValueError('explicit future immediate-purpose selection required')
    if type(retry_until_gain) is not bool or retry_until_gain and not include_retry:
        raise ValueError('explicit optional retry termination requires the retry bank')
    if sampling not in ('independent', 'stratified'):
        raise ValueError('known continuation sampling mode required')
    if continuation_scope not in ('root', 'shared') or continuation_scope == 'shared' and not include_retry:
        raise ValueError('shared continuations require the retry bank')
    if sampling == 'stratified':
        from reflex.rollout_sampling import stratified_uniforms
        stratified_uniforms(seeds, 0)
    source = digest(actor.record())
    # Reuse the existing distinction between the modeled physical process and
    # the hypothetical person's changing beliefs. No true-world law is input.
    model = owned_rollout(actor, observe=observe, events=events, terminal=terminal,
        assess=assess, record=lambda w: w, request_for=request_for, feedback=feedback,
        model_id=model_id, target=target, tail_steps=horizon-1, seeds=seeds,
        world_model=world_model, cumulative_preference=cumulative_preference,
        tail_recovery=tail_recovery)

    def planner(contexts, decisions):
        if digest(actor.record()) != source or len(contexts) != 1:
            raise ValueError('forecast requires the unchanged factual owner')
        c = contexts[0]
        cache=OrderedDict();cache_hits=0;recovery_computations=0
        roots = sorted(a['id'] for a in c['actions'] if a['legal'] and not a['known_failure'])
        paths, audits, mapping, policies = {}, {}, {}, {}
        steps = 0; purpose_reflex_calls = 0; reused_steps = 0
        for root_index, root in enumerate(roots):
            first = model.owner_transitions(copy.deepcopy(world), root, actor, copy.deepcopy(c))
            modes = [('reflex', None, False)]
            if include_retry:
                modes += [('retry', key, False) for key in (roots if continuation_scope == 'shared' else (root,))]
                if retry_until_gain:
                    modes += [('retry', key, True) for key in (roots if continuation_scope == 'shared' else (root,))]
            for mode, preferred, until_gain in modes:
                suffix = f'-{roots.index(preferred):03d}' if mode == 'retry' and continuation_scope == 'shared' else ''
                if until_gain:
                    suffix += '-until-gain'
                proposal = f'candidate-{root_index:03d}-{mode}{suffix}'
                mapping[proposal] = root
                policies[proposal] = dict(kind=mode, preferred=preferred)
                if until_gain:
                    policies[proposal]['termination'] = 'public-purpose-above-origin'
                branches, rows = [], []
                for response in first:
                    for seed in seeds:
                        if until_gain:
                            origin = Goal(**assess(world))
                            first_goal = Goal(**assess(response.state))
                            if first_goal.target != origin.target:
                                raise ValueError('one declared purpose throughout retry termination required')
                            if first_goal.value > origin.value + 1e-12:
                                # Termination after the forced root makes this
                                # exact path the ordinary owned continuation.
                                # Reuse its completed learner and full effects,
                                # not an approximation or a shorter trajectory.
                                ordinary = f'candidate-{root_index:03d}-reflex'
                                index = len(rows)
                                branches.append(copy.deepcopy(paths[ordinary][index]))
                                rows.append(copy.deepcopy(audits[ordinary][index]))
                                rows[-1]['retry_termination_step'] = 1
                                count = len(rows[-1]['actions'])
                                steps += count; reused_steps += count
                                continue
                        own, w = copy.deepcopy(actor), copy.deepcopy(world)
                        flows, actions, falls, corrections, prefixes = [], [], [], [], []
                        current_request = copy.deepcopy(request)
                        recovery_calls = 0
                        retry_finished = False
                        retry_termination_step = None
                        origin = Goal(**assess(world)) if until_gain else None
                        for t in range(horizon):
                            if terminal(w):
                                break
                            if t:
                                current_request = request_for(observe(copy.deepcopy(w), copy.deepcopy(own.state)))
                            stage = own._prepare(current_request)
                            context = stage['effective']
                            legal = {a['id'] for a in context['actions'] if a['legal'] and not a['known_failure']}
                            key = root if t == 0 else None
                            retry_active = mode == 'retry' and not retry_finished
                            if t and retry_active and preferred in legal and preferred in stage['progress_allowed']:
                                key = preferred
                            elif t and retry_active:
                                falls.append(t)
                            model_owner = copy.deepcopy(own)
                            future = None
                            if key is None and tail_recovery and recovery_calls < model.frontier.future_planner_limit:
                                stamp=digest([w,own.record()]) if recovery_cache else None
                                if recovery_cache and stamp in cache:
                                    entry=cache[stamp];cache.move_to_end(stamp);cache_hits+=1
                                    if entry['present']:
                                        future=lambda cs,ds,entry=entry:copy.deepcopy(entry['forecast'])
                                else:
                                    recovery_computations+=1
                                    future = model.frontier.future_planner(copy.deepcopy(own),
                                        copy.deepcopy(current_request),copy.deepcopy(w))
                                    if recovery_cache:
                                        def put(value,stamp=stamp):
                                            cache[stamp]=value
                                            while len(cache)>recovery_cache:cache.popitem(last=False)
                                        if future is None:put(dict(present=False))
                                        else:
                                            def memo(cs,ds,original=future,put=put):
                                                f=original(cs,ds)
                                                put(dict(present=True,forecast=copy.deepcopy(f)))
                                                return f
                                            future=memo
                                if future is not None:recovery_calls += 1
                            if key is None and future is None and tail_purpose_reflex:
                                # The factual owner has a declared public goal.
                                # Reuse that same existing one-step evaluator
                                # for its hypothetical fallback, including when
                                # passive income makes coarse effects positive.
                                # Personality, outcome risks, learning and retry
                                # proposals are not replaced by a new reward.
                                person = copy.deepcopy(own)
                                immediate = owned_rollout(person, observe=observe, events=events,
                                    terminal=terminal, assess=assess, record=lambda w: w,
                                    request_for=request_for, feedback=feedback, model_id=model_id,
                                    target=target, tail_steps=0, tree_steps=1,
                                    world_model=world_model, cumulative_preference=cumulative_preference)
                                proposed = immediate.planner(person, copy.deepcopy(current_request), copy.deepcopy(w),
                                    _forecast_only=True)
                                def complete_purpose(cs, ds, proposed=proposed):
                                    f = proposed(cs, ds)
                                    if f is None:
                                        raise ValueError('incomplete future immediate-purpose forecast')
                                    return f
                                future = complete_purpose
                                purpose_reflex_calls += 1
                            result = DecisionLoop.decide_batch([(own, current_request)], False, future,
                                _counterfactual_roots=(key,) if key is not None else None)[0]
                            key = result['decision']['action_id']
                            if t == 0:
                                after, flow = copy.deepcopy(response.state), copy.deepcopy(response.flow)
                            else:
                                outcomes = model.owner_transitions(copy.deepcopy(w), key, model_owner, copy.deepcopy(result['context']))
                                u = (stratified_uniforms(seeds, context['tick'])[seeds.index(seed)]
                                     if sampling == 'stratified' else
                                     np.random.default_rng(np.random.SeedSequence([seed, context['tick']])).random())
                                mass = 0.
                                selected = outcomes[-1]
                                for outcome in outcomes:
                                    mass += outcome.probability
                                    if u < mass:
                                        selected = outcome
                                        break
                                after, flow = copy.deepcopy(selected.state), copy.deepcopy(selected.flow)
                                if purpose_estimator == 'conditional':
                                    # Same model and sampled owner learning. This
                                    # removes sampled one-step purpose noise;
                                    # it cannot correct a wrong world model.
                                    corrections.append(math.fsum(
                                        o.probability * Goal(**assess(o.state)).value
                                        for o in outcomes) - Goal(**assess(after)).value)
                            resolve_feedback(own, result['ticket'], feedback(w, after, key, result['context'], flow))
                            flows.append(flow); actions.append(key); w = after; steps += 1
                            if until_gain and not retry_finished:
                                observed_goal = Goal(**assess(w))
                                if observed_goal.target != origin.target:
                                    raise ValueError('one declared purpose throughout retry termination required')
                                retry_finished = observed_goal.value > origin.value + 1e-12
                                if retry_finished:
                                    retry_termination_step = t + 1
                            if capture_prefixes:
                                prefixes.append(dict(assessment=Goal(**assess(w)).record(),terminal=terminal(w),
                                    correction=math.fsum(corrections),world=copy.deepcopy(w)))
                        goal = Goal(**assess(w))
                        probability = response.probability / len(seeds)
                        flows.extend(effect() for _ in range(horizon-len(flows)))
                        branches.append(Branch(probability, tuple(flows), (1.,)*horizon))
                        rows.append(dict(actions=actions, assessment=goal.record(), terminal=terminal(w),
                            fallback_steps=falls, recovery_calls=recovery_calls, final_owner=digest(own.record())))
                        if until_gain:
                            rows[-1]['retry_termination_step'] = retry_termination_step
                        if purpose_estimator == 'conditional':
                            rows[-1]['purpose_mean_estimate'] = goal.value + math.fsum(corrections)
                        if capture_prefixes:rows[-1]['prefixes']=prefixes
                paths[proposal], audits[proposal] = tuple(branches), rows
        forecast = goal_forecast(c, paths, audits, horizon=horizon,
            unit='owner-decisions', target=target, policy=actor.policy,
            plan_roots=mapping, settlement_weight='absolute',
            objective_origin=Goal(**assess(world)), opportunity_regret=True,
            cumulative_preference=cumulative_preference, outcome_limit=256,
            coherent_cumulative_mean=coherent_cumulative_mean)
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
            purpose_estimator=purpose_estimator,
            coherent_cumulative_mean=coherent_cumulative_mean,
            tail_recovery=tail_recovery,
            recovery_cache=dict(capacity=recovery_cache,hits=cache_hits,computations=recovery_computations),
            source_semantics='learning owner (optional existing recovery) or retry a public action with that owner as fallback; only selected root is executed')
        # Keep the default forecast bytes compatible with the stable version.
        if sampling != 'independent':
            metadata['continuation_portfolio']['sampling'] = sampling
        if continuation_scope != 'root':
            metadata['continuation_portfolio'].update(continuation_scope=continuation_scope,
                source_semantics='every real first action shares the same public retry bank and learning-owner fallback; only the selected first action executes')
        if tail_purpose_reflex:
            metadata['continuation_portfolio']['tail_purpose_reflex'] = dict(
                kind='existing-one-step-owned-purpose', calls=purpose_reflex_calls,
                semantics='Existing H2 recovery first; otherwise evaluate the public immediate goal with the same owner. '
                          'Forced retry proposals remain proposals. No factual or hidden outcomes are supplied.')
        if retry_until_gain:
            metadata['continuation_portfolio']['reused_steps'] = reused_steps
            metadata['continuation_portfolio']['retry_until_gain'] = dict(
                kind='public-purpose-above-origin',
                semantics='Additional proposal ends root retry after the modeled public goal exceeds its root value. '
                          'Same learning owner then decides; original full retry and reflex remain. '
                          'Only the actual selected root executes.')
        forecast = replace(forecast, audit=metadata, max_regret=fraction*spread)
        if digest(actor.record()) != source:
            raise ValueError('counterfactual learning contaminated factual owner')
        if captured is not None:
            captured.update(forecast=forecast, modeled=audits, simulated_steps=steps, paths=paths)
        return forecast
    return planner
