"""Offline execution alignment, not a confidence or causal-learning signal.

Adapters supply public-state fingerprints and subsequently revealed actions.
A differing action is a same-state disagreement only when public pre-states
match. Diverged states, missing actions and terminal padding stay separate.
No registry, mutable NPC memory, policy update or world access lives here.
"""
from .core import identifier


def _validate(trace, predicted=False):
    if not isinstance(trace, dict):
        raise ValueError('execution trace required')
    for key in ('run', 'group', 'target', 'unit'):
        identifier(trace.get(key))
    start = trace.get('start')
    horizon = trace.get('horizon')
    if type(start) is not int or start < 0 or type(horizon) is not int or not 1 <= horizon <= 16:
        raise ValueError('nonnegative start and bounded horizon required')
    steps = trace.get('steps')
    if not isinstance(steps, list) or len(steps) > horizon or predicted and not steps:
        raise ValueError('bounded ordered execution steps required')
    previous = None
    for offset, step in enumerate(steps):
        if step is None and not predicted:
            previous = None; continue
        if not isinstance(step, dict):
            raise ValueError('execution step required')
        if type(step.get('index')) is not int or step['index'] != start + offset:
            raise ValueError('explicit consecutive observation-unit indices required')
        identifier(step.get('before')); identifier(step.get('after'))
        if type(step.get('terminal')) is not bool:
            raise ValueError('explicit absorbing-terminal flag required')
        if previous is not None and (previous['terminal'] or previous['after'] != step['before']):
            raise ValueError('continuous transitions ending at terminal required')
        for side in ('self', 'opponent'):
            if side not in step:
                raise ValueError('explicit action observation or None required')
            actions = step[side]
            if actions is None and not predicted:
                continue  # Missing observations are not empty action sets.
            if not isinstance(actions, dict):
                raise ValueError('explicit actor/action map required')
            for actor, action in actions.items():
                identifier(actor); identifier(action)
        if step.get('self') is not None and step.get('opponent') is not None and set(step['self']) & set(step['opponent']):
            raise ValueError('self and opponent owners must be disjoint')
        if step['terminal'] and offset != len(steps)-1:
            raise ValueError('no observations after absorbing terminal')
        previous = step
    if predicted and len(steps) < horizon and not steps[-1]['terminal']:
        raise ValueError('model must cover its horizon or reach terminal')


def compare_execution(predicted, observed):
    """Compare one sampled forecast with one matching, revealed execution.

    Trace identity is run/group/target/unit/start/horizon. Steps describe actual
    transitions only; terminal purpose-value padding does not execute actions.
    Partial nonterminal observations are censored. Public-state equivalence is
    the adapter's contract (including relevant public history for hidden-state
    games); equal actions alone do not establish policy or causal equivalence.
    Even a fully aligned sample is not permission to train from its error.
    """
    _validate(predicted, predicted=True); _validate(observed)
    for key in ('run', 'group', 'target', 'unit', 'start', 'horizon'):
        if predicted[key] != observed[key]:
            raise ValueError('execution identity mismatch: ' + key)
    result = dict(compared_steps=0, state_matched_steps=0, state_diverged_steps=0,
                  self_compared=0, self_differences=0, opponent_compared=0,
                  opponent_differences=0, same_inputs_successor_differences=0,
                  missing_observation_steps=0, terminal_only_steps=0,
                  terminal_boundary_differences=0, first_self_difference=None,
                  first_opponent_difference=None, first_state_divergence=None,
                  steps=[])
    psteps, osteps = predicted['steps'], observed['steps']
    for offset in range(predicted['horizon']):
        p = psteps[offset] if offset < len(psteps) else None
        o = osteps[offset] if offset < len(osteps) else None
        row = dict(offset=offset, self='not-comparable', opponent='not-comparable')
        if p is None or o is None:
            pended = bool(psteps and psteps[-1]['terminal'] and p is None)
            oended = bool(osteps and osteps[-1] is not None and osteps[-1]['terminal'] and offset >= len(osteps))
            if o is None and not oended:
                row['state'] = 'observation-missing'; result['missing_observation_steps'] += 1
            elif pended and oended:
                row['state'] = 'terminal-only'; result['terminal_only_steps'] += 1
            else:
                row['state'] = 'terminal-boundary-difference'; result['terminal_boundary_differences'] += 1
        else:
            result['compared_steps'] += 1
            if p['before'] != o['before']:
                row['state'] = 'different'; result['state_diverged_steps'] += 1
                if result['first_state_divergence'] is None: result['first_state_divergence'] = offset
            else:
                row['state'] = 'same'; result['state_matched_steps'] += 1
                for side in ('self', 'opponent'):
                    if o[side] is None:
                        row[side] = 'unobserved'; continue
                    result[side + '_compared'] += 1
                    differs = p[side] != o[side]
                    row[side] = 'different' if differs else 'same'
                    result[side + '_differences'] += int(differs)
                    if differs and result['first_' + side + '_difference'] is None:
                        result['first_' + side + '_difference'] = offset
                if row['self'] == row['opponent'] == 'same' and (p['after'], p['terminal']) != (o['after'], o['terminal']):
                    result['same_inputs_successor_differences'] += 1
            row['successor_same'] = (p['after'], p['terminal']) == (o['after'], o['terminal'])
        result['steps'].append(row)
    return result
