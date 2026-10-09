"""Read-only decision evidence; never turns a loss or a forecast into a label."""
import copy
import math
from reflex.core import Policy, compile_batch, digest


def immediate(context):
    """Exact policy scores beside game-supplied consequences, without retuning."""
    saved = digest(context)
    b = compile_batch([context]); p = Policy(principle_priority='finite')
    d = p.decide(b, True); record = d.records(b)[0]
    choices = []
    for j, a in enumerate(context['actions']):
        choices.append(dict(action=a['id'], eligible=bool(d.eligible[0, j]),
            score=float(d.scores[0, j]), confidence=a['confidence'],
            outcomes=copy.deepcopy(a['outcomes'])))
    assert saved == digest(context)
    return dict(choice=record['action_id'], state=record['next_state'], choices=choices,
                context_hash=saved, personality=copy.deepcopy(context['personality']),
                values=copy.deepcopy(context['values']), needs=copy.deepcopy(context['needs']))


def continuation(branch, executed):
    """Actions alone do not certify matched model states or causal credit."""
    actions = branch['actions']
    if not actions or not isinstance(actions, list) or not all(isinstance(x, str) for x in actions):
        raise ValueError('nonempty modeled action sequence required')
    first = next((i for i, (a, b) in enumerate(zip(actions, executed)) if a != b), None)
    if first is not None: status = 'continuation_changed'
    elif len(executed) < len(actions): status = 'censored'
    else: status = 'actions_matched_states_unchecked'
    return dict(status=status, first_difference=None if first is None else first + 1,
        modeled_actions=list(actions), executed_actions=list(executed[:len(actions)]),
        modeled_goal=copy.deepcopy(branch['assessment']), terminal=branch['terminal'],
        causal_error_assigned=False)


def forecast_frame(trace, index):
    """Separate stated purpose, modeled continuation and observed one-turn flow."""
    t = trace[index]; d = t['deliberation']; chosen = d['selected_purpose']; best = d['best_purpose']
    if not all(type(x) in (int, float) and math.isfinite(x) and -1 <= x <= 1 for x in (chosen, best)):
        raise ValueError('bounded purpose evidence required')
    if chosen > best + 1e-9: raise ValueError('selected purpose exceeds reported best')
    purpose = t['purpose']; feedback = t['feedback']['purpose']
    measured = None
    if purpose is not None and feedback is not None:
        measured = dict(level_before=purpose['level'], level_after=feedback['level'],
            readiness_before=purpose['readiness'], readiness_after=feedback['readiness'],
            maintained=feedback['maintained'], completed=feedback['completed'])
    return dict(turn=index + 1, action=t['root'], before=copy.deepcopy(t['before']),
        after=copy.deepcopy(t['after']), flow=copy.deepcopy(t['flow']),
        actual_goal=copy.deepcopy(t['goal']), state=copy.deepcopy(t['state']),
        selected_model_purpose=chosen, best_model_purpose=best, model_purpose_gap=best-chosen,
        progress=measured, progress_audit=copy.deepcopy(t['progress']),
        activity=None if purpose is None else copy.deepcopy(purpose['activities'].get(t['root'])),
        branches=[continuation(b, [x['root'] for x in trace[index:]]) for b in t['endpoints']],
        verdict='pending_human_review', source_persona_hash=t['persona_hash'])


def unchanged_streak(frames):
    """Observable lack of movement, not an automatic stupidity/wait veto."""
    result = []; n = 0
    for f in frames:
        p = f.get('progress')
        if p is None: n = 0; result.append(None); continue
        same = abs(p['level_after']-p['level_before']) < 1e-9 and abs(p['readiness_after']-p['readiness_before']) < 1e-9
        n = n + 1 if same and not p['completed'] else 0
        result.append(n)
    return result


def render(data, template):
    import json
    if template.count('__REVIEW_DATA__') != 1: raise ValueError('one data slot required')
    text = json.dumps(data, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    return template.replace('__REVIEW_DATA__', text)
