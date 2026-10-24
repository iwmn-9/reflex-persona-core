"""Configuration-bound persistence of an existing RecedingOwner.

No policy is selected here. Host callbacks and predictor implementations remain
trusted code supplied by the application; model_id must version their semantics.
The digest detects corruption, not malicious rewriting of an untrusted save.
"""
import copy
import json
from dataclasses import asdict
from .core import compile_batch, digest
from .decision_loop import DecisionLoop


VERSION = 'receding-owner-checkpoint-v1'
CALLBACKS = frozenset(('observe', 'events', 'terminal', 'assess', 'record',
                       'request_for', 'feedback'))
DEFAULTS = dict(tail_recovery_mode='stalled', tail_recovery_horizon=2,
                sampling_standard_errors=None, cumulative_preference=None,
                tail_goal_refinement=False)


def plain(value):
    # A JSON round trip removes aliases and gives tuples one portable encoding.
    return json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False))


def configuration(loop, actor):
    options = dict(DEFAULTS, **loop.options)
    for name in CALLBACKS:
        options.pop(name)
    preference = options['cumulative_preference']
    if preference is not None:
        from .cumulative_preference import CumulativePreference
        if type(preference) is not CumulativePreference:
            raise ValueError('known cumulative preference checkpoint required')
        options['cumulative_preference'] = asdict(preference)
    options['sampling_check'] = (None if loop.sampling_check is None
                                 else asdict(loop.sampling_check))
    extension = loop._checkpoint_extension()
    if extension is not None:
        options['controller'] = extension
    dynamic = {'state', 'last_tick', 'strategy', 'predictor', 'memory', 'evidence',
               'pressure', 'progress', 'social'}
    settings = {k: v for k, v in actor.items() if k not in dynamic}
    settings['memory'] = {k: v for k, v in actor['memory'].items()
                          if k not in ('entries', 'last_tick')}
    settings['evidence'] = {k: v for k, v in actor['evidence'].items() if k != 'entries'}
    for name in ('pressure', 'progress'):
        settings[name] = None if name not in actor else actor[name]['config']
    social = actor.get('social')
    settings['social'] = None if social is None else dict(
        kind=social['kind'], settings={k: v for k, v in social['state'].items()
                                     if k not in ('entries', 'events', 'last_tick')})
    predictor = loop.actor.predictor
    # A predictor's restored() validates its own model identity and parameters.
    settings['predictor_type'] = (None if predictor is None else
                                  f'{type(predictor).__module__}.{type(predictor).__qualname__}')
    return plain(dict(planning=options, actor=settings))


def boundary(loop, public, actor):
    context = loop.observe(copy.deepcopy(public), copy.deepcopy(actor.state))
    compile_batch([context], allow_no_actions=loop.options['terminal'](copy.deepcopy(public)))
    if (any(context[k] != getattr(actor, k) for k in ('scope', 'personality', 'values', 'seed'))
            or context['tick'] <= actor.last_tick):
        raise ValueError('matching public world after the resolved owner tick required')
    return plain(dict(world=loop.options['record'](copy.deepcopy(public)), context=context))


def checkpoint(loop, public):
    actor = loop.actor.record()  # Reject pending actual feedback before any work.
    payload = plain(dict(version=VERSION, configuration=configuration(loop, actor),
                         boundary=boundary(loop, public, loop.actor), actor=actor))
    return dict(payload=payload, sha256=digest(payload))


def restored(template, public, saved):
    # Restoration is staged; failures never replace a live owner or world.
    expected = configuration(template, template.actor.record())
    try:
        if not isinstance(saved, dict) or set(saved) != {'payload', 'sha256'}:
            raise ValueError('complete receding owner checkpoint required')
        payload = plain(saved['payload'])
        if (not isinstance(payload, dict)
                or set(payload) != {'version', 'configuration', 'boundary', 'actor'}
                or payload['version'] != VERSION or digest(payload) != saved['sha256']):
            raise ValueError('checkpoint version, fields or checksum mismatch')
        if digest(expected) != digest(payload['configuration']):
            raise ValueError('checkpoint owner or planning configuration changed')
        context = template.observe(copy.deepcopy(public), copy.deepcopy(payload['actor']['state']))
        actor = DecisionLoop.from_record(context, payload['actor'], policy=template.actor.policy,
            read_control=template.actor.read_control, predictor=template.actor.predictor,
            decision_metric=template.actor.decision_metric, response_adoption=template.actor.response_adoption,
            allow_inactive=template.options['terminal'](copy.deepcopy(public)))
        actual = actor.record()
        if (digest(configuration(template, actual)) != digest(expected)
                or digest(actual) != digest(payload['actor'])
                or digest(boundary(template, public, actor)) != digest(payload['boundary'])):
            raise ValueError('checkpoint owner, schema or public boundary mismatch')
        return template._with_actor(actor)
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError('invalid receding owner checkpoint') from error
