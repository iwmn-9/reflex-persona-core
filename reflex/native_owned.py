"""Explicit boundary for the Rust factual owner and the frozen Python H32 generator.

The native owner chooses, learns, and persists. This module generates modeled
continuations from a detached copy of its factual state; it owns no live actor,
world RNG, or actual feedback. It does NOT make the generator native Rust.
Only the accepted public-conditional/default-progress composition is exported.
"""
import copy
from dataclasses import asdict

from .core import FEATURES, NEEDS, TRAITS, VALUES, digest, character_key
from .decision_loop import DecisionLoop
from .fixed_horizon import FixedHorizonOwner
from .receding_checkpoint import configuration
from .pressure import PressureConfig
from .progress import ProgressConfig
from .judgment import ReadControl
from .event_reader import EventReader
from .segmented_prediction import SegmentedEventLearner


def observation(context):
    actions = copy.deepcopy(context['actions'])
    for action in actions:
        for row in action['outcomes']:
            row['needs'] = [row['needs'].get(k, 0.) for k in NEEDS]
            row['values'] = [row['values'].get(k, 0.) for k in VALUES]
            row['style'] = [row['style'].get(k, 0.) for k in TRAITS]
    return dict(tick=context['tick'], needs=[copy.deepcopy(context['needs'][k]) for k in NEEDS],
        facts=copy.deepcopy(context['facts']), actions=actions, opponent=copy.deepcopy(context['opponent']))


def exported_state(actor):
    r = actor.record(); pred = r['predictor']; history = pred['memory']['history']
    # Reversible progress key recovery uses only current failure records. The
    # importer keeps their opaque reference hashes when a concrete key is unknown.
    failures = [dict(identity=k, count=n) for k,n in r['progress']['failures']]
    return dict(intent=r['state'] or dict(primary_need=None,mode=None,intent_action=None,age=0,mode_urgency=0.),
        last_tick=None if r['last_tick'] < 0 else r['last_tick'],
        experience=dict(entries=[dict(binding=dict(method=e['method'],situation=e['situation'],
            estimated=[FEATURES.index(f) for f in e['estimated']]),count=e['count'],samples=e['samples']) for e in r['memory']['entries']]),
        evidence=dict(entries=copy.deepcopy(r['evidence']['entries'])),
        prediction=dict(observations=pred['memory']['observations'],history=[dict(inputs={k:e[k] for k in ('context','legal','models')},outcome=e['outcome']) for e in history]),
        predictor_tick=None if pred['last_tick'] < 0 else pred['last_tick'],
        pressure=dict(debt=[r['pressure']['debt'][k] for k in NEEDS],stalls=[r['pressure']['stalls'][k] for k in NEEDS]),
        pressure_tick=None if r['pressure']['last_tick'] < 0 else r['pressure']['last_tick'],
        progress=dict(**{k:r['progress'][k] for k in ('peak','ready_peak','stalls','wait_spent')},failures=failures))


class NativeOwnedBoundary:
    """Host-created, immutable configuration; only detached computations follow.

    The host versions rule/callback semantics through FixedHorizonOwner.model_id.
    Supply a fresh template and its initial public world. Every call supplies the
    CURRENT native learned state and public world explicitly. Unsupported actor
    paths fail instead of silently dropping social/exact/experimental behavior.
    """
    def __init__(self, template, initial_public):
        if type(template) is not FixedHorizonOwner or template.actor.last_tick != -1:
            raise ValueError('fresh fixed-H32 template required')
        a = template.actor; o = template.options
        if (o['tail_steps'] != 31 or o['seeds'] != (701,702,703) or template.recovery_cache != 64
            or a.social is not None or a.progress is None or a.pressure is None or a.progress.config != ProgressConfig()
            or a.pressure.config != PressureConfig() or a.read_control != ReadControl(max_nodes=64)
            or type(a.predictor) is not EventReader or type(a.predictor.learner) is not SegmentedEventLearner
            or a.predictor.key != 'event-response' or a.memory.capacity != 64
            or a.decision_metric != 'purpose' or a.response_adoption != 'joint'
            or a.evidence.capacity != 64 or a.evidence.window != 12 or a.evidence.min_trials != 4 or a.evidence.margin != .02):
            raise ValueError('supported frozen public-conditional owner configuration required')
        c = template.observe(copy.deepcopy(initial_public), None)
        priority = dict(kind=a.policy.principle_priority)
        if priority['kind'] == 'personal': priority['rigidity'] = a.policy.commitments[character_key(c)]
        self._template = copy.deepcopy(template)
        learner = a.predictor.learner
        self._configuration = dict(actor=dict(scope=c['scope'],seed=c['seed'],personality=[c['personality'][k] for k in TRAITS],
            values=[c['values'][k] for k in VALUES],objective=c['objective'],policy=dict(priority=priority,residual=a.policy.residual.tolist())),
            predictor=dict(model_id=o['model_id'],factors=list(learner.factors),names=list(learner.names),window=learner.window),
            contract=digest(configuration(template,a.record())),purpose_target=o['target'])

    @property
    def configuration(self): return copy.deepcopy(self._configuration)

    def actor(self, public, state):
        """Reconstruct the forecast's private learner; never replaces the template."""
        source = copy.deepcopy(state); t = self._template
        expected = set(exported_state(t.actor))
        if not isinstance(source,dict) or set(source) != expected:
            raise ValueError('complete native factual owner state required')
        r = t.actor.record(); tick = -1 if source['last_tick'] is None else source['last_tick']
        r.update(state=None if tick < 0 else source['intent'],last_tick=tick)
        r['memory'].update(last_tick=tick,entries=[dict(method=e['binding']['method'],situation=e['binding']['situation'],
            estimated=[FEATURES[i] for i in e['binding']['estimated']],count=e['count'],samples=e['samples']) for e in source['experience']['entries']])
        r['evidence']['entries'] = source['evidence']['entries']
        r['predictor']['last_tick'] = -1 if source['predictor_tick'] is None else source['predictor_tick']
        r['predictor']['memory'].update(observations=source['prediction']['observations'],
            history=[dict(**e['inputs'],outcome=e['outcome']) for e in source['prediction']['history']])
        r['pressure'].update(debt=dict(zip(NEEDS,source['pressure']['debt'],strict=True)),
            stalls=dict(zip(NEEDS,source['pressure']['stalls'],strict=True)),
            last_tick=-1 if source['pressure_tick'] is None else source['pressure_tick'])
        progress=source['progress']
        r['progress'].update(**{k:progress[k] for k in ('peak','ready_peak','stalls','wait_spent')},last_tick=tick,
            failures=[[e['identity'],e['count']] for e in progress['failures']])
        c=t.observe(copy.deepcopy(public),copy.deepcopy(r['state']))
        return DecisionLoop.from_record(c,r,policy=t.actor.policy,read_control=t.actor.read_control,predictor=t.actor.predictor,
            decision_metric='purpose',response_adoption='joint',allow_inactive=t.options['terminal'](copy.deepcopy(public)))

    def input(self, public, state):
        actor=self.actor(public,state); c=self._template.observe(copy.deepcopy(public),copy.deepcopy(actor.state))
        req=self._template.request_for(copy.deepcopy(c)); p=actor.predictor
        if (req.reader is not None or req.value_proposal is not None or req.social_bindings is not None
            or req.invalidate or any(req.exact.values()) or req.outcome_target != 'immediate'):
            raise ValueError('unsupported override/exact/invalidation path at native boundary')
        factors,legal,models=p._inputs(c); conditional={}
        for action in c['actions']:
            rows=[dict(p.outcome_for(copy.deepcopy(c),action['id'],z),p=1/len(legal)) for z in legal]
            temp=copy.deepcopy(c); temp['actions']=[dict(action,outcomes=rows)]
            conditional[action['id']]=observation(temp)['actions'][0]['outcomes']
        return dict(observation=observation(c),bindings={k:dict(method=b.method,situation=b.situation,
            estimated=[FEATURES.index(f) for f in b.estimated]) for k,b in req.bindings.items()},
            responses=dict(context=list(factors),legal=list(legal),models=[[models[k][z] for z in legal] for k in p.learner.names]),
            conditional=conditional,purpose=asdict(req.purpose),threatened=req.threatened,maintains_advantage=req.maintains_advantage)

    def forecast(self, public, state):
        actor=self.actor(public,state); owner=self._template._with_actor(actor)
        owner.decide(copy.deepcopy(public)); f=owner.last['forecast']; means=f.return_means
        if any(getattr(f,k) is not None for k in ('continuity','stability_reference','replan_check','purpose_sampling','goal_ties')):
            raise ValueError('experimental selection extension has no native contract')
        return dict(owner=self.configuration['actor'],state=copy.deepcopy(state['intent']),tick=f.contexts[0]['tick'],
            target=f.target,horizon=f.horizon,max_regret=f.max_regret,
            units=dict(unit=f.return_frame.unit,effects=list(f.return_frame.effects[0]),root_terms=f.return_frame.root_terms[0]),
            plans=[dict(root=f.roots[a['id']][0],action=a,purpose=f.purpose[a['id']],
                correction=None if means is None else list(means.corrections[0][means.plans.index(a['id'])]))
                for a in observation(f.contexts[0])['actions']])

    @staticmethod
    def feedback(fb):
        if fb.social_events: raise ValueError('social feedback has no native profile')
        if fb.event is not None and (not isinstance(fb.event,dict) or set(fb.event) != {'revealed_action'}):
            raise ValueError('one explicit actual public response required')
        return dict(observed=None if fb.observed is None else list(map(float,fb.observed)),
            response=None if fb.event is None else fb.event['revealed_action'],
            needs=None if fb.needs is None else dict(progress=[fb.needs.get(k) for k in NEEDS],
                maintained=[k in fb.maintained for k in NEEDS],completed=[k in fb.completed for k in NEEDS]),
            purpose=None if fb.purpose is None else asdict(fb.purpose))
